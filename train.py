import torch.utils
from train_utils import EpochType
from log import create_logger
import torch
import time
import os
import math

from collections import defaultdict, namedtuple

import push

import json

from torch.utils.data import DataLoader

from pathlib import Path
import shutil
from model import ProtoTSNet

import numpy as np

from deepproblog.model import Model as DPLModel, Term
from deepproblog.train import TrainObject as DPLTrainObject
from deepproblog.dataset import DataLoader as DPLDataLoader
from deepproblog.utils.confusion_matrix import ConfusionMatrix
from deepproblog.semiring import Result
from deepproblog.query import Query

from statistics import mean

from contextlib import contextmanager

from typing import Optional, List, Callable

ProtoTSCoeffs = namedtuple('ProtoTSCoeffs', 'dpl_loss,l1_addon,sep,clst,l1', defaults=(1,0,0,0,0))

def experiment_setup(experiment_dir):
    os.makedirs(experiment_dir, exist_ok=True)

    shutil.copy(src=Path.cwd()/'autoencoder.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'datasets_utils.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'dpl.ipynb', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'main.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'model.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'push.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'train_utils.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'train.py', dst=experiment_dir)
    shutil.copy(src=Path.cwd()/'proto_logic.pl', dst=experiment_dir)
    
    return experiment_dir

def train_prototsnet_DPL(
    dpl_model: DPLModel,
    ptsnet: ProtoTSNet,
    experiment_dir,
    device,
    coeffs,
    train_dataset,
    train_loader: DataLoader,
    test_loader: DataLoader,
    class_specific,
    num_warm_epochs,
    push_start_epoch,
    push_epochs,
    num_logic_only_epochs,
    num_epochs=1000,
    learning_rates=None,
    features_lr=1e-3,  # effective only if learning_rates is None
    lr_sched_setup=None,
    loss_function=None,
    pos_weight=1,
    neg_weight=1,
    probab_threshold=0.5,
    loss_uses_negatives=False,
    custom_hooks=None,
    early_stopper=None,
    log=None
):
    save_files = True if experiment_dir is not None else False
    logclose = lambda: None
    if save_files:
        if not isinstance(experiment_dir, Path):
            experiment_dir = Path(experiment_dir)
        experiment_setup(experiment_dir)
        models_dir = experiment_dir / 'models'
        os.makedirs(models_dir, exist_ok=True)
        proto_dir = experiment_dir / 'protos'
        os.makedirs(proto_dir, exist_ok=True)
        if log is None:
            log, logclose = create_logger(experiment_dir / "log.txt", display=True)
    else:
        models_dir = None
        proto_dir = None
        log = print

    if learning_rates is None:
        learning_rates = {
            EpochType.JOINT: {
                'features': features_lr,
                'add_on_layers': 1e-3,
                'prototype_vectors': 1e-3
            },
            EpochType.WARM: {
                'add_on_layers': 1e-3,
                'prototype_vectors': 1e-3
            }
        }

    try:
        params = {
            "proto_num": ptsnet.prototype_shape[0],
            "proto_features": ptsnet.prototype_shape[1],
            "proto_len_latent": ptsnet.prototype_shape[2],
            "coeffs": coeffs._asdict(),
            "num_warm_epochs": num_warm_epochs,
            "push_start_epoch": push_start_epoch,
            "epochs": num_epochs,
            "learning_rates": str(learning_rates),
            "loss_uses_negatives": loss_uses_negatives,
        }
        with open(experiment_dir / "params.json", "w") as f:
            json.dump(params, f, indent=4)

        trainer = ProtoTSNetTrainer(
            dpl_model,
            ptsnet,
            device,
            train_dataset,
            train_loader,
            test_loader,
            class_specific=class_specific,
            num_epochs=num_epochs,
            num_warm_epochs=num_warm_epochs,
            push_start_epoch=push_start_epoch,
            push_epochs=push_epochs,
            num_logic_only_epochs=num_logic_only_epochs,
            coeffs=coeffs,
            learning_rates=learning_rates,
            lr_sched_setup=lr_sched_setup,
            loss_function=loss_function,
            pos_weight=pos_weight,
            neg_weight=neg_weight,
            probab_threshold=probab_threshold,
            loss_uses_negatives=loss_uses_negatives,
            proto_save_dir=proto_dir,
            early_stopper=early_stopper,
            hooks=[
                lambda t, _: t.dump_stats(experiment_dir / "stats.json"),
                get_verbose_logger()
            ] + ([] if custom_hooks is None else custom_hooks),
            log=log,
        )

        if save_files:
            shutil.rmtree(proto_dir)
            os.makedirs(proto_dir, exist_ok=True)

        trainer.train()

        if save_files:
            dpl_model.save_state(models_dir / 'last-epoch.pth')
            trainer.dump_stats(experiment_dir / 'stats.json')

        return trainer
    finally:
        logclose()


def best_stat_saver(stat_name, file_path, mode='min'):
    best = None
    push_best = None
    can_save_overal = False
    cmp = lambda a,b: a < b if mode == 'min' else a > b
    def save_best_stat(t, _):
        nonlocal best, push_best, can_save_overal

        found_better = False
        curr_loss = t.latest_stat(stat_name, raw=True)
        if t.curr_epoch_type in [EpochType.PUSH]:
            can_save_overal = True
            if push_best is None or cmp(curr_loss['value'], push_best['value']):
                push_best = curr_loss
                found_better = True

        if best is None or cmp(curr_loss['value'], best['value']):
            best = curr_loss
            found_better = True

        if found_better:
            with open(file_path, 'w') as f:
                json.dump({
                    'push': push_best,
                    'overall': best if can_save_overal else None
                }, f)
    return save_best_stat


def get_verbose_logger():
    def verbose_log_epoch(t: ProtoTSNetTrainer, _):
        t.log(f"Epoch stats:")
        if t.curr_epoch_type != EpochType.PUSH:
            t.log(f"    {'train accu:':25s} {t.latest_stat('accu_train')*100:.2f}%")
            t.log(f"    {'train overall loss:':25s} {t.latest_stat('loss_train')}")
            t.log(f"    {'train dpl cross ent:':25s} {t.latest_stat('dpl_loss_train')}")
            t.log(f"    {'train avg probab true:':25s} {t.latest_stat('avg_probab_for_positive_train')}")
            t.log(f"    {'train avg probab false:':25s} {t.latest_stat('avg_probab_for_negative_train')}")
            # t.log(f"    {'test overall loss:':25s} {t.latest_stat('loss_test')}")
            # t.log(f"    {'test cross_ent loss:':25s} {t.latest_stat('cross_ent_test')}")
            # t.log(f"    {'cluster loss:':25s} {t.latest_stat('cluster_test')}")
            # t.log(f"    {'avg separation loss:':25s} {t.latest_stat('avg_separation_test')}")
            t.log(f"    {'cluster loss:':25s} {t.latest_stat('cluster_train')}")
            if t.class_specific:
                t.log(f"    {'separation loss:':25s} {t.latest_stat('separation_train')}")
            t.log(f"    {'l1 loss:':25s} {t.latest_stat('l1_loss_train')}")
            t.log(f"    {'l1_addon loss:':25s} {t.latest_stat('l1_addon_train')}")
            t.log(f"    {'train time:':25s} {t.latest_stat('time_train')}")
        if t.latest_stat('did_run_test'):
            t.log(f"    {'test time:':25s} {t.latest_stat('time_test')}")
            t.log(f"    {'test accu:':25s} {t.latest_stat('accu_test')*100:.2f}%")
            t.log(f"    {'test avg probab true:':25s} {t.latest_stat('avg_probab_for_positive_test')}")
            t.log(f"    {'test avg probab false:':25s} {t.latest_stat('avg_probab_for_negative_test')}")
        t.log(f"    {'epoch time:':25s} {t.latest_stat('epoch_time')}", flush=True)
        if t.curr_epoch_type == EpochType.JOINT:
            joint_lr = t.latest_stat('joint_lr')
            if joint_lr is not None:
                t.log(f"    {'joint lr:':25s} {joint_lr}", flush=True)
    return verbose_log_epoch


class ProtoTSNetTrainer:
    def __init__(
        self,
        model: DPLModel,
        ptsnet: ProtoTSNet,
        device,
        train_dataset,
        train_loader: DataLoader,
        test_loader: DataLoader,
        class_specific,
        num_epochs,
        num_warm_epochs,
        push_start_epoch,
        push_epochs,
        num_logic_only_epochs,
        coeffs: ProtoTSCoeffs,
        learning_rates,
        lr_sched_setup=None,
        loss_function=None,
        pos_weight=1,
        neg_weight=1,
        probab_threshold=0.5,
        loss_uses_negatives=False,
        proto_save_dir=None,
        early_stopper=None,
        hooks=None,
        log=print,
    ):
        self.model = model
        self.ptsnet = ptsnet
        self.device = device
        self.ptsnet.to(self.device)

        self.class_specific = class_specific

        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.optimizers = {}
        self.joint_lr_scheduler = None

        self.proto_save_dir = proto_save_dir

        self.early_stopper = early_stopper

        self.loss_function = loss_function
        if self.loss_function is None:
            self.loss_function = self.cross_entropy
        self.loss_uses_negatives = loss_uses_negatives
        self.positive_weight = pos_weight
        self.negative_weight = neg_weight
        self.threshold = probab_threshold

        self.hooks = hooks
        if self.hooks is None:
            self.hooks = []

        self.num_train_epochs = num_epochs
        self.num_warm_epochs = num_warm_epochs
        self.push_start = push_start_epoch
        self.did_st_push = False
        self.push_epochs = push_epochs
        self.num_logic_only_epochs = num_logic_only_epochs

        self.coeffs = coeffs._asdict()

        self._stats = defaultdict(list)
        self.curr_epoch_type = None
        self.curr_epoch = 1
        self.curr_true_epoch = 1
        self.logic_only_epoch = 1
        self.run_type_str = None  # 'train', 'test', 'val'

        self.log = log

        self._setup_optimizers(learning_rates, lr_sched_setup)

    def _setup_optimizers(self, learning_rates, lr_sched_setup):
        joint_optimizer_specs = [
            {'params': self.ptsnet.features.parameters(), 'lr': learning_rates[EpochType.JOINT]['features']},
            {'params': self.ptsnet.add_on_layers.parameters(), 'lr': learning_rates[EpochType.JOINT]['add_on_layers']},
            {'params': self.ptsnet.prototype_vectors, 'lr': learning_rates[EpochType.JOINT]['prototype_vectors']},
        ]
        warm_optimizer_specs = [
            {'params': self.ptsnet.add_on_layers.parameters(), 'lr': learning_rates[EpochType.WARM]['add_on_layers']},
            {'params': self.ptsnet.prototype_vectors, 'lr': learning_rates[EpochType.WARM]['prototype_vectors']},
        ]

        self.optimizers[EpochType.JOINT] = torch.optim.Adam(joint_optimizer_specs)
        self.optimizers[EpochType.WARM] = torch.optim.Adam(warm_optimizer_specs)

        if lr_sched_setup is not None:
            self.joint_lr_scheduler = lr_sched_setup(self.optimizers[EpochType.JOINT])

    def _add_stat(self, stat_name, value):
        if self.run_type_str is not None and not stat_name.endswith(self.run_type_str):
            stat_name = f'{stat_name}_{self.run_type_str}'

        stat = {
            'true_epoch': self.curr_true_epoch,
            'epoch': self.curr_epoch,
            'epoch_type': self.curr_epoch_type.name,
            'value': value,
        }
        self._stats[stat_name].append(stat)

    def latest_stat(self, stat_name, raw=False):
        if stat_name not in self._stats:
            return None
        s = self._stats[stat_name][-1]
        return s if raw else s.get('value')

    def latest_stats(self):
        return {n: lst[-1] for n, lst in self._stats.items()}

    def _latest_stats_values(self):
        return {stat: vals[-1].get('value') for stat, vals in self._stats.items()}

    def stats(self):
        return dict(self._stats)

    def dump_stats(self, path):
        tmp_path = str(path) + '.tmp'
        with open(tmp_path, 'w') as f:
            json.dump(self._stats, f)
        os.rename(tmp_path, path)

    def _call_hooks(self):
        for hook in self.hooks:
            hook(self, self.ptsnet)

    @contextmanager
    def report_epoch_summary(self):
        start = time.time()
        yield
        end = time.time()
        self._add_stat('epoch_time', end - start)
        if self.joint_lr_scheduler is not None:
            try:
                self._add_stat('joint_lr', self.joint_lr_scheduler._last_lr[0])
            except AttributeError:
                pass

    def _warm_epoch(self):
        self.log(f"epoch: {self.curr_epoch:3d} (WARM)")
        with self.report_epoch_summary():
            self._set_epoch_type(EpochType.WARM)

            self._single_train_round()
            # self._single_validation_round()
            self._single_test_round()

    def _joint_epoch(self):
        self.log(f"epoch: {self.curr_epoch:3d} (JOINT)")
        with self.report_epoch_summary():
            self._set_epoch_type(EpochType.JOINT)

            self._single_train_round()
            # self._single_validation_round()
            self._single_test_round()

            if self.joint_lr_scheduler is not None:
                self.joint_lr_scheduler.step()

    def _logic_only_epoch(self):
        self.log(f"epoch: {self.curr_epoch:3d} (LOGIC ONLY: {self.logic_only_epoch}/{self.num_logic_only_epochs})")
        with self.report_epoch_summary():
            self._set_epoch_type(EpochType.LOGIC_ONLY)

            self._single_train_round()
            self._single_test_round()

    def _push_protos(self):
        self.log(f"epoch: {self.curr_epoch:3d} (PUSH)")
        self.curr_true_epoch += 1
        with self.report_epoch_summary():
            # require_grad does not matter here, as we are not training, only pushing protos and testing
            self._set_epoch_type(EpochType.PUSH)

            prototype_ts_filename_prefix = 'prototype-ts'
            prototype_self_act_filename_prefix = 'prototype-self-act'
            proto_bounds_filename_prefix = 'bounds'

            push.push_prototypes(
                torch.utils.data.DataLoader(self.train_dataset.get_for_torch_dl(), batch_size=self.train_loader.batch_size, shuffle=True),
                prototype_network=self.ptsnet,
                class_specific=self.class_specific,
                preprocess_input_function=None,
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=self.proto_save_dir,
                epoch_number=self.curr_epoch,
                proto_ts_filename_prefix=prototype_ts_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bounds_filename_prefix,
                save_prototype_class_identity=True,
                device=self.device)

            # self._single_validation_round()
            self._single_test_round()

    def _set_epoch_type(self, epoch_type: EpochType):
        self.curr_epoch_type = epoch_type
        if hasattr(self.ptsnet.features, 'set_requires_grad'):
            self.ptsnet.features.set_requires_grad(epoch_type in [EpochType.JOINT])
        else:
            for p in self.ptsnet.features.parameters():
                p.requires_grad = epoch_type in [EpochType.JOINT]
        for p in self.ptsnet.add_on_layers.parameters():
            p.requires_grad = epoch_type in [EpochType.JOINT, epoch_type.WARM]
        self.ptsnet.prototype_vectors.requires_grad = epoch_type in [EpochType.JOINT, epoch_type.WARM]

    def _single_train_round(self):
        self.run_type_str = 'train'
        self.model.train()
        self._train(self.train_loader, optimizer=self.optimizers.get(self.curr_epoch_type))
        self.run_type_str = None

    def _single_test_round(self):
        self.run_type_str = 'test'
        if not self._test_round_cond():
            self._add_stat('did_run', False)
            self.run_type_str = None
            return
        self._add_stat('did_run', True)
        self.model.eval()
        self._test(self.test_loader)
        self.run_type_str = None

    def _test_round_cond(self):
        if self.curr_epoch_type == EpochType.LOGIC_ONLY:
            return self.logic_only_epoch % 5 == 0
        else:
            return self.curr_epoch % 5 == 0

    def _st_push_condition(self):
        return self.curr_epoch >= self.push_start

    def _warm_epoch_condition(self):
        return self.curr_epoch <= self.num_warm_epochs

    def get_loss(self, batch: List[Query], loss_fn: Callable, pos_weight, neg_weight) -> torch.Tensor:
        """
        Calculates and propagates the loss for a given batch of queries and loss function.
        :param batch: The batch of queries.
        :param loss_fn: The loss function.
        :return: The average loss over the batch
        """
        total_loss = 0
        results = self.model.solve(batch)
        results = [
            (results[i], batch[i]) for i in range(len(batch)) if len(results[i]) > 0
        ]
        for r, q in results:
            curr_loss = loss_fn(
                r, q.p, weight=1 / len(results), q=q.substitute().query
            )
            # print(f'curr_loss: {curr_loss}, q.p: {q.p}, curr_loss * weight: {curr_loss * (pos_weight if q.p == 1 else neg_weight)}')
            total_loss += curr_loss * (pos_weight if q.p == 1 else neg_weight)
        return total_loss, results

    def get_loss_with_negatives(
        self, batch: List[Query], loss_fn: Callable
    ) -> torch.Tensor:
        """
        Calculates and propagates the loss for a given batch of queries and loss function.
        This includes negative examples. Negative examples are found by using the query.replace_var method.
        :param batch: The batch of queries.
        :param loss_fn: The loss function.
        :return: The average loss over the batch
        """
        total_loss = 0

        results = self.model.solve([q.variable_output() for q in batch])
        results = [(results[i], batch[i]) for i in range(len(batch))]

        for r, q in results:
            expected = q.substitute().query
            try:
                total_loss += loss_fn(
                    r, q.p, weight=1 / len(results), q=expected
                )
            except KeyError:
                self.get_loss([q], loss_fn)
            neg_proofs = [x for x in r if x != expected]
            for neg in neg_proofs:
                # print('penalizing wrong answer {} vs {}'.format(q.substitute().query, k))
                total_loss += loss_fn(
                    r, 0, weight=1 / (len(results) * len(neg_proofs)), q=neg
                )
        return total_loss, results

    @staticmethod
    def cross_entropy(
        result: Result,
        target: float,
        weight: float,
        q: Optional[Term] = None,
        eps: float = 1e-12,
    ) -> float:
        """
        This method is copied from deepproblog.semiring.graph_semiring.GraphSemiring
        The only difference is lack of loss.backward() call, it is done in the _train method.
        """

        result = result.result
        if len(result) == 0:
            print("No results found for {}".format(q))
            return 0
        if q is None:
            if len(result) == 1:
                q, p = next(iter(result.items()))
            else:
                raise ValueError(
                    "q is None and number of results is {}".format(len(result))
                )
        else:
            p = result[q]
        if type(p) is float:
            loss = (
                -(target * math.log(p + eps) + (1.0 - target) * math.log(1.0 - p + eps))
                * weight
            )
        else:
            if target == 1.0:
                loss = -torch.log(p + eps) * weight
            elif target == 0.0:
                loss = -torch.log(1.0 - p + eps) * weight
            else:
                loss = (
                    -(
                        target * torch.log(p + eps)
                        + (1.0 - target) * torch.log(1.0 - p + eps)
                    )
                    * weight
                )
        return loss

    def _train(self, dataloader, optimizer):
        start = time.time()
        n_examples = 0
        n_correct = 0
        n_batches = 0
        total_dpl_loss = 0
        total_l1_loss = 0
        total_cluster_cost = 0
        # separation cost is meaningful only for class_specific
        total_separation_cost = 0
        total_avg_separation_cost = 0
        total_loss = 0
        probab_for_positive = []
        probab_for_negative = []
        self.model.optimizer.step_epoch()

        for batch in dataloader:
            queries_batch, ts_batch, label = [q for q, *_ in batch], [ts for _, ts, _ in batch], [l for *_, l in batch]
            # input = image.to(self.device)
            # target = label.to(self.device)

            with torch.enable_grad():
                # FIXME: only first element of batch is taken
                min_distances = self.ptsnet.min_distances(ts_batch[0].view(1, *ts_batch[0].shape))

                if self.loss_uses_negatives:
                    dpl_loss, results = self.get_loss_with_negatives(queries_batch, self.loss_function)
                else:
                    dpl_loss, results = self.get_loss(queries_batch, self.loss_function, self.positive_weight, self.negative_weight)

                if self.class_specific:
                    max_dist = (self.ptsnet.prototype_shape[1]
                                * self.ptsnet.prototype_shape[2])

                    # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                    # calculate cluster cost
                    prototypes_of_correct_class = torch.t(self.ptsnet.prototype_class_identity[:,label]).to(self.device)
                    inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=0)
                    cluster_cost = torch.mean(max_dist - inverted_distances)

                    # calculate separation cost
                    prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                    inverted_distances_to_nontarget_prototypes, _ = \
                        torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=0)
                    separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                    # calculate avg cluster cost
                    avg_separation_cost = \
                        torch.sum(min_distances * prototypes_of_wrong_class, dim=0) / torch.sum(prototypes_of_wrong_class, dim=0)
                    avg_separation_cost = torch.mean(avg_separation_cost)

                    total_separation_cost += separation_cost.item()
                    total_avg_separation_cost += avg_separation_cost.item()
                else:
                    min_distance, _ = torch.min(min_distances, dim=1)
                    cluster_cost = torch.mean(min_distance)

                l1_addon = self.ptsnet.add_on_layers[0].weight.norm(p=1)

                # print(self.model.tensor_parameters)
                if self.model.tensor_parameters:
                    l1 = torch.norm(torch.stack(self.model.tensor_parameters), p=2)
                else:
                    l1 = torch.tensor(0.0)
                # print(l1)

                # evaluation statistics
                # _, predicted = torch.max(output.data, 1)
                # n_examples += target.size(0)
                # n_correct += (predicted == target).sum().item()
                n_examples += len(queries_batch)
                for r, q in results:
                    expected = q.substitute().query
                    if q.p == 1:
                        probab_for_positive.append(r.result[expected].item())
                        if r.result[expected] >= self.threshold:
                            n_correct += 1
                    elif q.p == 0:
                        # if type(r.result[expected]) is float:
                        #     print(self.model.parameters)
                        probab_for_negative.append(r.result[expected].item())
                        if r.result[expected] < self.threshold:
                            n_correct += 1
                    if self.loss_uses_negatives:
                        neg_proofs = [x for x in r if x != expected]
                        for neg in neg_proofs:
                            n_examples += 1
                            probab_for_negative.append(r.result[neg].item())
                            if r.result[neg] < self.threshold:
                                n_correct += 1

                n_batches += 1
                total_dpl_loss += dpl_loss.item()
                total_cluster_cost += cluster_cost.item()

            # compute gradient and do SGD step
            if self.class_specific:
                if self.coeffs is not None:
                    loss = (self.coeffs['dpl_loss'] * dpl_loss
                        + self.coeffs['clst'] * cluster_cost
                        + self.coeffs['sep'] * separation_cost
                        + self.coeffs['l1_addon'] * l1_addon)
                else:
                    loss = dpl_loss + 0.8 * cluster_cost - 0.08 * separation_cost
            else:
                if self.coeffs is not None:
                    loss = (self.coeffs['dpl_loss'] * dpl_loss
                        + self.coeffs['clst'] * cluster_cost
                        + self.coeffs['l1_addon'] * l1_addon)
                else:
                    loss = dpl_loss + 0.8 * cluster_cost

            loss += self.coeffs['l1'] * l1
            total_l1_loss += l1.item()
            total_loss += loss.item()
            if optimizer is not None:
                optimizer.zero_grad()
            self.model.optimizer.zero_grad()
            loss.backward()
            # if self.curr_epoch_type == EpochType.LOGIC_ONLY:
            #     self.log(f"gradients: {self.model.optimizer._params_grad}")
            if optimizer is not None:
                optimizer.step()
            # if self.curr_epoch_type != EpochType.LOGIC_ONLY:
            self.model.optimizer.step()

            # del input
            # del target
            # del output
            # del predicted
            # del min_distances

        end = time.time()

        self._add_stat('time', end - start)
        self._add_stat('loss', total_loss / n_batches)
        self._add_stat('dpl_loss', total_dpl_loss / n_batches)
        self._add_stat('l1_loss', total_l1_loss / n_batches)
        self._add_stat('cluster', total_cluster_cost / n_batches)
        if self.class_specific:
            self._add_stat('separation', total_separation_cost / n_batches)
            self._add_stat('avg_separation', total_avg_separation_cost / n_batches)
        self._add_stat('accu', n_correct / n_examples)
        self._add_stat('avg_probab_for_positive', mean(probab_for_positive) if len(probab_for_positive) > 0 else 0)
        self._add_stat('avg_probab_for_negative', mean(probab_for_negative) if len(probab_for_negative) > 0 else 0)
        self._add_stat('l1_addon', self.ptsnet.add_on_layers[0].weight.norm(p=1).item())

    def _test(self, dataloader: DPLDataLoader):
        self.log(f"############# {'Test round':25s} #############")

        start = time.time()
        
        threshold = self.threshold
        empty_answer = "false"

        confusion_matrix = ConfusionMatrix()
        probabs_per_pred = defaultdict(list)
        probabs_per_true = defaultdict(list)
        for batch in dataloader:
            batch = [q for q, *_ in batch]
            for query in batch:
                answer = self.model.solve([query])[0]
                if len(answer.result) == 0:
                    predicted = empty_answer
                    self.log(f"No answer for query {query}")
                else:
                    predicted_p = float(answer.result[query.substitute().query])
                    predicted = "true" if predicted_p >= threshold else "false"
                    probabs_per_pred[predicted].append(predicted_p)
                    probabs_per_true["true" if query.p == 1 else "false"].append(predicted_p)
                actual = "true" if query.p >= threshold else "false"
                confusion_matrix.add_item(predicted, actual)
        # for c in probabilities:
        #     self.log(f"Average probability for class {c}: {mean(probabilities[c])}")

        self.log("Confusion matrix:")
        self.log(str(confusion_matrix))
        acc = confusion_matrix.accuracy()
        # self.log(f"Accuracy: {acc}")

        end = time.time()

        self._add_stat('avg_probab_for_positive', mean(probabs_per_true['true']) if len(probabs_per_true['true']) > 0 else 0)
        self._add_stat('avg_probab_for_negative', mean(probabs_per_true['false']) if len(probabs_per_true['false']) > 0 else 0)
        self._add_stat('accu', acc)
        self._add_stat('time', end - start)

    def train(self):
        self.log('Starting training')
        t_start = time.time()

        while self.curr_epoch <= self.num_train_epochs:
            if self._warm_epoch_condition():
                self._warm_epoch()
            else:
                self._joint_epoch()
            self._call_hooks()

            if self._st_push_condition() and self.curr_epoch in self.push_epochs:
                self.did_st_push = True
                self._push_protos()
                self._call_hooks()

                self.log(f"############# {'Adjusting logic probabs':25s} #############")
                for self.logic_only_epoch in range(1, self.num_logic_only_epochs+1):
                    self._logic_only_epoch()
                    self._call_hooks()

            if self.early_stopper and self.early_stopper(self):
                self.log(f'Early stopping condition met on epoch {self.curr_epoch}, aborting')
                break

            self.curr_epoch += 1
            self.curr_true_epoch += 1

        t_end = time.time()
        self.log(f'Finished training in {t_end - t_start:.2f} seconds')
        self._add_stat('total_time', t_end - t_start)
