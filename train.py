from train_utils import EarlyStopping, EpochType
import torch
import time
import os

from collections import defaultdict, namedtuple

import push

import json

from torch.utils.data import DataLoader

from pathlib import Path
import os
import torch
import shutil
from model import ProtoTSNet
import json
from datasets_utils import TrainTestDS
from train_utils import BestModelCheckpointer

from deepproblog.model import Model as DPLModel
from deepproblog.train import TrainObject as DPLTrainObject

from contextlib import contextmanager

ProtoTSCoeffs = namedtuple('ProtoTSCoeffs', 'crs_ent,clst,sep,l1,l1_addon', defaults=(1,0,0,0,0))


def train_prototsnet_DPL(
    model: DPLModel,
    ptsnet: ProtoTSNet,
    experiment_dir,
    device,
    coeffs,
    train_loader: DataLoader,
    test_loader: DataLoader,
    val_loader: DataLoader = None,
    num_epochs=1000,
    num_warm_epochs=50,
    push_start_epoch=40,
    push_epochs=None,
    learning_rates=None,
    features_lr=1e-3,  # effective only if learning_rates is None
    lr_sched_setup=None,
    custom_hooks=None,
    early_stopper=None,
    log=print,
):
    save_files = True if experiment_dir is not None else False
    if save_files:
        if not isinstance(experiment_dir, Path):
            experiment_dir = Path(experiment_dir)
        models_dir = experiment_dir / 'models'
        os.makedirs(models_dir, exist_ok=True)
        proto_dir = experiment_dir / 'protos'
        os.makedirs(proto_dir, exist_ok=True)
    else:
        models_dir = None
        proto_dir = None

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

    if push_epochs is None:
        push_epochs = range(0, num_epochs, 20)

    overall_best_checkpointer = BestModelCheckpointer(
        stat_name="cross_ent_val",
        mode="min",
        model_save_path=models_dir,
        model_name="overall_best",
    )
    push_only_best_checkpointer = BestModelCheckpointer(
        stat_name="cross_ent_val",
        mode="min",
        add_save_cond=lambda t, _: t.curr_epoch_type in [EpochType.PUSH],
        model_save_path=models_dir,
        model_name="push_best",
    )
    trainer = ProtoTSNetTrainer(
        model,
        ptsnet,
        device,
        train_loader,
        test_loader,
        val_loader,
        num_epochs=num_epochs,
        num_warm_epochs=num_warm_epochs,
        push_start_epoch=push_start_epoch,
        push_epochs=push_epochs,
        coeffs=coeffs,
        learning_rates=learning_rates,
        lr_sched_setup=lr_sched_setup,
        proto_save_dir=proto_dir,
        early_stopper=early_stopper,
        hooks=[
            (overall_best_checkpointer if save_files and val_loader is not None else lambda _: None),
            (push_only_best_checkpointer if save_files and val_loader is not None else lambda _: None),
            lambda t, _: t.dump_stats(experiment_dir / "stats.json"),
        ] + ([] if custom_hooks is None else custom_hooks),
        log=log,
    )

    if save_files:
        shutil.rmtree(proto_dir)
        os.makedirs(proto_dir, exist_ok=True)

    trainer.train()

    model.save_state(models_dir / 'last-epoch.pth')
    trainer.dump_stats(experiment_dir / 'stats.json')

    return trainer


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


def get_verbose_logger(dataset_name):
    def verbose_log_epoch(t: ProtoTSNetTrainer, _):
        if t.val_loader is not None:
            t.log(f"epoch: {t.curr_epoch:3d} ({t.curr_epoch_type.name}) - {dataset_name}")
            t.log(f"    {'val acc:':25s} {t.latest_stat('accu_val')*100:.2f}%")
            t.log(f"    {'train overall loss:':25s} {t.latest_stat('loss_train')}")
            t.log(f"    {'train cross_ent loss:':25s} {t.latest_stat('cross_ent_train')}")
            t.log(f"    {'val overall loss:':25s} {t.latest_stat('loss_val')}")
            t.log(f"    {'val cross_ent loss:':25s} {t.latest_stat('cross_ent_val')}")
            t.log(f"    {'cluster loss:':25s} {t.latest_stat('cluster_val')}")
            t.log(f"    {'separation loss:':25s} {t.latest_stat('separation_val')}")
            t.log(f"    {'avg separation loss:':25s} {t.latest_stat('avg_separation_val')}")
            t.log(f"    {'l1_addon loss:':25s} {t.latest_stat('l1_addon_val')}")
            t.log(f"    {'train time:':25s} {t.latest_stat('time_train')}")
            t.log(f"    {'val time:':25s} {t.latest_stat('time_val')}")
            t.log(f"    {'epoch time:':25s} {t.latest_stat('epoch_time')}", flush=True)
            if t.curr_epoch_type == EpochType.JOINT:
                t.log(f"    {'joint lr:':25s} {t.latest_stat('joint_lr')}", flush=True)
            if t.latest_stat('did_run_test'):
                t.log(f"    Testing:")
                t.log(f"    {'test acc:':25s} {t.latest_stat('accu_test')*100:.2f}%")
                t.log(f"    {'test overall loss:':25s} {t.latest_stat('loss_test')}")
                t.log(f"    {'test cross_ent loss:':25s} {t.latest_stat('cross_ent_test')}")
                t.log(f"    {'test time:':25s} {t.latest_stat('time_test')}")
        else:
            t.log(f"epoch: {t.curr_epoch:3d} ({t.curr_epoch_type.name}) - {dataset_name}")
            t.log(f"    {'test acc:':25s} {t.latest_stat('accu_test')*100:.2f}%")
            t.log(f"    {'train overall loss:':25s} {t.latest_stat('loss_train')}")
            t.log(f"    {'train cross_ent loss:':25s} {t.latest_stat('cross_ent_train')}")
            t.log(f"    {'test overall loss:':25s} {t.latest_stat('loss_test')}")
            t.log(f"    {'test cross_ent loss:':25s} {t.latest_stat('cross_ent_test')}")
            t.log(f"    {'cluster loss:':25s} {t.latest_stat('cluster_test')}")
            t.log(f"    {'separation loss:':25s} {t.latest_stat('separation_test')}")
            t.log(f"    {'avg separation loss:':25s} {t.latest_stat('avg_separation_test')}")
            t.log(f"    {'l1_addon loss:':25s} {t.latest_stat('l1_addon_test')}")
            t.log(f"    {'train time:':25s} {t.latest_stat('time_train')}")
            t.log(f"    {'test time:':25s} {t.latest_stat('time_test')}")
            t.log(f"    {'epoch time:':25s} {t.latest_stat('epoch_time')}", flush=True)
            if t.curr_epoch_type == EpochType.JOINT:
                t.log(f"    {'joint lr:':25s} {t.latest_stat('joint_lr')}", flush=True)
    return verbose_log_epoch


class ProtoTSNetTrainer(DPLTrainObject):
    def __init__(
        self,
        model: DPLModel,
        ptsnet: ProtoTSNet,
        device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs,
        num_warm_epochs,
        push_start_epoch,
        push_epochs,
        coeffs: ProtoTSCoeffs,
        learning_rates,
        lr_sched_setup=None,
        proto_save_dir=None,
        early_stopper=None,
        hooks=None,
        log=print,
    ):
        self.model = model
        self.ptsnet = ptsnet
        self.device = device
        # self.ptsnet.to(self.device)

        self.class_specific = True

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        self.optimizers = {}

        self.proto_save_dir = proto_save_dir

        self.early_stopper = early_stopper

        self.hooks = hooks
        if self.hooks is None:
            self.hooks = []

        self.num_train_epochs = num_epochs
        self.num_warm_epochs = num_warm_epochs
        self.push_start = push_start_epoch
        self.did_st_push = False
        self.push_epochs = push_epochs

        self.coeffs = coeffs._asdict()

        self._stats = defaultdict(list)
        self.curr_epoch_type = None
        self.curr_epoch = 1
        self.curr_true_epoch = 1
        self.run_type_str = None  # 'train', 'test', 'val'

        self.log = log

        self.joint_lr_scheduler = None

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

    def _call_checkpointers(self):
        for hook in self.hooks:
            hook(self, self.ptsnet)

    @contextmanager
    def report_epoch_summary(self):
        start = time.time()
        yield
        end = time.time()
        self._add_stat('epoch_time', end - start)
        if self.joint_lr_scheduler is not None:
            self._add_stat('joint_lr', self.joint_lr_scheduler._last_lr[0])

    def _warm_epoch(self):
        with self.report_epoch_summary():
            self._set_epoch_type(EpochType.WARM)

            self._single_train_round()
            self._single_validation_round()
            self._single_test_round()

    def _joint_epoch(self):
        with self.report_epoch_summary():
            self._set_epoch_type(EpochType.JOINT)

            self._single_train_round()
            self._single_validation_round()
            self._single_test_round()

            if self.joint_lr_scheduler is not None:
                self.joint_lr_scheduler.step()

    def _push_protos(self):
        self.curr_true_epoch += 1
        with self.report_epoch_summary():
            # require_grad does not matter here, as we are not training, only pushing protos and testing
            self._set_epoch_type(EpochType.PUSH)

            prototype_ts_filename_prefix = 'prototype-ts'
            prototype_self_act_filename_prefix = 'prototype-self-act'
            proto_bounds_filename_prefix = 'bounds'

            push.push_prototypes(
                self.train_loader,
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

            self._single_validation_round()
            self._single_test_round()

    def _set_epoch_type(self, epoch_type: EpochType):
        self.curr_epoch_type = epoch_type
        self.ptsnet.features.set_requires_grad(epoch_type in [EpochType.JOINT])
        for p in self.ptsnet.add_on_layers.parameters():
            p.requires_grad = epoch_type in [EpochType.JOINT, epoch_type.WARM]
        self.ptsnet.prototype_vectors.requires_grad = epoch_type in [EpochType.JOINT, epoch_type.WARM]

    def _single_train_round(self):
        self.run_type_str = 'train'
        self.model.train()
        self._train_or_test(self.train_loader, optimizer=self.optimizers[self.curr_epoch_type])
        self.run_type_str = None

    def _single_validation_round(self):
        if self.val_loader is None:
            return
        self.run_type_str = 'val'
        self.model.eval()
        self._train_or_test(self.val_loader, optimizer=None)
        self.run_type_str = None

    def _single_test_round(self):
        self.run_type_str = 'test'
        if not self._test_round_cond():
            self._add_stat('did_run', False)
            self.run_type_str = None
            return
        self._add_stat('did_run', True)
        self.model.eval()
        self._train_or_test(self.test_loader, optimizer=None)
        self.run_type_str = None

    def _test_round_cond(self):
        return self.curr_epoch % 10 == 0

    def _train_or_test(self, dataloader, optimizer=None):
        is_train = optimizer is not None
        start = time.time()
        n_examples = 0
        n_correct = 0
        n_batches = 0
        total_cross_entropy = 0
        total_cluster_cost = 0
        # separation cost is meaningful only for class_specific
        total_separation_cost = 0
        total_avg_separation_cost = 0
        total_loss = 0

        for i, (image, label) in enumerate(dataloader):
            input = image.to(self.device)
            target = label.to(self.device)

            # torch.enable_grad() has no effect outside of no_grad()
            grad_req = torch.enable_grad() if is_train else torch.no_grad()
            with grad_req:
                output, min_distances = self.ptsnet(input)

                # compute loss
                cross_entropy = torch.nn.functional.cross_entropy(output, target)

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

                    l1_addon = self.ptsnet.add_on_layers[0].weight.norm(p=1)
                else:
                    min_distance, _ = torch.min(min_distances, dim=1)
                    cluster_cost = torch.mean(min_distance)

                # evaluation statistics
                _, predicted = torch.max(output.data, 1)
                n_examples += target.size(0)
                n_correct += (predicted == target).sum().item()

                n_batches += 1
                total_cross_entropy += cross_entropy.item()
                total_cluster_cost += cluster_cost.item()
                total_separation_cost += separation_cost.item()
                total_avg_separation_cost += avg_separation_cost.item()

            # compute gradient and do SGD step
            if self.class_specific:
                if self.coeffs is not None:
                    loss = (self.coeffs['crs_ent'] * cross_entropy
                        + self.coeffs['clst'] * cluster_cost
                        + self.coeffs['sep'] * separation_cost
                        + self.coeffs['l1_addon'] * l1_addon)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost
            else:
                if self.coeffs is not None:
                    loss = (self.coeffs['crs_ent'] * cross_entropy
                        + self.coeffs['clst'] * cluster_cost
                        + self.coeffs['l1_addon'] * l1_addon)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost

            total_loss += loss.item()
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            del input
            del target
            del output
            del predicted
            del min_distances

        end = time.time()

        self._add_stat('time', end - start)
        self._add_stat('loss', total_loss / n_batches)
        self._add_stat('cross_ent', total_cross_entropy / n_batches)
        self._add_stat('cluster', total_cluster_cost / n_batches)
        if self.class_specific:
            self._add_stat('separation', total_separation_cost / n_batches)
            self._add_stat('avg_separation', total_avg_separation_cost / n_batches)
        self._add_stat('accu', n_correct / n_examples)
        self._add_stat('l1_addon', self.ptsnet.add_on_layers[0].weight.norm(p=1).item())

    def _st_push_condition(self):
        return self.curr_epoch >= self.push_start

    def _warm_epoch_condition(self):
        return self.curr_epoch <= self.num_warm_epochs

    def train(self):
        self.log('Starting training')
        t_start = time.time()

        while self.curr_epoch <= self.num_train_epochs:
            if self._warm_epoch_condition():
                self._warm_epoch()
            else:
                self._joint_epoch()
            self._call_checkpointers()

            if self._st_push_condition() and self.curr_epoch in self.push_epochs:
                self.did_st_push = True
                self._push_protos()
                self._call_checkpointers()

            if self.early_stopper and self.early_stopper(self):
                self.log(f'Early stopping condition met on epoch {self.curr_epoch}, aborting')
                break

            self.curr_epoch += 1
            self.curr_true_epoch += 1

        t_end = time.time()
        self.log(f'Finished training in {t_end - t_start:.2f} seconds')
        self._add_stat('total_time', t_end - t_start)
