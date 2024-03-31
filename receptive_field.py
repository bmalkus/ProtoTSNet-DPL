import math

def compute_layer_rf_info(layer_filter_size, layer_stride, layer_padding,
                          previous_layer_rf_info):
    n_in = previous_layer_rf_info[0] # input size
    j_in = previous_layer_rf_info[1] # receptive field jump of input layer
    r_in = previous_layer_rf_info[2] # receptive field size of input layer
    start_in = previous_layer_rf_info[3] # center of receptive field of input layer

    if layer_padding == 'SAME':
        n_out = math.ceil(float(n_in) / float(layer_stride))
        if (n_in % layer_stride == 0):
            pad = max(layer_filter_size - layer_stride, 0)
        else:
            pad = max(layer_filter_size - (n_in % layer_stride), 0)
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    elif layer_padding == 'VALID':
        n_out = math.ceil(float(n_in - layer_filter_size + 1) / float(layer_stride))
        pad = 0
        assert(n_out == math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1) # sanity check
        assert(pad == (n_out-1)*layer_stride - n_in + layer_filter_size) # sanity check
    else:
        # layer_padding is an int that is the amount of padding on one side
        pad = layer_padding * 2
        n_out = math.floor((n_in - layer_filter_size + pad)/layer_stride) + 1

    pL = math.floor(pad/2)

    j_out = j_in * layer_stride
    r_out = r_in + (layer_filter_size - 1)*j_in
    start_out = start_in + ((layer_filter_size - 1)/2 - pL)*j_in
    return [n_out, j_out, r_out, start_out]

def compute_rf_protoL_at_spatial_location(ts_len, fmap_proto_idx, protoL_rf_info):
    n = protoL_rf_info[0]
    j = protoL_rf_info[1]
    r = protoL_rf_info[2]
    start = protoL_rf_info[3]
    assert(fmap_proto_idx < n)

    center = start + (fmap_proto_idx*j)

    rf_start_idx = max(int(center - (r/2)), 0)
    rf_end_idx = min(int(center + (r/2)), ts_len)

    return [rf_start_idx, rf_end_idx]

def compute_rf_prototype(ts_len, prototype_patch_index, protoL_rf_info):
    ts_idx = prototype_patch_index[0]
    fmap_proto_idx = prototype_patch_index[1]
    # print(ts_idx, fmap_proto_idx, ts_len, protoL_rf_info)
    rf_indices = compute_rf_protoL_at_spatial_location(ts_len,
                                                       fmap_proto_idx,
                                                       protoL_rf_info)
    return [ts_idx, rf_indices[0], rf_indices[1]]

def compute_rf_prototypes(ts_len, prototype_patch_indices, protoL_rf_info):
    rf_prototypes = []
    for prototype_patch_index in prototype_patch_indices:
        img_index = prototype_patch_index[0]
        height_index = prototype_patch_index[1]
        width_index = prototype_patch_index[2]
        rf_indices = compute_rf_protoL_at_spatial_location(ts_len,
                                                           height_index,
                                                           width_index,
                                                           protoL_rf_info)
        rf_prototypes.append([img_index, rf_indices[0], rf_indices[1]])
    return rf_prototypes

def compute_proto_layer_rf_info(ts_len, cfg, prototype_kernel_size):
    rf_info = [ts_len, 1, 1, 0.5]

    for v in cfg:
        if v == 'M':
            rf_info = compute_layer_rf_info(layer_filter_size=2,
                                            layer_stride=2,
                                            layer_padding='SAME',
                                            previous_layer_rf_info=rf_info)
        else:
            rf_info = compute_layer_rf_info(layer_filter_size=3,
                                            layer_stride=1,
                                            layer_padding='SAME',
                                            previous_layer_rf_info=rf_info)

    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=prototype_kernel_size,
                                                layer_stride=1,
                                                layer_padding='VALID',
                                                previous_layer_rf_info=rf_info)

    return proto_layer_rf_info

def compute_proto_layer_rf_info_v2(img_size, layer_filter_sizes, layer_strides, layer_paddings, prototype_kernel_size):
    assert(len(layer_filter_sizes) == len(layer_strides))
    assert(len(layer_filter_sizes) == len(layer_paddings))

    rf_info = [img_size, 1, 1, 0.5]

    for i in range(len(layer_filter_sizes)):
        filter_size = layer_filter_sizes[i]
        stride_size = layer_strides[i]
        padding_size = layer_paddings[i]
        rf_info = compute_layer_rf_info(layer_filter_size=filter_size,
                                layer_stride=stride_size,
                                layer_padding=padding_size,
                                previous_layer_rf_info=rf_info)

    proto_layer_rf_info = compute_layer_rf_info(layer_filter_size=prototype_kernel_size,
                                                layer_stride=1,
                                                layer_padding='VALID',
                                                previous_layer_rf_info=rf_info)

    return proto_layer_rf_info

