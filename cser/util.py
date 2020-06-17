import csv
import json
import os
import torch

from cser.dataset import Span


def create_directorie(d):
    if d and not os.path.exists(d):
        os.makedirs(d)

    return d


def save_config(log_path, cfg, filename):
    dic = convert_config_to_dic(cfg)
    path = os.path.join(log_path, '%s.json' % filename)
    with open(path, 'w') as f:
        json.dump(dic, f)


def convert_config_to_dic(cfg):
    d = dict(cfg._sections)
    for k in d:
        d[k] = dict(d[k])

    return d


def export_csv(path, headers, rows):
    try:
        with open(path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, dialect='excel')
            writer.writerow(headers)
            writer.writerows(rows)
    except Exception as e:
        print("Write an CSV file to path: %s, Case: %s" % (path, e))


def padded_stack(tensors, padding=0):
    dim_count = len(tensors[0].shape)

    max_shape = [max([t.shape[d] for t in tensors]) for d in range(dim_count)]
    padded_tensors = []

    for t in tensors:
        e = extend_tensor(t, max_shape, fill=padding)
        padded_tensors.append(e)

    stacked = torch.stack(padded_tensors)
    return stacked


def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :
                        tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :
                        tensor_shape[3]] = tensor

    return extended_tensor


def get_span_tokens(tokens, span):
    inside = False
    span_tokens = []

    for t in tokens:
        if t.span[0] == span[0]:
            inside = True

        if inside:
            span_tokens.append(t)

        if inside and t.span[1] == span[1]:
            return Span(span_tokens)

    return None


def to_device(batch, device):
    converted_batch = dict()
    for key in batch.keys():
        converted_batch[key] = batch[key].to(device)

    return converted_batch


def swap(v1, v2):
    return v2, v1
