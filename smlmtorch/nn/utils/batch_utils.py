import torch
from torch.utils.data import default_collate

batch_stack = default_collate

def move_to_device(x, device):
    if type(x) == dict:
        return {k:v.to(device) for k,v in x.items()}
    else:
        return x.to(device)


def batch_cat(batch, dim=0):
    """
    Like default_collate but concatenates tensors instead of stacking them
    """
    assert len(batch) > 0, 'batch is empty'

    if type(batch[0]) == dict:
        # list of dicts -> dict of tensors
        return {k:batch_cat([b[k] for b in batch]) for k in batch[0].keys()}
    
    elif type(batch[0]) == tuple:
        # list of tuples -> tuple of tensors
        return tuple(batch_cat([b[i] for b in batch]) for i in range(len(batch[0])))

    elif type(batch[0]) == list:
        # list of lists -> list of tensors
        return [batch_cat([b[i] for b in batch]) for i in range(len(batch[0]))]

    return torch.cat(batch, dim=dim)


def batch_dict_index(batch, idx_tuple):
    """
    Index a batch of tensors with a tuple of indices
    """
    assert len(batch) > 0, 'batch is empty'

    if type(batch) == dict:
        return {k:batch[k][idx_tuple] for k in batch.keys()}

    return batch[idx_tuple]

