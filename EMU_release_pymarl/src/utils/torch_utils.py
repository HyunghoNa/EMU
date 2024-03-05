import torch


def to_cuda(module, device=None):
    """Assume module in cpu"""
    if device is None:
        return module.cuda()
    elif device == 'cpu':
        return module
    else:
        return module.to(device)
