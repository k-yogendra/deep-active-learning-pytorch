"""Loss functions."""

import torch.nn as nn

from pycls.core.config import cfg

# Define the loss functions in a dictionary
_loss_funs = {
    "bce_with_logits": nn.BCEWithLogitsLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    # Add other loss functions as needed
}

def get_loss_fun():
    """Retrieves the loss function."""
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), \
        'Loss function \'{}\' not supported'.format(cfg.MODEL.LOSS_FUN)
    return _loss_funs[cfg.MODEL.LOSS_FUN]().cuda()

def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
