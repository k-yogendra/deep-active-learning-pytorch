# This file is modified from the official pycls repository

"""Model and loss construction functions."""

from pycls.core.net import SoftCrossEntropyLoss
from pycls.models.resnet import *
from pycls.models.vgg import *
from pycls.models.alexnet import *
from pycls.models.mlp import MLPClassifierTorch  # Import the MLP model

import torch

# Supported models
_models = {
    # VGG style architectures
    'vgg11': vgg11,
    'vgg11_bn': vgg11_bn,
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19': vgg19,
    'vgg19_bn': vgg19_bn,

    # ResNet style architectures
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2,

    # AlexNet architecture
    'alexnet': alexnet,

    # MLP architecture
    'mlp': lambda input_dim, num_classes, use_dropout=True: MLPClassifierTorch(
    input_dim=input_dim,
    num_classes=num_classes,
    use_dropout=use_dropout
)  # Use 30 as placeholder input dimension
}

# Supported loss functions
_loss_funs = {"cross_entropy": SoftCrossEntropyLoss}


def get_model(cfg):
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.MODEL.TYPE in _models.keys(), err_str.format(cfg.MODEL.TYPE)
    return _models[cfg.MODEL.TYPE]


def get_loss_fun(cfg):
    """Gets the loss function class specified in the config."""
    err_str = "Loss function type '{}' not supported"
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), err_str.format(cfg.MODEL.LOSS_FUN)
    return _loss_funs[cfg.MODEL.LOSS_FUN]


def build_model(cfg):
    """Builds the model."""
    model_class = get_model(cfg)

    if cfg.MODEL.TYPE == "mlp":
        input_dim = cfg.DATASET.INPUT_DIM  # Fetch input dimension from the config
        num_classes = cfg.MODEL.NUM_CLASSES
        model = model_class(input_dim=input_dim, num_classes=num_classes)
    else:
        model = model_class(num_classes=cfg.MODEL.NUM_CLASSES, use_dropout=True)

    if cfg.DATASET.NAME == 'MNIST':
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    return model


def build_loss_fun(cfg):
    """Builds the loss function."""
    return get_loss_fun(cfg)()


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
