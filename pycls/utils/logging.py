#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Logging."""

import builtins
import decimal
import torch
import logging
import os
import simplejson
import sys

from pycls.core.config import cfg

# import pycls.utils.distributed as du

# Show filename and line number in logs
_FORMAT = '[%(asctime)s %(filename)s: %(lineno)3d]: %(message)s'

# Log file name (for cfg.LOG_DEST = 'file')
_LOG_FILE = 'stdout.log'

# Printed json stats lines will be tagged w/ this
_TAG = 'json_stats: '


def _suppress_print():
    """Suppresses printing from the current process."""
    def ignore(*_objects, _sep=' ', _end='\n', _file=sys.stdout, _flush=False):
        pass
    builtins.print = ignore


def setup_logging(cfg):
    """Sets up the logging."""
    # Enable logging only for the master process
    # if du.is_master_proc():
    if True:
        # Clear the root logger to prevent any existing logging config
        # (e.g. set by another module) from messing with our setup
        logging.root.handlers = []
        # Construct logging configuration
        logging_config = {
            'level': logging.INFO,
            'format': _FORMAT,
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
        # Log either to stdout or to a file
        if cfg.LOG_DEST == 'stdout':
            logging_config['stream'] = sys.stdout
        else:
            logging_config['filename'] = os.path.join(cfg.EXP_DIR, _LOG_FILE)
        # Configure logging
        logging.basicConfig(**logging_config)
    else:
        _suppress_print()


def get_logger(name):
    """Retrieves the logger."""
    return logging.getLogger(name)

import numpy as np

import simplejson
import torch
import decimal

def tensor_to_serializable(obj):
    """Converts tensors to serializable types."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist() if obj.numel() > 1 else obj.item()
    return obj

def make_serializable(stats):
    """
    Converts the stats dictionary to a serializable format.
    Handles tensors, float32, and potential circular references.
    """
    serializable_stats = {}
    for k, v in stats.items():
        # Convert tensors to serializable types
        if isinstance(v, torch.Tensor):
            v = v.cpu().tolist() if v.numel() > 1 else v.item()
        # Convert float32 to Python float
        elif isinstance(v, np.float32):
            v = float(v)
        # Convert floats to decimals for fixed-length float representation
        elif isinstance(v, float):
            v = decimal.Decimal('{:.12f}'.format(v))
        serializable_stats[k] = v
    return serializable_stats

def log_json_stats(stats):
    """Logs JSON stats with handling for tensors and serialization."""
    # Convert stats to a serializable format
    stats = make_serializable(stats)
    
    try:
        json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
        logger = get_logger(__name__)
        logger.info('{:s}{:s}'.format(_TAG, json_stats))
    except TypeError as e:
        print(f"TypeError while serializing stats: {e}")
        print(f"Stats causing issue: {stats}")



# def log_json_stats(stats):
#     """Logs json stats."""
#     # Decimal + string workaround for having fixed len float vals in logs
#     print(stats)
#     print(stats.keys)
#     print(stats.values)
#     stats = {
#         k: decimal.Decimal('{:.12f}'.format(v)) if isinstance(v, float) else v
#         for k, v in stats.items()
#     }
#     # json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
#     json_stats = simplejson.dumps(stats, default=tensor_to_serializable, sort_keys=True, use_decimal=True)


#     logger = get_logger(__name__)
#     logger.info('{:s}{:s}'.format(_TAG, json_stats))


def load_json_stats(log_file):
    """Loads json_stats from a single log file."""
    with open(log_file, 'r') as f:
        lines = f.readlines()
    json_lines = [l[l.find(_TAG) + len(_TAG):] for l in lines if _TAG in l]
    json_stats = [simplejson.loads(l) for l in json_lines]
    return json_stats


def parse_json_stats(log, row_type, key):
    """Extract values corresponding to row_type/key out of log."""
    vals = [row[key] for row in log if row['_type'] == row_type and key in row]
    if key == 'iter' or key == 'epoch':
        vals = [int(val.split('/')[0]) for val in vals]
    return vals


def get_log_files(log_dir, name_filter=''):
    """Get all log files in directory containing subdirs of trained models."""
    names = [n for n in sorted(os.listdir(log_dir)) if name_filter in n]
    files = [os.path.join(log_dir, n, _LOG_FILE) for n in names]
    f_n_ps = [(f, n) for (f, n) in zip(files, names) if os.path.exists(f)]
    files, names = zip(*f_n_ps)
    return files, names
