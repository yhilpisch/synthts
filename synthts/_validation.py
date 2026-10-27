"""Shared validation for model construction and training."""

from numbers import Integral

import numpy as np
import torch


def integer(name, value, minimum=1):
    """Require an integer at or above the specified minimum."""
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or value < minimum):
        raise ValueError(f'{name} must be an integer >= {minimum}')


def dimensions(**values):
    """Require strictly positive model dimensions."""
    for name, value in values.items():
        integer(name, value)


def finite_nonnegative(**values):
    """Require finite, nonnegative scalar settings."""
    for name, value in values.items():
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')


def training_data(data):
    """Require a non-empty, finite observation-by-feature matrix."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or min(data.shape) == 0:
        raise ValueError('data_scaled must have shape (n, features)')
    if not np.isfinite(data).all():
        raise ValueError('data_scaled must contain only finite values')
    return data


def training_settings(data, batch_size, log_interval, **budgets):
    """Validate common training inputs, allowing skipped phases."""
    integer('batch_size', batch_size)
    integer('log_interval', log_interval)
    for name, value in budgets.items():
        integer(name, value, minimum=0)
    return training_data(data)


def temporal_lag(seq_len, max_lag):
    """Require a supported temporal loss horizon."""
    integer('seq_len', seq_len, minimum=2)
    integer('stat_max_lag', max_lag)
    if max_lag >= seq_len:
        raise ValueError('stat_max_lag must be smaller than seq_len')


def backward(loss):
    """Reject non-finite objectives before backpropagation."""
    if not torch.isfinite(loss).all():
        raise FloatingPointError('non-finite training loss')
    loss.backward()


def step(optimizer):
    """Reject non-finite gradients before changing parameters."""
    params = [p for group in optimizer.param_groups for p in group['params']]
    if any(p.grad is not None and not torch.isfinite(p.grad).all()
           for p in params):
        raise FloatingPointError('non-finite training gradient')
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in params):
        raise FloatingPointError('non-finite model parameter')
