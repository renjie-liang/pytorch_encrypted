"""Tensors representing non-native data types (like fixed-point precision)."""
from __future__ import absolute_import
import torch

from .native import native_factory
# from .int100 import int100factory
from .fixed import (
    _validate_fixedpoint_config,
    fixed100,
    fixed100_ni,
    fixed64,
    fixed64_ni,
)

int32factory = native_factory(torch.int32)
int64factory = native_factory(torch.int64)

# assert _validate_fixedpoint_config(fixed100, int100factory)
# assert _validate_fixedpoint_config(fixed100_ni, int100factory)
assert _validate_fixedpoint_config(fixed64, int64factory)
assert _validate_fixedpoint_config(fixed64_ni, int64factory)

__all__ = [
    'native_factory',
    'int32factory',
    'int64factory',
    # 'int100factory',
]
