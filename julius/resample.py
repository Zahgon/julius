# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020
"""
Differentiable, Pytorch based resampling.
Implementation of Julius O. Smith algorithm for resampling.
See https://ccrma.stanford.edu/~jos/resample/ for details.
This implementation is specially optimized for when new_sr / old_sr is a fraction
with a small numerator and denominator when removing the gcd (e.g. new_sr = 700, old_sr = 500).

Very similar to [bmcfee/resampy](https://github.com/bmcfee/resampy) except this implementation
is optimized for the case mentioned before, while resampy is slower but more general.

"""

import math
from typing import Optional

import torch
from torch.nn import functional as F

from .core import sinc
from .utils import simple_repr


class ResampleFrac(torch.nn.Module):
    """
    Resampling from the sample rate `old_sr` to `new_sr`.
    """
    def __init__(self, old_sr: int, new_sr: int, zeros: int = 24, rolloff: float = 0.945):
        """
        Args:
            old_sr (int): sample rate of the input signal x.
            new_sr (int): sample rate of the output.
            zeros (int): number of zero crossing to keep in the sinc filter.
            rolloff (float): use a lowpass filter that is `rolloff * new_sr / 2`,
                to ensure sufficient margin due to the imperfection of the FIR filter used.
                Lowering this value will reduce anti-aliasing, but will reduce some of the
                highest frequencies.

        Shape:

            - Input: `[*, T]`
            - Output: `[*, T']` with `T' = int(new_sr * T / old_sr)


        .. caution::
            After dividing `old_sr` and `new_sr` by their GCD, both should be small
            for this implementation to be fast.

        >>> import torch
        >>> resample = ResampleFrac(4, 5)
        >>> x = torch.randn(1000)
        >>> print(len(resample(x)))
        1250
        """
        super().__init__()
        if not isinstance(old_sr, int) or not isinstance(new_sr, int):
            raise ValueError("old_sr and new_sr should be integers")
        gcd = math.gcd(old_sr, new_sr)
        self.old_sr = old_sr // gcd
        self.new_sr = new_sr // gcd
        self.zeros = zeros
        self.rolloff = rolloff

        self._init_kernels()

    def _init_kernels(self):
        pass

    def forward(self, x: torch.Tensor, output_length: Optional[int] = None, full: bool = False):
        """
        Resample x.
        Args:
            x (Tensor): signal to resample, time should be the last dimension
            output_length (None or int): This can be set to the desired output length
                (last dimension). Allowed values are between 0 and
                ceil(length * new_sr / old_sr). When None (default) is specified, the
                floored output length will be used. In order to select the largest possible
                size, use the `full` argument.
            full (bool): return the longest possible output from the input. This can be useful
                if you chain resampling operations, and want to give the `output_length` only
                for the last one, while passing `full=True` to all the other ones.
        """
        pass

    def __repr__(self):
        return simple_repr(self)


def resample_frac(x: torch.Tensor, old_sr: int, new_sr: int,
                  zeros: int = 24, rolloff: float = 0.945,
                  output_length: Optional[int] = None, full: bool = False):
    """
    Functional version of `ResampleFrac`, refer to its documentation for more information.

    ..warning::
        If you call repeatidly this functions with the same sample rates, then the
        resampling kernel will be recomputed everytime. For best performance, you should use
        and cache an instance of `ResampleFrac`.
    """
    pass


# Easier implementations for downsampling and upsampling by a factor of 2
# Kept for testing and reference

def _kernel_upsample2_downsample2(zeros):
    # Kernel for upsampling and downsampling by a factor of 2. Interestingly,
    # it is the same kernel used for both.
    pass


def _upsample2(x, zeros=24):
    """
    Upsample x by a factor of two. The output will be exactly twice as long as the input.
    Args:
        x (Tensor): signal to upsample, time should be the last dimension
        zeros (int): number of zero crossing to keep in the sinc filter.

    This function is kept only for reference, you should use the more generic `resample_frac`
    one. This function does not perform anti-aliasing filtering.
    """
    pass


def _downsample2(x, zeros=24):
    """
    Downsample x by a factor of two. The output length is half of the input, ceiled.
    Args:
        x (Tensor): signal to downsample, time should be the last dimension
        zeros (int): number of zero crossing to keep in the sinc filter.

    This function is kept only for reference, you should use the more generic `resample_frac`
    one. This function does not perform anti-aliasing filtering.
    """
    pass
