# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020
"""
Signal processing or PyTorch related utilities.
"""
import math
import typing as tp

import torch
from torch.nn import functional as F


def sinc(x: torch.Tensor):
    """
    Implementation of sinc, i.e. sin(x) / x

    __Warning__: the input is not multiplied by `pi`!
    """
    pass


def pad_to(tensor: torch.Tensor, target_length: int, mode: str = 'constant', value: float = 0):
    """
    Pad the given tensor to the given length, with 0s on the right.
    """
    pass


def hz_to_mel(freqs: torch.Tensor):
    """
    Converts a Tensor of frequencies in hertz to the mel scale.
    Uses the simple formula by O'Shaughnessy (1987).

    Args:
        freqs (torch.Tensor): frequencies to convert.

    """
    pass


def mel_to_hz(mels: torch.Tensor):
    """
    Converts a Tensor of mel scaled frequencies to Hertz.
    Uses the simple formula by O'Shaughnessy (1987).

    Args:
        mels (torch.Tensor): mel frequencies to convert.
    """
    pass


def mel_frequencies(n_mels: int, fmin: float, fmax: float):
    """
    Return frequencies that are evenly spaced in mel scale.

    Args:
        n_mels (int): number of frequencies to return.
        fmin (float): start from this frequency (in Hz).
        fmax (float): finish at this frequency (in Hz).


    """
    pass


def volume(x: torch.Tensor, floor=1e-8):
    """
    Return the volume in dBFS.
    """
    pass


def pure_tone(freq: float, sr: float = 128, dur: float = 4, device=None):
    """
    Return a pure tone, i.e. cosine.

    Args:
        freq (float): frequency (in Hz)
        sr (float): sample rate (in Hz)
        dur (float): duration (in seconds)
    """
    pass


def unfold(input, kernel_size: int, stride: int):
    """1D only unfolding similar to the one from PyTorch.
    However PyTorch unfold is extremely slow.

    Given an input tensor of size `[*, T]` this will return
    a tensor `[*, F, K]` with `K` the kernel size, and `F` the number
    of frames. The i-th frame is a view onto `i * stride: i * stride + kernel_size`.
    This will automatically pad the input to cover at least once all entries in `input`.

    Args:
        input (Tensor): tensor for which to return the frames.
        kernel_size (int): size of each frame.
        stride (int): stride between each frame.

    Shape:

        - Inputs: `input` is `[*, T]`
        - Output: `[*, F, kernel_size]` with `F = 1 + ceil((T - kernel_size) / stride)`


    ..Warning:: unlike PyTorch unfold, this will pad the input
        so that any position in `input` is covered by at least one frame.
    """
    pass
