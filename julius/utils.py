# File under the MIT license, see https://github.com/adefossez/julius/LICENSE for details.
# Author: adefossez, 2020
"""
Non signal processing related utilities.
"""

import inspect
import typing as tp
import sys
import time


def simple_repr(obj, attrs: tp.Optional[tp.Sequence[str]] = None,
                overrides: dict = {}):
    """
    Return a simple representation string for `obj`.
    If `attrs` is not None, it should be a list of attributes to include.
    """
    pass


class MarkdownTable:
    """
    Simple MarkdownTable generator. The column titles should be large enough
    for the lines content. This will right align everything.

    >>> import io  # we use io purely for test purposes, default is sys.stdout.
    >>> file = io.StringIO()
    >>> table = MarkdownTable(["Item Name", "Price"], file=file)
    >>> table.header(); table.line(["Honey", "5"]); table.line(["Car", "5,000"])
    >>> print(file.getvalue().strip())  # Strip for test purposes
    | Item Name | Price |
    |-----------|-------|
    |     Honey |     5 |
    |       Car | 5,000 |
    """
    def __init__(self, columns, file=sys.stdout):
        self.columns = columns
        self.file = file

    def _writeln(self, line):
        pass

    def header(self):
        pass

    def line(self, line):
        pass


class Chrono:
    """
    Measures ellapsed time, calling `torch.cuda.synchronize` if necessary.
    `Chrono` instances can be used as context managers (e.g. with `with`).
    Upon exit of the block, you can access the duration of the block in seconds
    with the `duration` attribute.

    >>> with Chrono() as chrono:
    ...     _ = sum(range(10_000))
    ...
    >>> print(chrono.duration < 10)  # Should be true unless on a really slow computer.
    True
    """
    def __init__(self):
        self.duration = None

    def __enter__(self):
        self._begin = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tracebck):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.duration = time.time() - self._begin
