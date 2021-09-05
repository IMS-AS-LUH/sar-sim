# This module is intended to separate repeated configurable (mathematical) operations used in the simulation

import cmath
import math
from typing import NamedTuple, Callable

import numpy as np
import scipy.signal as signal


class WindowDescriptor(NamedTuple):
    name: str
    """The name for selecting and storing parameter"""
    factory: Callable[[int, bool, float], np.ndarray]
    """Example: `lambda n, s, p: signal.windows.tukey(N, sym=s, alpha=p)`"""
    parameter_name: str = None
    parameter_default: float = 0

    def __str__(self):
        return self.name


SUPPORTED_WINDOWS = [
    WindowDescriptor('Rect', lambda n, s, a: np.ones(n)),
    WindowDescriptor('Triangular', lambda n, s, a: signal.windows.triang(n, sym=s)),
    WindowDescriptor('Hamming', lambda n, s, a: signal.windows.hamming(n, sym=s)),
    WindowDescriptor('Hann', lambda n, s, a: signal.windows.hann(n, sym=s)),
    WindowDescriptor('Blackman', lambda n, s, a: signal.windows.blackman(n, sym=s)),
    WindowDescriptor('Blackman-Harris', lambda n, s, a: signal.windows.blackmanharris(n, sym=s)),
    WindowDescriptor('Chebyshev', lambda n, s, a: signal.windows.chebwin(n, at=a, sym=s), 'at', 0),
    WindowDescriptor('Cosine', lambda n, s, a: signal.windows.cosine(n, sym=s)),
    WindowDescriptor('Flat Top', lambda n, s, a: signal.windows.flattop(n, sym=s)),
    WindowDescriptor('Tukey', lambda n, s, a: signal.windows.tukey(n, alpha=a, sym=s), 'alpha', 0.5),
    WindowDescriptor('Kaiser', lambda n, s, a: signal.windows.kaiser(n, beta=a, sym=s), 'beta', 0.25),
]


def create_window(name: str, parameter: float, n: int, sym: bool = True):
    for window in SUPPORTED_WINDOWS:
        if window.name == name:
            return window.factory(n, sym, parameter)
    raise NotImplementedError(f'Unknown window name: {name}')

