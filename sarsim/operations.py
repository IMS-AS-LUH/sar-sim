# This module is intended to separate repeated configurable (mathematical) operations used in the simulation

import cmath
import math
from typing import NamedTuple, Callable, Optional

import numpy as np
import scipy.signal as signal

class Window:
    def __init__(self, factory: Callable[[int, bool, float], np.ndarray], param_name: Optional[str] = None, param_default: Optional[float] = None) -> None:
        # Example: lambda n, s, p: signal.windows.tukey(N, sym=s, alpha=p)
        self.factory = factory
        self.parameter_name = param_name
        self.parameter_default = param_default


SUPPORTED_WINDOWS = {
    'Rect' : Window(lambda n, s, a: np.ones(n)),
    'Triangular' : Window(lambda n, s, a: signal.windows.triang(n, sym=s)),
    'Hamming' : Window(lambda n, s, a: signal.windows.hamming(n, sym=s)),
    'Hann' : Window(lambda n, s, a: signal.windows.hann(n, sym=s)),
    'Blackman' : Window(lambda n, s, a: signal.windows.blackman(n, sym=s)),
    'Blackman_Harris' : Window(lambda n, s, a: signal.windows.blackmanharris(n, sym=s)),
    'Chebyshev' : Window(lambda n, s, a: signal.windows.chebwin(n, at=a, sym=s), 'at', 0),
    'Cosine' : Window(lambda n, s, a: signal.windows.cosine(n, sym=s)),
    'Flat_Top' : Window(lambda n, s, a: signal.windows.flattop(n, sym=s)),
    'Tukey' : Window(lambda n, s, a: signal.windows.tukey(n, alpha=a, sym=s), 'alpha', 0.5),
    'Kaiser' : Window(lambda n, s, a: signal.windows.kaiser(n, beta=a, sym=s), 'beta', 0.25),
}

def create_window(name: str, parameter: float, n: int, sym: bool = True):
    if not name in SUPPORTED_WINDOWS:
        raise NotImplementedError(f'Unknown window name: {name}')
    return SUPPORTED_WINDOWS[name].factory(n, sym, parameter)

