from typing import Tuple


_SI_PREFIX = [
    (1e12, 'T'),
    (1e9, 'G'),
    (1e6, 'M'),
    (1e3, 'k'),
    (1, ''),
    (1e-3, 'm'),
    (1e-6, 'u'),
    (1e-9, 'n'),
    (1e-12, 'p'),
    (1e-15, 'f'),
]


def choose_si_scale(value: float, unit: str = None) -> Tuple[float, str]:
    """
    Chooses a proper SI prefix for the given unit to represent a value.
    Then: Take the unscaled value, divide by factor to get scaled value.
    :param value: The unscaled numeric value to represent
    :param unit: The base unit without prefix or None to get only the prefix
    :return: Tuple of scale factor and prefixed unit
    """
    for factor, name in _SI_PREFIX:
        if value >= factor:
            return factor, f'{name}{unit or ""}'
    return 1, unit


def scale_si_unit(value: float, unit: str = None) -> Tuple[float, str]:
    for factor, name in _SI_PREFIX:
        if value >= factor:
            return value / factor, f'{name}{unit or ""}'
    return value, unit


def format_si_unit(value: float, unit: str = None) -> str:
    value, unit = scale_si_unit(value, unit)
    return f'{value:.3f} {unit}'

