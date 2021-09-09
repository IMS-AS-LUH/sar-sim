from typing import NamedTuple, Union, List, TextIO


class SimpleReflector(NamedTuple):
    """
    Simples possible reflector:
    Has zero size, one single coordinate and only the amplitude
    """
    x: float
    y: float
    z: float
    amplitude: float = 1.0


class SimulationScene(object):
    def __init__(self):
        # These internal lists shall not be altered directly!
        self._simple_reflectors: List[SimpleReflector] = []

    def __add__(self, other: Union[SimpleReflector]):
        if isinstance(other, SimpleReflector):
            self._simple_reflectors.append(other)
        return self

    def get_simple_reflectors(self):
        for e in self._simple_reflectors:
            yield e

    def __hash__(self):
        """
        :return: A hash useable for cache hinting
        """
        # TODO: Make this actually hashing sensibly for caching!
        return hash(tuple(
            map(hash, self._simple_reflectors)
        ))


def create_default_scene() -> SimulationScene:
    s = SimulationScene()
    s += SimpleReflector(0, 0, 0, 1)
    return s


def create_reflector_array_scene(
        count_x: int = 1,
        count_y: int = 1,
        start_x: float = 0.0,
        start_y: float = 0.0,
        spacing_x: float = 0.0375,
        spacing_y: float = 0.0375,
        amplitude: float = 1,
        z: float = 0.0
) -> SimulationScene:
    s = SimulationScene()
    x = start_x
    for ix in range(count_x):
        y = start_y
        for iy in range(count_y):
            s += SimpleReflector(x, y, z, amplitude)
            y += spacing_y
        x += spacing_x
    return s
