from typing import NamedTuple, Tuple, Any, Union, List, Iterable
import numpy as np


class SimParameterType(NamedTuple):
    type: type
    unit: str = None
    min: Union[int, float, None] = None
    max: Union[int, float, None] = None
    choices: list = None


class SimParameter(NamedTuple):
    type: SimParameterType
    name: str
    symbol: str
    default: Any = None
    info: str = None

    def __str__(self):
        return f'{self.name}: {self.type}'

    def human_name(self):
        return self.name.replace('_', ' ').title()


# We define reusable parameter types ...
_CARRIER_FREQUENCY = SimParameterType(float, unit='Hz', min=1e3, max=300e9)
_ADC_FREQUENCY = SimParameterType(float, unit='Hz', min=1, max=300e6)
_RAMP_TIME = SimParameterType(float, unit='s', min=1e-6, max=100)
_METERS = SimParameterType(float, unit='m', min=-100e3, max=100e3)
_COUNT = SimParameterType(int, min=1, max=32000)
_POS_HALF_ANGLE = SimParameterType(float, unit='°', min=0, max=180)

# ... to make a list of all changeable parameters here.
SAR_SIM_PARAMETERS = (
    SimParameter(_CARRIER_FREQUENCY, 'fmcw_start_frequency', 'f_1', 68e9),
    SimParameter(_CARRIER_FREQUENCY, 'fmcw_stop_frequency', 'f_2', 92e9),
    SimParameter(_ADC_FREQUENCY, 'fmcw_adc_frequency', 'f_adc', 1e6),
    SimParameter(_RAMP_TIME, 'fmcw_ramp_duration', 'T_ramp', 8e-3),

    SimParameter(_METERS, 'azimuth_start_position', 'az_x0', -2.5),
    SimParameter(_METERS, 'azimuth_stop_position', 'az_xm', 2.5),

    SimParameter(_COUNT, 'azimuth_count', 'n_az', 201),
    SimParameter(_POS_HALF_ANGLE, 'azimuth_3db_angle_deg', 'ant_a', 7.5),
    SimParameter(_POS_HALF_ANGLE, 'azimuth_compression_beam_limit', 'ac_bl', 7.5),

    SimParameter(_METERS, 'flight_height', 'az_z0', 1.0),
    SimParameter(_METERS, 'flight_distance_to_scene_center', 'r_sc', 4.5),

    SimParameter(_METERS, 'image_start_x', 'img_x0', -1),
    SimParameter(_METERS, 'image_start_y', 'img_y0', -1),
    SimParameter(_METERS, 'image_stop_x', 'img_xm', 1),
    SimParameter(_METERS, 'image_stop_y', 'img_ym', 1),

    SimParameter(_COUNT, 'image_count_x', 'n_x', 501),
    SimParameter(_COUNT, 'image_count_y', 'n_y', 501),

    SimParameter(SimParameterType(float, unit='x', min=1, max=32), 'range_compression_fft_min_oversample', 'rc_fft_os', 16),
)


class SarSimParameterState(object):
    @staticmethod
    def _internal_name(parameter: SimParameter):
        return f'_v_{parameter.name}'

    def __init__(self):
        for parameter in SAR_SIM_PARAMETERS:
            self.__setattr__(SarSimParameterState._internal_name(parameter), parameter.type.type(parameter.default))
        pass

    def _setter(self, parameter: SimParameter, internal_name: str, value):
        old_value = self.__getattribute__(internal_name)
        #print(f'Before set {parameter.name} to {value}, old: {old_value}')
        self.__setattr__(internal_name, value)
        #print(f'After set {parameter.name} to {value}')

    def get_value(self, parameter: SimParameter):
        return self.__getattribute__(SarSimParameterState._internal_name(parameter))

    def set_value(self, parameter: SimParameter, value):
        self._setter(parameter, SarSimParameterState._internal_name(parameter), value)

    @staticmethod
    def get_parameters() -> Tuple[SimParameter, ...]:
        return SAR_SIM_PARAMETERS


def _finalize():
    for parameter in SAR_SIM_PARAMETERS:
        internal_name = SarSimParameterState._internal_name(parameter)
        setattr(SarSimParameterState, parameter.name, property(
            lambda s, i=internal_name: s.__getattribute__(i),
            lambda s, v, i=internal_name: s._setter(parameter, i, v)
        ))


_finalize()


def create_state():
    return SarSimParameterState()


class SimImage(NamedTuple):
    data: np.array
    x0: float
    y0: float
    dx: float
    dy: float


