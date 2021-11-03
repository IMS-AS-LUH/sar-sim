from typing import NamedTuple, Tuple, Any, Union, List, Iterable
import numpy as np
import configparser
from .operations import SUPPORTED_WINDOWS

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
    category: str = None

    def __str__(self):
        return f'{self.name}: {self.type}'

    def human_name(self):
        return self.name.replace('_', ' ').title()


# We define reusable parameter types ...
_CARRIER_FREQUENCY = SimParameterType(float, unit='Hz', min=1e3, max=300e9)
_ADC_FREQUENCY = SimParameterType(float, unit='Hz', min=1, max=300e6)
_WIGGLE_FREQUENCY = SimParameterType(float, unit='1/m', min=0.1, max=100)
_RAMP_TIME = SimParameterType(float, unit='s', min=1e-6, max=100)
_METERS = SimParameterType(float, unit='m', min=-100e3, max=100e3)
_COUNT = SimParameterType(int, min=1, max=32000)
_POS_HALF_ANGLE = SimParameterType(float, unit='°', min=0, max=180)
_WINDOW = SimParameterType(str, choices=[x.name for x in SUPPORTED_WINDOWS])
_WINDOW_PARAM = SimParameterType(float)
_PERCENT = SimParameterType(float, unit='%', min=0, max=100)
_FACTOR = SimParameterType(float, min=0)


# ... to make a list of all changeable parameters here.
SAR_SIM_PARAMETERS = (
    SimParameter(_CARRIER_FREQUENCY, 'fmcw_start_frequency', 'f_1', 68e9, category='Acquisition'),
    SimParameter(_CARRIER_FREQUENCY, 'fmcw_stop_frequency', 'f_2', 92e9, category='Acquisition'),
    SimParameter(_ADC_FREQUENCY, 'fmcw_adc_frequency', 'f_adc', 1e6, category='Acquisition'),
    SimParameter(_RAMP_TIME, 'fmcw_ramp_duration', 'T_ramp', 8e-3, category='Acquisition'),

    SimParameter(_METERS, 'azimuth_start_position', 'az_x0', -2.5, category='Acquisition'),
    SimParameter(_METERS, 'azimuth_stop_position', 'az_xm', 2.5, category='Acquisition'),

    SimParameter(_COUNT, 'azimuth_count', 'n_az', 201, category='Acquisition'),
    SimParameter(_POS_HALF_ANGLE, 'azimuth_3db_angle_deg', 'ant_a', 7.5, category='Acquisition'),
    SimParameter(_POS_HALF_ANGLE, 'azimuth_compression_beam_limit', 'ac_bl', 7.5, category='Acquisition'),

    SimParameter(_WINDOW, 'azimuth_compression_window', 'ac_wnd', 'Rect', category='Azimuth Compression'),
    SimParameter(_WINDOW_PARAM, 'azimuth_compression_window_parameter', 'ac_wnd_param', 0, category='Azimuth Compression'),

    SimParameter(_METERS, 'flight_height', 'az_z0', 1.0, category='Acquisition'),
    SimParameter(_METERS, 'flight_distance_to_scene_center', 'r_sc', 4.5, category='Acquisition'),

    SimParameter(_FACTOR, 'flight_wiggle_global_scale', 'wiggle_scale', 0, category='Flight path'),
    SimParameter(_METERS, 'flight_wiggle_amplitude_azimuth', 'wiggle_az_ampl', 0.05, category='Flight path'),
    SimParameter(_METERS, 'flight_wiggle_amplitude_range', 'wiggle_rg_ampl', 0.01, category='Flight path'),
    SimParameter(_METERS, 'flight_wiggle_amplitude_height', 'wiggle_z_ampl', 0.015, category='Flight path'),
    SimParameter(_WIGGLE_FREQUENCY, 'flight_wiggle_frequency_azimuth', 'wiggle_az_freq', 2, category='Flight path'),
    SimParameter(_WIGGLE_FREQUENCY, 'flight_wiggle_frequency_range', 'wiggle_rg_freq', 4, category='Flight path'),
    SimParameter(_WIGGLE_FREQUENCY, 'flight_wiggle_frequency_height', 'wiggle_z_freq', 5, category='Flight path'),

    SimParameter(_METERS, 'image_start_x', 'img_x0', -1, category='Azimuth Compression'),
    SimParameter(_METERS, 'image_start_y', 'img_y0', -1, category='Azimuth Compression'),
    SimParameter(_METERS, 'image_stop_x', 'img_xm', 1, category='Azimuth Compression'),
    SimParameter(_METERS, 'image_stop_y', 'img_ym', 1, category='Azimuth Compression'),

    SimParameter(_COUNT, 'image_count_x', 'n_x', 501, category='Azimuth Compression'),
    SimParameter(_COUNT, 'image_count_y', 'n_y', 501, category='Azimuth Compression'),

    SimParameter(SimParameterType(float, unit='x', min=1, max=32), 'range_compression_fft_min_oversample', 'rc_fft_os', 16, category='Range Compression'),
    SimParameter(_WINDOW, 'range_compression_window', 'rc_wnd', 'Rect', category='Range Compression'),
    SimParameter(_WINDOW_PARAM, 'range_compression_window_parameter', 'rc_wnd_param', 0, category='Range Compression'),
    SimParameter(_PERCENT, 'range_compression_used_bandwidth', 'rc_cut_bw', 100, category='Range Compression')
)


class SarSimParameterState(object):
    @staticmethod
    def _internal_name(parameter: SimParameter):
        return f'_v_{parameter.name}'

    def __init__(self):
        # <dynamic_properties>
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

    def write_to_file(self, filename: str):
        cfg = configparser.ConfigParser()
        cfg['params'] = {}

        for param in self.get_parameters():
            cfg['params'][param.name] = str(self.get_value(param)) # everything must be a string

        with open(filename, 'w') as f:
            cfg.write(f)

    @staticmethod
    def read_from_file(filename: str):
        cfg = configparser.ConfigParser()
        cfg.read(filename)

        state = SarSimParameterState()

        for param in SarSimParameterState.get_parameters():
            if param.name not in cfg['params']:
                print(f'Load Parameter Note: Using default for {param.name}: {state.get_value(param)}')
                continue
            if param.type.type == int:
                val = cfg['params'].getint(param.name)
            elif param.type.type == float:
                val = cfg['params'].getfloat(param.name)
            elif param.type.type == bool:
                val = cfg['params'].getboolean(param.name)
            elif param.type.type == str:
                val = cfg['params'].get(param.name)
            else:
                raise NotImplementedError(f"Type {param.type.type} is not supported by the config file read yet")

            state.set_value(param, val)

        return state

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


def write_simstate_stub_file() -> None:
    """
    Creates a stub-file called simstate.pyi for this file (simstate.py).
    It includes the dynamic parameters to enable easier IDE coding in the classes using the SarSimParameterState.
    """
    stub_file_name = f'{__file__}i'
    with open(stub_file_name, 'w', encoding='utf-8') as f:
        with open(__file__, 'r', encoding='utf-8') as me:
            f.write('"""\n')
            f.write('Auto-Generated Stub File for better IDE coding assistance.\n')
            f.write('DO NOT EDIT THIS FILE DIRECTLY, NOT CHECK INTO GIT.\n')
            f.write(f'Edit {__file__} instead and run the sar-sim module with --write-stubs.\n')
            f.write('"""\n\n')
            for line in me:
                if line.strip() == '# <dynamic_properties>':
                    indent = ' '*line.index('#')
                    f.write(f'{indent}# BEGIN OF ADDED DYNAMIC PROPERTIES >>\n')
                    for parameter in SAR_SIM_PARAMETERS:
                        f.write(f'{indent}self.{parameter.name}: {parameter.type.type.__name__} = {parameter.default}\n')
                        f.write(f'{indent}"""\n')
                        f.write(f'{indent}**{parameter.symbol} [{parameter.type.unit}]** {parameter.human_name()}\n\n')
                        if parameter.type.min is not None or parameter.type.max is not None:
                            f.write(f'{indent}*Range: {parameter.type.min} to {parameter.type.max}*\n\n')
                        if parameter.info is not None:
                            indented_info = parameter.info.replace('\n', f'\n{indent}')
                            f.write(f'{indent}{indented_info}')
                        if parameter.category is not None:
                            f.write(f'{indent}Category: {parameter.category}')
                        f.write(f'{indent}"""\n\n')
                    f.write(f'{indent}# << END OF ADDED DYNAMIC PROPERTIES\n')
                else:
                    f.write(line)


class SimImage(NamedTuple):
    data: np.array
    x0: float
    y0: float
    dx: float
    dy: float


