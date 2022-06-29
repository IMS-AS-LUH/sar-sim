from typing import Optional

import math

import numpy as np
import scipy.io as sio
import array
import os
from configparser import ConfigParser

from sarsim.operations import SUPPORTED_WINDOWS
from . import simstate

class SarData(object):
    def __init__(self):
        self.cfg: ConfigParser = ConfigParser()
        self.sim_state: simstate.SarSimParameterState = simstate.SarSimParameterState()
        self.fmcw_lines: Optional[list] = None
        self.rg_comp_data: Optional[np.ndarray] = None
        self.flight_path: np.ndarray = np.array([])
        self.name: str = ""
        self.has_range_compressed_data: bool = False

    @staticmethod
    def load_fmcw_binary(bin_file: str, lines: int, line_length: int) -> list:
        data = []
        with open(bin_file, 'rb') as f:
            for i in range(lines):
                a = array.array('i')
                assert a.itemsize == 4, "sizeof(int) != 4"
                a.fromfile(f, line_length)
                data.append(np.array(a, dtype=np.single)/2.0**15)
        return data

    @staticmethod
    def load_range_comp_binary(bin_file: str, lines: int) -> np.ndarray:
        with open(bin_file, 'rb') as f:
            data = np.fromfile(f, dtype=np.complex64)
            return data.reshape((lines, -1))

    def _load_cfg_values(self):
        cfg = self.cfg

        is_rg_comp = cfg['general'].getboolean('data_is_range_compressed', fallback=False)
        self.has_range_compressed_data = is_rg_comp

        if not is_rg_comp:
            # Known Sensor types:
            # - "80": new, small ("RUBv2") 80 GHz sensor from the RUB
            # - "114": new, small ("RUBv2") 114 GHz sensor from the RUB
            # - "RUB_Bike_80": unknown sensor used to capture the "bike" scene from the RUB (same as our old 80 GHz?)
            if str(cfg['general'].get('radartype')) not in ['80', '144', 'RUB_Bike_80']:
                raise Exception("Cannot import - sensor type not implemented")

        sim = self.sim_state
        # *** Paramters with defaults for the demonstrator ***
        sim.fmcw_adc_frequency = cfg['params'].getfloat('adc_frequency', fallback=1.0e6)  # Default implied from Sensor type
        # note that azimuth_3db_angle_deg is used for simulation, azimuth_compression_beam_limit is used for processing
        sim.azimuth_3db_angle_deg = 30  # Guessed somewhat, TODO: Measure some day
        sim.r0 = cfg['params'].getfloat('sensor_range_delay', -0.052) # Default implied from Sensor type
        sim.signal_speed = cfg['params'].getfloat('signal_speed', fallback=simstate.suggested_c_speeds['Air'])

        # *** Load Compatible Parameters ***
        if not is_rg_comp:
            sim.fmcw_start_frequency = cfg['params'].getfloat('start_frequency')
            sim.fmcw_stop_frequency = cfg['params'].getfloat('stop_frequency')
            sim.fmcw_ramp_duration = cfg['params'].getfloat('ramp_duration')
        else:
            # we cheat a bit here and use the center freq. as start freq. and fc+B as stop freq.
            # this does not make so much sense, because we are dealing with already range-compressed
            # data, and therefore cannot tell anymore if it was aquired using a FMCW radar. Anyway,
            # this gives the correct results due to how the backprojection is implemented
            sim.fmcw_start_frequency = cfg['params'].getfloat('center_frequency')
            sim.fmcw_stop_frequency = cfg['params'].getfloat('center_frequency') + cfg['params'].getfloat('bandwidth')
            sim.fmcw_ramp_duration = 0

        sim.azimuth_start_position = cfg['params'].getfloat('azimuth_start_position')
        sim.azimuth_stop_position = cfg['params'].getfloat('azimuth_end_position')
        # Note: azimuth_speed_limit not used YET - TODO: Add when we introduce velocity
        sim.azimuth_count = cfg['params'].getint('azimuth_count')

        sim.azimuth_compression_beam_limit = cfg['params'].getfloat('gbp_beam_limit', fallback=30)        

        # Guessed somewhat, TODO: Qualify sensible settings
        sim.azimuth_compression_window = SUPPORTED_WINDOWS['Rect']
        sim.range_compression_window = SUPPORTED_WINDOWS['Tukey']
        sim.range_compression_window_parameter = 0.25

        # *** Decode Flightpath ***
        if not is_rg_comp:
            img_offset_z = cfg['params'].getfloat('gbp_image_z', fallback=-0.05)
        else:
            img_offset_z = cfg['params'].getfloat('gbp_image_z', fallback=0) # use 0 as fallback for z height for rg compressed, this makes more sense
        transposed_flight_path = np.array([
            [float(x) for x in str(cfg['fpath']['x']).split(',')],
            [float(x) for x in str(cfg['fpath']['y']).split(',')],
            [float(x) - img_offset_z for x in str(cfg['fpath']['z']).split(',')]
        ], dtype=np.single)

        if transposed_flight_path.shape[1] != sim.azimuth_count or transposed_flight_path.shape[0] != 3:
            raise Exception("Unexpected dimensions of flight-path array from cfg file.")

        self.flight_path = np.array(transposed_flight_path).transpose()

        # *** Calculate derived parameters ***
        image_region = cfg['params'].get('gbp_image_region', fallback='(0.25, 0.5, 2.75, 3.0)')
        if image_region[0] != '(' or image_region[-1] != ')':
            raise Exception("Invalid tuple in config file")
        image_region = image_region[1:-1].split(',')
        if len(image_region) != 4:
            raise Exception("Invalid tuple in config file")
        # x,y to x,y rectangle (aka. az,rg)
        image_region = [float(x.strip()) for x in image_region]

        sim.image_start_x = image_region[0]
        sim.image_start_y = image_region[1]
        sim.image_stop_x = image_region[2]
        sim.image_stop_y = image_region[3]

        sim.image_count_y = cfg['params'].getint('gbp_image_pixels_ range', fallback=1024)
        sim.image_count_x = round(sim.image_count_y *
                                  (image_region[2] - image_region[0]) /
                                  (image_region[3] - image_region[1])
                                  )

        fax = np.average(transposed_flight_path[0])
        fay = np.average(transposed_flight_path[1])
        faz = np.average(transposed_flight_path[2])

        icx = (sim.image_start_x + sim.image_stop_x) / 2
        icy = (sim.image_start_y + sim.image_stop_y) / 2
        # icz = 0 by definition

        sim.flight_height = faz # type: ignore # floating[Any] vs. float...
        sim.flight_distance_to_scene_center = math.sqrt((fax-icx)**2 + (fay-icy)**2)
        if not is_rg_comp:
            sim.range_compression_fft_min_oversample = math.floor(
                cfg['params'].getint('range_compression_nfft', fallback=32768) /
                (sim.fmcw_ramp_duration * sim.fmcw_adc_frequency)
            )

    @classmethod
    def import_from_directory(cls, directory: str) -> 'SarData':
        sd = SarData()
        capture_id = os.path.basename(directory)
        if not capture_id.endswith('.sardata'):
            raise Exception("Selected folder does not look like a valid *.sardata archive.")
        capture_id = capture_id[0:-8]
        sd.name = capture_id

        # try to find the .cfg file. Per (new) spec it should be called "params.cfg"
        # but it used to have the same basename as the folder, or an unrelated name if
        # the folder was renamed. We try the first to options and, as a last resort,
        # check if there is a single .cfg file and use that.
        cfg_file_name = None
        for n in ['params.cfg', capture_id + '.cfg']:
            if os.path.exists(os.path.join(directory, n)):
                cfg_file_name = n
                break
        if cfg_file_name is None:
            cfg_candidates = [x for x in os.listdir(directory) if x.endswith('.cfg')]
            if len(cfg_candidates) != 1:
                raise Exception("Folder contains no/too many *.cfg files.")
            cfg_file_name = cfg_candidates[0]

        
        cfg = ConfigParser()
        if len(cfg.read(os.path.join(directory, cfg_file_name))) != 1:
            raise Exception("Cannot read cfg file in captured sardata.")
        sd.cfg = cfg
        sd._load_cfg_values()

        # now do the same to find the .bin file. The official name is "fmcw.bin",
        # except with data_is_range_compressed is set, then it should be called
        # "range_comp.bin" (and we don't try the other in that case!)
        if sd.has_range_compressed_data:
            bin_spec_name = 'range_comp.bin'
        else:
            bin_spec_name = 'fmcw.bin'

        bin_file_name = None
        for n in [bin_spec_name, capture_id + '.bin']:
            if os.path.exists(os.path.join(directory, n)):
                bin_file_name = n
                break
        if bin_file_name is None:
            bin_candidates = [x for x in os.listdir(directory) if x.endswith('.bin') and x != 'fmcw.bin' and x != 'range_comp.bin']
            if len(bin_candidates) != 1:
                raise Exception("Folder contains no/too many *.bin files.")
            bin_file_name = bin_candidates[0]

        if not sd.has_range_compressed_data:
            samples_per_rangeline = round(sd.sim_state.fmcw_ramp_duration * sd.sim_state.fmcw_adc_frequency)
            sd.fmcw_lines = cls.load_fmcw_binary(
                os.path.join(directory, bin_file_name),
                sd.sim_state.azimuth_count,
                samples_per_rangeline
            )
        else:
            sd.rg_comp_data = cls.load_range_comp_binary(os.path.join(directory, bin_file_name), sd.sim_state.azimuth_count)
        return sd
