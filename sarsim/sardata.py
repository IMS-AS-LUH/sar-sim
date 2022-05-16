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
        self.fmcw_lines: list = []
        self.flight_path: np.ndarray = np.array([])
        self.name: str = ""

    @staticmethod
    def load_fmcw_binary(bin_file: str, lines: int, line_length: int) -> list:
        data = []
        with open(bin_file, 'rb') as f:
            for i in range(lines):
                a = array.array('i')
                a.fromfile(f, line_length)
                data.append(np.array(a, dtype=np.single)/2.0**15)
        return data

    def _load_cfg_values(self):
        cfg = self.cfg
        if str(cfg['general'].get('radartype')) not in ['80', '144']:
            raise Exception("Cannot import - only 80 and 144 GHz Sensor types implemented")

        sim = self.sim_state
        # *** Fixed parameters not in cfg file ***
        sim.fmcw_adc_frequency = 1.0e6  # Implied from Sensor type
        sim.azimuth_3db_angle_deg = 30  # Guessed somewhat, TODO: Measure some day
        sim.r0 = -0.052 # Implied from Sensor type

        # *** Load Compatible Parameters ***
        sim.fmcw_start_frequency = cfg['params'].getfloat('start_frequency')
        sim.fmcw_stop_frequency = cfg['params'].getfloat('stop_frequency')
        sim.fmcw_ramp_duration = cfg['params'].getfloat('ramp_duration')

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
        img_offset_z = cfg['params'].getfloat('gbp_image_z', fallback=-0.05)
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

        sim.flight_height = faz
        sim.flight_distance_to_scene_center = math.sqrt((fax-icx)**2 + (fay-icy)**2)
        sim.range_compression_fft_min_oversample = math.floor(
            cfg['params'].getint('range_compression_nfft', fallback=32768) /
            (sim.fmcw_ramp_duration * sim.fmcw_adc_frequency)
        )

        pass

    @classmethod
    def import_from_directory(cls, directory: str) -> 'SarData':
        sd = SarData()
        capture_id = os.path.basename(directory)
        if not capture_id.endswith('.sardata'):
            raise Exception("Selected folder does not look like a valid *.sardata archive.")
        capture_id = capture_id[0:-8]
        sd.name = capture_id

        # try to find the .cfg and bin file. It usually has the same basename as the folder,
        # but might have a different name if the folder was renamed. Therefore we just check
        # if there is a single file with the correct extension.
        cfg_files = [x for x in os.listdir(directory) if x.endswith('.cfg')]
        bin_files = [x for x in os.listdir(directory) if x.endswith('.bin')]

        if len(cfg_files) != 1:
            raise Exception("Folder contains no/too many *.cfg files.")

        if len(bin_files) != 1:
            raise Exception("Folder contains no/too many *.bin files.")

        cfg = ConfigParser()
        if len(cfg.read(os.path.join(directory, cfg_files[0]))) != 1:
            raise Exception("Cannot read cfg file in captured sardata.")
        sd.cfg = cfg
        sd._load_cfg_values()
        samples_per_rangeline = round(sd.sim_state.fmcw_ramp_duration * sd.sim_state.fmcw_adc_frequency)
        sd.fmcw_lines = cls.load_fmcw_binary(
            os.path.join(directory, bin_files[0]),
            sd.sim_state.azimuth_count,
            samples_per_rangeline
        )
        return sd
