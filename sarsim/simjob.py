import cmath
import math

import numpy as np
import scipy.signal as signal
from . import operations
from . import sardata
from . import simscene

CUDA_NUMBA_AVAILABLE = True

try:
    from numba import cuda
except ModuleNotFoundError:
    CUDA_NUMBA_AVAILABLE = False
    print('Numba (CUDA) not found. Falling back to CPU.')

from . import simstate, profiling


def run_sim(state: simstate.SarSimParameterState,
            scene: simscene.SimulationScene,
            timestamper:profiling.TimeStamper = None,
            progress_callback: callable = None,
            loaded_data: sardata.SarData = None):
    timestamper = timestamper or profiling.TimeStamper()
    ac_use_cuda = CUDA_NUMBA_AVAILABLE
    use_loaded_data = loaded_data is not None

    progress_callback = progress_callback or (lambda _0, _1: None)
    progress_callback(0, 'Preparing Simulation')

    # FMCW Sensor Ramp Parameters
    #state.fmcw_start_frequency: float
    #state.fmcw_stop_frequency: float
    #state.fmcw_ramp_duration: float
    #state.fmcw_adc_frequency: float

    # Base flight path parameter (azimuth)
    #state.azimuth_start_position: float
    #state.azimuth_stop_position: float
    #state.azimuth_count: int

    # state.azimuth_3db_angle_deg: float

    #state.flight_height: float
    #state.flight_distance_to_scene_center: float

    # Environmental Constants
    signal_speed = 2.99709e8

    ## PREPARE DATA ##
    timestamper.tic('Data Preparation')

    # Optimal azimuth x positions
    azimuth_x = np.linspace(state.azimuth_start_position, state.azimuth_stop_position, state.azimuth_count)

    print(
        f'Azimuth: {state.azimuth_count} Positions spaced {azimuth_x[1] - azimuth_x[0]}m yield {azimuth_x[-1] - azimuth_x[0]}m Track.')

    # Optimal flight path 3D [state.azimuth_count, 3]
    flight_path = None
    if use_loaded_data:
        flight_path = loaded_data.flight_path
    else:
        flight_path = np.array([
            [x, -state.flight_distance_to_scene_center, state.flight_height]
            for x in azimuth_x
        ])

    # FMCW Simulation
    fmcw_bw = abs(state.fmcw_stop_frequency - state.fmcw_start_frequency)
    fmcw_up = True if state.fmcw_stop_frequency > state.fmcw_start_frequency else False
    fmcw_slope = (state.fmcw_stop_frequency - state.fmcw_start_frequency) / state.fmcw_ramp_duration
    fmcw_samples = math.ceil(state.fmcw_ramp_duration * state.fmcw_adc_frequency)
    fmcw_t = np.array([float(i) / state.fmcw_adc_frequency for i in range(fmcw_samples)])

    timestamper.toc()
    fmcw_lines = None
    if use_loaded_data:
        progress_callback(.05, 'Using loaded FMCW Data')
        fmcw_lines = loaded_data.fmcw_lines
    else:
        progress_callback(.05, 'FMCW Simulation')

        print(f'FMCW: {state.fmcw_start_frequency * 1e-9:.1f} GHz to {state.fmcw_stop_frequency * 1e-9:.1f} GHz '
              f'yields {fmcw_bw * 1e-9:.1f} GHz Bandwidth of {"up" if fmcw_up else "down"}-Ramp.')

        print(f'FMCW: {state.fmcw_ramp_duration * 1e3:.1f} ms Ramp-Time at {state.fmcw_adc_frequency * 1e-3:.1f} kHz '
              f'yields {fmcw_samples:d} samples.')

        timestamper.tic('FMCW Simulation')

        fmcw_lines = _fmcw_sim(flight_path, fmcw_samples, scene, signal_speed, state.fmcw_start_frequency, fmcw_slope, fmcw_t, state.azimuth_3db_angle_deg)

        timestamper.toc()

    ## Range Compression
    progress_callback(.25, 'Range Compression')
    timestamper.tic('Range Compression')
    # wnd = signal.hann(M=len(fmcw_lines[0]), sym=False)
    # wnd = signal.windows.tukey(M=len(fmcw_lines[0]), sym=False, alpha=0.25)

    rc_used_bandwidth = state.range_compression_used_bandwidth / 100
    if rc_used_bandwidth >= 0.99999999:
        wnd = operations.create_window(state.range_compression_window, state.range_compression_window_parameter, len(fmcw_lines[0]), False)
    else:
        original_length = len(fmcw_lines[0])
        used_part = round(rc_used_bandwidth*original_length)
        used_offset = math.floor((original_length-used_part)/2)
        wnd = operations.create_window(state.range_compression_window, state.range_compression_window_parameter, used_part, False)
        wnd = np.pad(wnd, (used_offset, original_length-used_part-used_offset), 'constant', constant_values=(0, 0))

    #nfft = 16 * 1024  # len(fmcw_lines[0])
    print(f'Range-FFT: Have {len(wnd)} samples, minimum oversampling: {state.range_compression_fft_min_oversample:.3f}x')
    nfft = state.range_compression_fft_min_oversample * len(fmcw_lines[0])
    nfft = 2**math.ceil(math.log2(nfft))
    print(f'Using FFT length of {nfft} to get actual oversampling of {nfft/len(wnd):.3f}x')
    rc_lines = [
        np.fft.rfft(fmcw_line * wnd, n=nfft)
        for fmcw_line in fmcw_lines
    ]
    timestamper.toc()

    ## Azimuth Compression
    progress_callback(.5, 'Azimuth Compression')
    timestamper.tic('Azimuth Compression')
    image_x = np.linspace(state.image_start_x, state.image_stop_x, state.image_count_x)
    image_y = np.linspace(state.image_start_y, state.image_stop_y, state.image_count_y)

    ac_wnd = operations.create_window(state.azimuth_compression_window, state.azimuth_compression_window_parameter,
                                   len(fmcw_lines), False)

    print(
        f'Image: {state.image_count_x}x{state.image_count_y} Pixels spaced {image_x[1] - image_x[0]:.3f}x{image_y[1] - image_y[0]:.3f}m.')

    image = np.array([[complex(0, 0)] * state.image_count_y] * state.image_count_x)

    PC1 = -4 * math.pi * state.fmcw_start_frequency / signal_speed
    PC2 = 4 * math.pi * fmcw_slope / (signal_speed ** 2)

    r0 = 0
    fif = (state.fmcw_adc_frequency / 2) / (2 * fmcw_slope) * signal_speed
    rmax = r0 + fif
    r_vector = np.linspace(r0, rmax, len(rc_lines[0]))

    r_scale = r_vector[1] - r_vector[0]
    r_idx_max_sub1 = len(rc_lines[0]) - 2

    if ac_use_cuda:
        nx = len(image_x)
        ny = len(image_y)
        na = len(flight_path)

        beamlimit = state.azimuth_compression_beam_limit / 180 * math.pi

        @cuda.jit()
        def ac_kernel(flight_path_array, image, rc_lines):
            ix, iy = cuda.grid(2)
            if ix >= nx or iy >= ny:
                return
            temp = complex(0, 0)
            x = image_x[ix]
            y = image_y[iy]
            for ia in range(0, na):
                flight_point = flight_path_array[ia]
                a = ac_wnd[ia]

                delta_x = x - flight_point[0]
                delta_y = y - flight_point[1]
                delta_z = 0 - flight_point[2]
                delta_r = math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
                phi_a = math.atan2(delta_x, delta_y)
                pc = PC1 * delta_r + PC2 * delta_r ** 2
                # sample = np.interp(delta_r, r_vector, rc_lines[ia])
                sample_index = (delta_r - r0) / r_scale
                sample_int = math.floor(sample_index)
                sample_frac = sample_index - sample_int
                if sample_int < 0 or sample_int > r_idx_max_sub1 or abs(phi_a) > beamlimit:
                    sample = complex(0, 0)
                else:
                    sample_int = int(sample_int)
                    sample = rc_lines[ia][sample_int] * (1 - sample_frac) + rc_lines[ia][sample_int + 1] * sample_frac
                temp = temp + sample * cmath.exp(complex(0, pc)) * a
            image[ix][iy] = temp

        blocksize = (16, 16)
        gridsize = (math.ceil(nx / blocksize[0]), math.ceil(ny / blocksize[1]))
        rc_lines_array = np.array(rc_lines)
        ac_kernel[gridsize, blocksize](flight_path, image, rc_lines_array)

    else:
        for ia in range(len(flight_path)):
            flight_point = flight_path[ia]
            a = ac_wnd[ia]
            for ix in range(len(image_x)):
                x = image_x[ix]
                for iy in range(len(image_y)):
                    y = image_y[iy]
                    delta_x = x - flight_point[0]
                    delta_y = y - flight_point[1]
                    delta_z = 0 - flight_point[2]
                    delta_r = math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)
                    pc = PC1 * delta_r + PC2 * delta_r ** 2
                    # sample = np.interp(delta_r, r_vector, rc_lines[ia])
                    sample_index = (delta_r - r0) / r_scale
                    sample_int = math.floor(sample_index)
                    sample_frac = sample_index - sample_int
                    if sample_int < 0 or sample_int > r_idx_max_sub1:
                        sample = complex(0, 0)
                    else:
                        sample_int = int(sample_int)
                        sample = rc_lines[ia][sample_int] * (1 - sample_frac) + rc_lines[ia][
                            sample_int + 1] * sample_frac
                    image[ix][iy] = image[ix][iy] + sample * cmath.exp(complex(0, pc)) * a

    timestamper.toc()
    progress_callback(1, 'Finished')
    return dict(
        raw=simstate.SimImage(np.array(fmcw_lines),
                              azimuth_x[0], 0, azimuth_x[1] - azimuth_x[0], 1/state.fmcw_adc_frequency),
        rc=simstate.SimImage(np.array(rc_lines),
                             azimuth_x[0], r_vector[0], azimuth_x[1] - azimuth_x[0], r_vector[1] - r_vector[0]),
        ac=simstate.SimImage(np.array(image),
                             image_x[0], image_y[0], image_x[1] - image_x[0], image_y[1] - image_y[0]),
    )

_fmcw_cache = {}

def _fmcw_sim(flight_path, fmcw_samples, scene: simscene.SimulationScene, signal_speed, fmcw_start_frequency, fmcw_slope, fmcw_t, azimuth_3db_angle_deg):
    # We cache repeating call with identical parameters to speed up the simulation
    # TODO: Make this actually hashing sensibly for caching!
    cache_key = (
        hash(flight_path.data.tobytes()),
        fmcw_samples,
        hash(scene),
        signal_speed,
        fmcw_start_frequency,
        fmcw_slope,
        hash(fmcw_t.data.tobytes()),
        azimuth_3db_angle_deg,
    )
    cache_key = hash(cache_key)
    if cache_key in _fmcw_cache.keys():
        print("Using cached FMCW simulation")
        return _fmcw_cache[cache_key]

    # Antenna Diagram (dummy for now)
    def antenna_diagram(angle_rad: float) -> float:
        normalized = math.fabs(angle_rad * 180 / azimuth_3db_angle_deg / math.pi)
        if normalized < 2:
            return math.cos(math.pi * normalized / 4) ** 2
        else:
            return 0

    fmcw_lines = []

    for flight_point in flight_path:
        fmcw_line = np.array([0] * fmcw_samples)
        for reflector in scene.get_simple_reflectors():
            delta_vector_x = reflector.x - flight_point[0]
            delta_vector_y = reflector.y - flight_point[1]
            delta_vector_z = reflector.z - flight_point[2]
            delta_r = math.sqrt(delta_vector_x**2 +delta_vector_y**2 +delta_vector_z**2)
            delta_t = 2 * delta_r / signal_speed

            # b is the constant phase term
            b = (2 * math.pi * fmcw_start_frequency * delta_t) - (math.pi * fmcw_slope * delta_t * delta_t)
            # c is the angular frequency term
            c = 2 * math.pi * fmcw_slope * delta_t
            a = 0.5 * reflector.amplitude

            # Apply antenna diagram
            azimuth_angle = math.atan2(delta_vector_x, delta_vector_y)
            # elevation_angle = math.atan2(delta_vector.z, math.sqrt(delta_vector.x**2 + delta_vector.y**2))
            a = a * antenna_diagram(azimuth_angle)

            fmcw_line = fmcw_line + np.array([a * math.cos(b + c * t) for t in fmcw_t])
        fmcw_lines.append(fmcw_line)

    _fmcw_cache[cache_key] = fmcw_lines

    return fmcw_lines