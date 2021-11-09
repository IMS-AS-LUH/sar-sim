import cmath
import math
from math import pi
from typing import Callable, NamedTuple

import numpy as np
import scipy.signal as signal
from . import operations
from . import sardata
from . import simscene

# Environmental Constants
SIGNAL_SPEED = 2.99709e8

CUDA_NUMBA_AVAILABLE = True

try:
    from numba import cuda
except ModuleNotFoundError:
    CUDA_NUMBA_AVAILABLE = False
    print('Numba (CUDA) not found. Falling back to CPU.')

from . import simstate, profiling

class SimResult(NamedTuple):
    raw: simstate.SimImage
    rc: simstate.SimImage
    ac: simstate.SimImage
    af: simstate.SimImage
    fpath_exact: np.ndarray
    fpath_distorted: np.ndarray

def run_sim(state: simstate.SarSimParameterState,
            scene: simscene.SimulationScene,
            timestamper:profiling.TimeStamper = None,
            progress_callback: Callable[[float, str], None] = None,
            loaded_data: sardata.SarData = None,
            gpu_id: int = 0) -> SimResult:
    timestamper = timestamper or profiling.TimeStamper()
    ac_use_cuda = CUDA_NUMBA_AVAILABLE
    use_loaded_data = loaded_data is not None

    if ac_use_cuda:
        cuda.select_device(gpu_id)

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

    ## PREPARE DATA ##
    timestamper.tic('Data Preparation')

    # Optimal azimuth x positions
    # TODO: Abbandon this?
    azimuth_x = np.linspace(state.azimuth_start_position, state.azimuth_stop_position, state.azimuth_count)

    print(
        f'Azimuth: {state.azimuth_count} Positions spaced {azimuth_x[1] - azimuth_x[0]}m yield {azimuth_x[-1] - azimuth_x[0]}m Track.')

    # Optimal flight path 3D [state.azimuth_count, 3]
    flight_path = None
    if use_loaded_data:
        assert loaded_data is not None
        exact_flight_path = loaded_data.flight_path
    else:
        exact_flight_path = _make_flight_path(state)
    # Simulate flight path as received by GPS
    distorted_fligh_path = _distort_path(exact_flight_path, state)
    flight_path = distorted_fligh_path if state.use_distorted_path else exact_flight_path

    # FMCW Simulation
    fmcw_bw = abs(state.fmcw_stop_frequency - state.fmcw_start_frequency)
    fmcw_up = True if state.fmcw_stop_frequency > state.fmcw_start_frequency else False
    fmcw_slope = (state.fmcw_stop_frequency - state.fmcw_start_frequency) / state.fmcw_ramp_duration
    fmcw_samples = math.ceil(state.fmcw_ramp_duration * state.fmcw_adc_frequency)
    fmcw_t = np.array([float(i) / state.fmcw_adc_frequency for i in range(fmcw_samples)])

    timestamper.toc()
    fmcw_lines = None
    if use_loaded_data:
        assert loaded_data is not None
        progress_callback(.05, 'Using loaded FMCW Data')
        fmcw_lines = loaded_data.fmcw_lines
    else:
        progress_callback(.05, 'FMCW Simulation')

        print(f'FMCW: {state.fmcw_start_frequency * 1e-9:.1f} GHz to {state.fmcw_stop_frequency * 1e-9:.1f} GHz '
              f'yields {fmcw_bw * 1e-9:.1f} GHz Bandwidth of {"up" if fmcw_up else "down"}-Ramp.')

        print(f'FMCW: {state.fmcw_ramp_duration * 1e3:.1f} ms Ramp-Time at {state.fmcw_adc_frequency * 1e-3:.1f} kHz '
              f'yields {fmcw_samples:d} samples.')

        timestamper.tic('FMCW Simulation')

        fmcw_lines = _fmcw_sim(flight_path, fmcw_samples, scene, SIGNAL_SPEED, state.fmcw_start_frequency, fmcw_slope, fmcw_t, state.azimuth_3db_angle_deg)

        timestamper.toc()

    ## Range Compression
    progress_callback(.25, 'Range Compression')
    timestamper.tic('Range Compression')
    # wnd = signal.hann(M=len(fmcw_lines[0]), sym=False)
    # wnd = signal.windows.tukey(M=len(fmcw_lines[0]), sym=False, alpha=0.25)

    rc_lines = _range_compression(state, fmcw_lines)
    timestamper.toc()

    ## Azimuth Compression
    progress_callback(.5, 'Azimuth Compression')
    timestamper.tic('Azimuth Compression')
    image_x, image_y, image, r_vector = _azimuth_compression(state, ac_use_cuda, flight_path, rc_lines)

    ## Autofocus
    progress_callback(.75, 'Autofocus')
    timestamper.tic('Autofocus')
    af_image = _autofocus_pafo(state, rc_lines, image, flight_path)

    timestamper.toc()
    progress_callback(1, 'Finished')
    return SimResult(
        raw=simstate.SimImage(np.array(fmcw_lines),
                              azimuth_x[0], 0, azimuth_x[1] - azimuth_x[0], 1/state.fmcw_adc_frequency),
        rc=simstate.SimImage(np.array(rc_lines),
                             azimuth_x[0], r_vector[0], azimuth_x[1] - azimuth_x[0], r_vector[1] - r_vector[0]),
        ac=simstate.SimImage(np.array(image),
                             image_x[0], image_y[0], image_x[1] - image_x[0], image_y[1] - image_y[0]),
        af=simstate.SimImage(np.array(af_image),
                             image_x[0], image_y[0], image_x[1] - image_x[0], image_y[1] - image_y[0]),
        fpath_exact=exact_flight_path,
        fpath_distorted=distorted_fligh_path,
    )

if CUDA_NUMBA_AVAILABLE:
    @cuda.jit() # type: ignore
    def _ac_kernel(flight_path_array, image, rc_lines, image_x, image_y, PC1, PC2, r0, r_scale, r_idx_max_sub1, beamlimit, ac_wnd):
        ix, iy = cuda.grid(2) # type: ignore
        if ix >= len(image_x) or iy >= len(image_y):
            return
        temp = complex(0, 0)
        x = image_x[ix]
        y = image_y[iy]
        for ia in range(0, len(flight_path_array)):
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

def _azimuth_compression(state: simstate.SarSimParameterState, ac_use_cuda: bool, flight_path: np.ndarray, rc_lines: np.ndarray, single_pulse_mode: bool = False):
    image_x = np.linspace(state.image_start_x, state.image_stop_x, state.image_count_x)
    image_y = np.linspace(state.image_start_y, state.image_stop_y, state.image_count_y)

    ac_wnd = state.azimuth_compression_window.factory(len(flight_path), False, state.azimuth_compression_window_parameter)

    if not single_pulse_mode:
        print(f'Image: {state.image_count_x}x{state.image_count_y} Pixels spaced {image_x[1] - image_x[0]:.3f}x{image_y[1] - image_y[0]:.3f}m.')

    image = np.zeros((state.image_count_x, state.image_count_y), dtype=complex)

    fmcw_slope = (state.fmcw_stop_frequency - state.fmcw_start_frequency) / state.fmcw_ramp_duration
    PC1 = -4 * math.pi * state.fmcw_start_frequency / SIGNAL_SPEED
    PC2 = 4 * math.pi * fmcw_slope / (SIGNAL_SPEED ** 2)

    r0 = 0
    fif = (state.fmcw_adc_frequency / 2) / (2 * fmcw_slope) * SIGNAL_SPEED
    rmax = r0 + fif
    r_vector = np.linspace(r0, rmax, len(rc_lines[0]))

    r_scale = r_vector[1] - r_vector[0]
    r_idx_max_sub1 = len(rc_lines[0]) - 2

    beamlimit = state.azimuth_compression_beam_limit / 180 * math.pi

    if ac_use_cuda:
        nx = len(image_x)
        ny = len(image_y)

        # If we just pass the numpy arrays to the GPU kernel, Numba will copy the
        # results back, even if they are unchanged. We manually allocate and copy
        # the memory here, to avoid this
        flight_path_gpu = cuda.to_device(flight_path)
        rc_lines_gpu = cuda.to_device(rc_lines)
        image_x_gpu = cuda.to_device(image_x)
        image_y_gpu = cuda.to_device(image_y)
        ac_wnd_gpu = cuda.to_device(ac_wnd)
        image_gpu = cuda.device_array_like(image) # empty, because it is read-only

        blocksize = (16, 16)
        gridsize = (math.ceil(nx / blocksize[0]), math.ceil(ny / blocksize[1]))
        _ac_kernel[gridsize, blocksize](flight_path_gpu, image_gpu, rc_lines_gpu, image_x_gpu, image_y_gpu, PC1, PC2, r0, # type: ignore
            r_scale, r_idx_max_sub1, beamlimit, ac_wnd_gpu)
        
        # copy back image
        image_gpu.copy_to_host(image)
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
                    image[ix][iy] = image[ix][iy] + sample * cmath.exp(complex(0, pc)) * a
    
    return image_x, image_y, image, r_vector

def _range_compression(state: simstate.SarSimParameterState, fmcw_lines: list) -> np.ndarray:
    rc_used_bandwidth = state.range_compression_used_bandwidth / 100
    if rc_used_bandwidth >= 0.99999999:
        wnd = state.range_compression_window.factory(len(fmcw_lines[0]), False, state.range_compression_window_parameter)
    else:
        original_length = len(fmcw_lines[0])
        used_part = round(rc_used_bandwidth*original_length)
        used_offset = math.floor((original_length-used_part)/2)
        wnd = state.range_compression_window.factory(used_part, False, state.range_compression_window_parameter)
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
    
    return np.array(rc_lines)

_fmcw_cache = {}

def _fmcw_sim(flight_path, fmcw_samples, scene: simscene.SimulationScene, signal_speed, fmcw_start_frequency, fmcw_slope, fmcw_t, azimuth_3db_angle_deg) -> list:
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

def _distort_path(flight_path: np.ndarray, state: simstate.SarSimParameterState) -> np.ndarray:
    """Return a down-sampled version of the flight path, as it might have been received by a GPS receiver"""
    num_positions = int(state.distortion_sample_freqency * (state.azimuth_stop_position -state.azimuth_start_position))
    random_factor = state.distortion_random_factor

    indices = np.linspace(0, flight_path.shape[0]-1, num_positions, dtype=np.integer)
    support = np.arange(flight_path.shape[0])
    rand = np.random.default_rng(5019)

    flight_path = np.array([
        np.interp(support, indices, flight_path[indices, 0]),
        np.interp(support, indices, flight_path[indices, 1]),
        np.interp(support, indices, flight_path[indices, 2]),
    ]).T + rand.normal(scale=random_factor, size=flight_path.shape)

    return flight_path

def _make_flight_path(state: simstate.SarSimParameterState) -> np.ndarray:
    # optimal x positions
    azimuth_x = np.linspace(state.azimuth_start_position, state.azimuth_stop_position, state.azimuth_count)
    # add constant y and z position
    path = np.array([
        [x, -state.flight_distance_to_scene_center, state.flight_height]
        for x in azimuth_x
    ])
    # add wiggles
    unscaled_offsets = np.sin(np.stack((azimuth_x, azimuth_x, azimuth_x)).T * np.array([state.flight_wiggle_frequency_azimuth, state.flight_wiggle_frequency_range, state.flight_wiggle_frequency_height]))
    scaled_offsets = unscaled_offsets * np.array([state.flight_wiggle_amplitude_azimuth, state.flight_wiggle_amplitude_range, state.flight_wiggle_amplitude_height])
    path = path + state.flight_wiggle_global_scale * scaled_offsets
    return path

def _autofocus_pafo(state: simstate.SarSimParameterState, rc_lines: np.ndarray, ac_image: np.ndarray, flight_path: np.ndarray, rounds: int = 1, samples: int = 8, iterations = 2) -> np.ndarray:
    """Perform the PAFO autofocus.
    This implementation is currently CUDA-only
    :param rounds: Number of overall autofocs iterations (normally 1)
    :param samples: Number of parallel samples to use in each iteration (usually 8)
    :param iterations: Number of sampling iterations per phase (normally 2)
    :note: based on https://git.ims-as.uni-hannover.de/sar/stuff/theory/-/blob/master/matlab/runSoftwarePafo.m and BA Fallnich
    """
    assert samples % 2 == 0, "samples must be even"
    np.seterr('raise')
    # TODO: Move to CUDA
    # TODO: Port over the "valid" variable to only consider pixels in the sum etc. that are not exluded by the beamlimit. This could make things faster
    # make ac_image linear
    ac_image = np.ravel(ac_image)

    optimal_phases = np.zeros((rounds, len(rc_lines)))
    # AF can have multiple overall iterations
    for round in range(rounds):
        # Back-Project individual apertures and then find correct focus phase
        for az_index in range(len(rc_lines)):
            # Generate single pulse image
            _, _, single_pulse, _ = _azimuth_compression(state, True, flight_path[[az_index]], rc_lines[[az_index]], single_pulse_mode=True)
            single_pulse = np.ravel(single_pulse) # make linear

            # We do iterative runs of "parallel" phase search

            # Loop over the sampling rounds
            last_optimum = math.nan
            min_index = -1
            sample_spacing = math.nan
            metric_sums = np.array([])
            assert iterations > 1
            for iteration in range(iterations):
                # Determine points to evaluate
                sample_points = np.array([])
                sample_spacing = 2*pi / ((samples * (samples-2)) ** iteration)
                if iteration == 0:
                    sample_points = -pi + np.arange(samples) * sample_spacing
                else:
                    assert not math.isnan(last_optimum)
                    index_center = samples // 2
                    indices = np.arange(-index_center, index_center+1) # -N/2, ..., -1, 0, 1, ..., N/2
                    assert len(indices) == samples + 1
                    sample_points = last_optimum + indices * sample_spacing

                # Get the "subtract, then add rotated" factors
                candidate_factors = np.expm1(1j * sample_points)
                # Build Image and apply sharpnes metric (with summing)
                metric_sums = -np.sum(np.abs(ac_image + single_pulse * candidate_factors[:, np.newaxis]) ** 4, axis=1)
                assert metric_sums.shape[0] == len(candidate_factors)
                # Minimum point
                min_index = np.argmin(metric_sums)
                last_optimum = sample_points[min_index]

            # make sure the minimum is not on the edge
            if min_index == 0:
                min_index = 1
            if min_index == samples:
                min_index = samples - 1
            
            interpol_indices = [min_index-1, min_index, min_index+1]
            interpol_values = metric_sums[interpol_indices]
            
            # Parabolic interpolation
            if max(interpol_values) - min(interpol_values) > 1e-4:
                optimal_phase = last_optimum + sample_spacing/2 - sample_spacing*(interpol_values[2]-interpol_values[1])/(interpol_values[0]+interpol_values[2]-2*interpol_values[1])
            else: #parabola degenerated to a line
                optimal_phase = last_optimum

            # Apply:
            correction_factor = np.expm1(1j*optimal_phase)
            ac_image = ac_image + single_pulse * correction_factor
            optimal_phases[round, az_index] = optimal_phase

            if az_index % 8 == 0:
                print(f"{round:4}.{az_index:4} ({100 * (az_index/(len(rc_lines)*rounds)) + round/rounds:.3} %) of PAFO")

    return ac_image