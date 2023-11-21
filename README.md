# SAR Simulator

The SAR Simulator is a graphical tool for FMCW radar simulation and interactive exploration of parameters.
It can simulate FMCW radar signals based on an algorithmic scene description.
The resulting signal is then range-compressed and and image is reconstructed using backprojection (azimuth compression).
Various parameters can be changed using the GUI, allowing for interactive exploration.

Optionally existing signals (before or after range compression) can be loaded and processed with the simulator.
When available, GPU acceleration (using [Numba](https://numba.pydata.org/)/CUDA) is used.

This tool was created by the Architectures and Systems Group of the Institute of Microelectronic Systems ([IMS/AS](https://www.ims.uni-hannover.de/de/institut/architekturen-und-systeme/)) at the [Leibniz University](https://www.uni-hannover.de) in Germany.

![Screenshot](/doc/screenshot_1.png?raw=true)

## Getting Started

### Installation

Create virtual environmement (venv), install requirements:

    python3 -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt

Run as python module (Note: only usable with GUI currently):

    python3 -m sarsim --gui

Select parameters on the left, hit RUN SIM on bottom left.
Simulation will be done in backgound, status at bottom.
Graphs will be updated automatically when done.
See also CLI console output for status and information.
If not using GPU acceleration (CUDA), azimuth compression will take a while.

The SAR Simulator was tested on Ubuntu 20.04 LTS but should work on most Linux Distributions, and even on Windows.

## Features
- Exact FMCW Signal generation in phase-space
- Range and Azimuth compression
- Live-Update of preview graphs
- CUDA Acceleration of Azimuth Compression if possible

### Development tools
- Run ```python3 -m sarsim --write-stubs``` to enable IDE support for better autocompletion

## Citation

If you use the SAR Simulator in scientific work, please cite our paper "[Interactive synthetic aperture radar simulator generating and visualizing realistic FMCW data](https://ieeexplore.ieee.org/document/10106365)".

```bibtex
@INPROCEEDINGS{fahnemann2022interactive,
  author={Fahnemann, C. and Rother, N. and Blume, H.},
  booktitle={International Conference on Radar Systems (RADAR 2022)}, 
  title={Interactive synthetic aperture radar simulator generating and visualizing realistic FMCW data}, 
  year={2022},
  volume={2022},
  number={},
  pages={725-730},
  doi={10.1049/ICP.2023.1281}}
```
