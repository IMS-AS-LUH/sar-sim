# SAR Simulator

This is a tool to simulate the physical radar signal based on variable
parameters and then applying basic image formation algorithms.

### How to run
Install dependencies:
- Python packages: ```pyqtgraph``` 
- Related packages: ```pyqt5```, ```numpy```, ```scipy```
- Optional packages: ```numba``` (for CUDA)

Run as python module (Note: only usable with GUI currently):

    python3 -m sarsim --gui

Select parameters on the left, hit RUN SIM on bottom left.
Simulation will be done in backgound, status at bottom.
Graphs will be updated automatically when done.
See also CLI console output for status and information.
If not using GPU acceleration (CUDA), azimuth compression will take a while.

## Features
- Exact FMCW Signal generation in phase-space
- Range and Azimuth compression
- Live-Update of preview graphs
- CUDA Acceleration of Azimuth Compression if possible

### Development tools
- Run ```python3 -m sarsim --write-stubs``` to enable IDE support for better autocompletion

