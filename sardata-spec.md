# Specification for the IMS SARDATA File Format

**Status of this document**
This specification shall be treated as a living standard.
It describes the current format understood by the simulator, as well as recommended practices.
The format is intentionally designed to allow for extension in many points, and some current and future implementations will use additional files and fields not described here.
When possible, this documentation should be updated with these extensions.

## Introduction, Purpose

The IMS SARDATA Format is a file format designed at the [Institute for Microelectronic Systems, Architecture and Systems Group](https://ims.uni-hannover.de) at the Leibniz University Hannover to store files generated during the research on Synthetic Aperture Radar.
The format was designed with the following goals in mind:

- Extensibility
- Easy to implement
- Can be used on embedded platforms

### Extensibility

As we are doing research, requirements change quickly and new data needs to be stored.
To accommodate for this, the format should be easily extensible by adding new data types, without breaking old implementations.
While this produces some kind of chaos and incompatibility, it has proven to be crucial to use a format in practice.

### Easy to implement

When trying a new algorithm, hardware platform etc. it should be easy to read the existing files.
One should not have to spend a lot of time to implement a parser for the data files.

### Can be used on embedded platforms

We often use embedded platforms, where powerful languages like Python might not be available.
It should be therefore possible to read the file in C without too much effort.
This goes hand in hand with the aforementioned goal.

## File format overview

A SARDATA "file" is actually a *directory* at the file system level.
The folder acts as a simple container format to aggregate several related files.
When there is the need to have a single file, the folder may be packed using `zip` or `tar`.
Please note that most implementations cannot read packed SARDATA files.

Every SARDATA file (or directory, really) should have **an extension of `.sardata`**.
This directory **may contain arbitrary files**, although some are part of the specification below.
Adding more files can be useful to extend the information stored in a SARDATA file, or just to save some related results together with the source files.
It is recommended to only store additional files in the directory, if the are directly a "derived product" of the original data.
It is not recommend to store scripts and alike that can be used on several files in the directory.

At minimum, a SARDATA file should contain a `params.cfg` (describing the files content) and a `*.bin` file, containing the actual data.
In most cases this will be `fmcw.bin` file containing the raw real-valued FMCW samples (before range compression).
In some cases (for converted data) there might be `range_comp.bin` file containing already range-compressed complex samples.

## Known files

The following files are known to be stored in SARDATA files:

### params.cfg

**May have other names, see below**

A file describing the metadata of the SAR capture.
The file is in INI format and usually read and written using [Pythons `configparser`](https://docs.python.org/3/library/configparser.html).
An implementation may add additional fields to store more metadata.
All fields are optional, although the implementation may refuse to read a file if too many information is missing.
Currently, the following fields are known:

| Section   | Field Name        | Description |
|-----------|-------------------|-------------|
| general   | radartype         | Type of radar sensor used for capture. `80` and `144` indicate the corresponding new (v2) RUB FMCW sensors.
|           | capture_time      | ISO date of data acquisition.
|           | conversion_time   | ISO date of data conversion (from other/older format).
|           | original_filename | Filename of the conversion source.
|           | data_is_range_compressed | If true, the raw data is already range compressed. In this case, usually no `fmcw.bin` file is present, but a `range_comp.bin` file. See below for its format.
| params    | start_frequency   | Start frequency of FMCW ramp, in Hz.
|           | stop_frequency    | Stop frequency of FMCW ramp, in Hz.
|           | ramp_duration     | Duration of FMCW ramp, in seconds.
|           | adc_frequency     | Sample frequency of the ADC, in Hz.
|           | azimuth_count     | Number of azimuth samples.
|           | gbp_image_region  | Tuple of the recommended image region in meters, given as (Xmin, Ymin, Xmax, Ymax), e.g. `(0.25, 0.5, 2.75, 3.0)`.
|           | gbp_image_z       | Recommended height of image plane above ground, in meters. Assume `0` if missing.
|           | gbp_beam_limit    | Recommended beamlimit for processing, usually the opening angle of the antenna, in degrees.
|           | sensor_range_delay| Offset to the first range samples, in meters. Often known as `r0`.
|           | signal_speed      | Speed of light in the medium, in m/s.
|           | *                 | The IMS SAR Demo script will write all parameters (also the ones for real-time processing) to the params section.
|           | center_frequency  | For range-compressed data: Center frequency of the transmitter, in Hz.
|           | bandwidth         | For range-compressed data: Bandwidth of the transmitter, in Hz.
| fpath     | x                 | Comma delimited list of antenna coordinates, length should match `azimuth_count`. In meters, in the same coordinate system as `gbp_image_region`. This is the cross-range direction.
|           | y                 | See above. This is the range-direction.
|           | z                 | See above. This is the height.

### fmcw.bin

**May have other names, see below**

This file contains the raw, real-valued samples from the FMCW sensor.

The data is stored as a consecutive stream of 32-bit integer values (4 bytes per value).
The fast changing index is range, the slow changing index is azimuth, i.e. a list of range lines.

### data.mat

Matlab file, containing the same data as the other files.
Deprecated.

### data.pickle

[Pickle](https://docs.python.org/3/library/pickle.html) file, containing the same data as the other files.
Deprecated.

### info.txt

Text file containing a small description of the file.

### preview.png

PNG image with a preview of the fully processed data.
This is usually the output from the real-time processing during the capture.

### cam_picture.jpg

Camera picture showing a photo of the captured scene.
This version should be (roughly) cropped/corrected to show the same view as the output radar image.

### cam_picture_raw.jpg

Raw photo from the camera, before cropping and lens correction.
*Notice:* This may show more than intended, check for privacy before distribution.

### range_comp.bin

This file contains the complex output of the range compression.
If this file is present, `data_is_range_compressed` should be set to true in `params.cfg`.
This is usually used if the data was not originally captured by (our) FMCW sensor but converted from some other source.

The data is stored as a continuous stream of 32-bit floats (4 bytes per value).
The real value of every samples comes first, followed by the imaginary part.
The fast changing index is range, the slow changing index is azimuth, i.e. a list of range lines.

## Backward compatibility

### Other file names

In some implementations, the `.cfg` and `.bin` files have the same name as the `.sardata` file, eg. `capture_1234.sardata` would contain `capture_1234.cfg` and `capture_1234.bin`.
When the toplevel file is renamed (e.g. to `city_no_reflectors.sardata`) this link often breaks.
It is therefore not recommended to create files like this anymore, the embedded files should always be called `params.cfg` and `fmcw.bin`.
For compatibility reasons, implementations should try to read this kind of files.
When there is only a single `.cfg` file, threat it as `params.cfg`, when there is only a single `.bin` file threat it as `fmcw.bin`.

## Authors

This document was authored by [Niklas Rother](mailto:rother@ims.uni-hannover.de).
The file format was conceived by [Niklas Rother](mailto:rother@ims.uni-hannover.de) and [Christian Fahnemann](mailto:fahnemann@ims.uni-hannover.de)