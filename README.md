# napari-sim-processor

[![License](https://img.shields.io/pypi/l/napari-sim-processor.svg?color=green)](https://github.com/andreabassi78/napari-sim-processor/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-sim-processor.svg?color=green)](https://pypi.org/project/napari-sim-processor)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-sim-processor.svg?color=green)](https://python.org)
[![tests](https://github.com/andreabassi78/napari-sim-processor/workflows/tests/badge.svg)](https://github.com/andreabassi78/napari-sim-processor/actions)
[![codecov](https://codecov.io/gh/andreabassi78/napari-sim-processor/branch/main/graph/badge.svg)](https://codecov.io/gh/andreabassi78/napari-sim-processor)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-sim-processor)](https://napari-hub.org/plugins/napari-sim-processor)

A Napari plugin for the reconstruction of Structured Illumination microscopy (SIM) data with GPU acceleration (with pytorch, if installed).
Currently supports:    
   - conventional data with improved resolution in 1D (1 angle, 3 phases)
   - conventional data (3 angles, 3 phases)
   - hexagonal SIM (1 angle, 7 phases).

The SIM processing widget accepts image stacks organized in 5D (angle,phase,z,y,x).

For raw image stacks with multiple z-frames each plane is processed as described in:
	https://doi.org/10.1098/rsta.2020.0162
    
A reshape widget is availbale, to easily reshape the data if they are organized differently than 5D (angle,phase,z,y,x)
    
Support for N angles, M phases is in progress.
Support for 3D SIM with enhanced resolution in all directions is not yet available.
Multicolor reconstruction is not yet available.  

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/index.html
-->

## Installation

You can install `napari-sim-processor` via [pip]:

    pip install napari-sim-processor


To install latest development version :

    pip install git+https://github.com/andreabassi78/napari-sim-processor.git


## Usage

1) Open napari. 

2) Launch the reshape and sim-processor widgets.

3) Open your raw image stack (using the napari built-in or your own file openers).

4) If your image ordered as a 5D stack (angle, phase, z-frame, y, x) go to point 4. 

5) In the reshape widget, select the number of acquired angles, phases, and frames and press Reshape Stack. Note that the label axis of the viewer will be updated.

6) In the sim-reconstruction widget press the Select image layer button. Note that the number of phases and angles will be updated. 

7) Choose the correct parameters of the SIM acquisition (NA, pixelsize, M, etc.) and processing parameters (alpha, beta, w, eta, group).
	w: parameter of the weiner filter.
	eta: constant used for calibration. It should be slightly smaller than the carrier frequency (in pupil radius units). 
	group: for stacks with multiple z-frames, it is the number of frames that are used together for the calibration process.
	For details on the other parameters see https://doi.org/10.1098/rsta.2020.0162.

8) Calibrate the SIM processor, pressing the correspondent button. This will find the carrier frequencies (red circles if the Show Carrier checkbox is selected), the modulation amplitude and the phase, using cross correlation analysis.

9) Click on the checkboxes to show the power spectrum of the image or the cross-correlation, to see if the carrier frequency is found correctly

10) Run the reconstruction of a single plane (SIM reconstruction) or of a stack (Stack reconstruction). Click on the Batch reconstruction checkbox in order to process an entire stack in one shot. Click on the pytorch checkbox for gpu acceleration.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-sim-processor" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/andreabassi78/napari-sim-processor/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
