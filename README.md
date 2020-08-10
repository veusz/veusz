# [Veusz 3.2.1](https://veusz.github.io)

Veusz is a scientific plotting package.  It is designed to produce
publication-ready PDF or SVG output. Graphs are built-up by combining
plotting widgets. The user interface aims to be simple, consistent and
powerful.

Veusz provides GUI, Python module, command line, scripting, DBUS and
SAMP interfaces to its plotting facilities. It also allows for
manipulation and editing of datasets. Data can be captured from
external sources such as Internet sockets or other programs.

## Changes in 3.2.1
  * Fix too large page size in SVG export
  * Reenable compression for PNG export
  * Fix crash in HDF5 import dialog
  * If filename extension is missing in export dialog, add it and avoid crash
  * Take account of QT_LIBINFIX setting for unusual Qt installs
  * Add \wtilde text command to place a tilde over text

## Changes in 3.2
  * Add ability to plot image widget using boxes rather than a bitmap, with new drawing mode option
  * Add widget order option in key widget
  * Export dialog now uses multiple threads
  * Python 3.9 compatibility fixes
  * Show exception dialog if crash occurs outside main thread
  * Added Brazilian Portuguese description for desktop file
  * Use python3 by default for in-place run
  * Fix icons in tutorial
  * Fix case when positions in bar widget are set, then removed
  * Truly all files are shown in import dialog, if requested
  * Fix browse button in export dialog
  * Fix stylesheet for polygon widget
  * Fix invalid escape sequences warnings
  * Fix parametric date creation for non-English locales

## Features of package:

### Plotting features:
  * X-Y plots (with errorbars)
  * Line and function plots
  * Contour plots
  * Images (with colour mappings and colorbars)
  * Stepped plots (for histograms)
  * Bar graphs
  * Vector field plots
  * Box plots
  * Polar plots
  * Ternary plots
  * Plotting dates
  * Fitting functions to data
  * Stacked plots and arrays of plots
  * Nested plots
  * Plot keys
  * Plot labels
  * Shapes and arrows on plots
  * LaTeX-like formatting for text
  * Multiple axes
  * Axes with steps in axis scale (broken axes)
  * Axis scales using functional forms
  * Plotting functions of datasets
  * 3D point plots
  * 3D surface plots
  * 3D function plots
  * 3D volumetric plots

### Input and output:
  * PDF/EPS/PNG/SVG/EMF export
  * Dataset creation/manipulation
  * Embed Veusz within other programs
  * Text, HDF5, CSV, FITS, NPY/NPZ, QDP, binary and user-plugin importing
  * Data can be captured from external sources

### Extending:
  * Use as a Python module
  * User defined functions, constants and can import external Python functions
  * Plugin interface to allow user to write or load code to
    - import data using new formats
    - make new datasets, optionally linked to existing datasets
    - arbitrarily manipulate the document
  * Scripting interface
  * Control with DBUS and SAMP

### Other features:
  * Data filtering and manipulation
  * Data picker
  * Interactive tutorial
  * Multithreaded rendering

## Requirements for source install:
  * [Python](https://www.python.org/) 2.x (2.7 or greater required) or 3.x (3.3 or greater required)
  * [Qt](https://www.qt.io/) >= 5.5 (free edition)
  * [PyQt](http://www.riverbankcomputing.co.uk/software/pyqt/) >= 5.2  (Qt and SIP is required to be installed first)
  * [SIP](http://www.riverbankcomputing.co.uk/software/sip/) >= 4.15
  * [Numpy](http://numpy.scipy.org/) >= 1.7

## Optional requirements:
* [h5py](https://www.h5py.org/) (optional for HDF5 support)
* [pyemf](http://pyemf.sourceforge.net/) >= 2.0.0 (optional for EMF export)
  - [Python 3 port in development](https://github.com/jeremysanders/pyemf)
* [iminuit](https://github.com/scikit-hep/iminuit) or PyMinuit >= 1.12 (optional improved fitting)
* [dbus-python](https://dbus.freedesktop.org/doc/dbus-python/), for dbus interface
* [astropy](https://www.astropy.org/) (optional for VO table import or FITS import)
* [SAMPy](https://pypi.python.org/pypi/sampy/) or astropy >= 0.4 (optional for SAMP support)
* [Ghostscript](https://www.ghostscript.com/) (for EPS/PS output)

## License
Veusz is Copyright (C) 2003-2020 Jeremy Sanders <jeremy@jeremysanders.net>
 and contributors. It is licensed under the [GPL version 2 or greater](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html).

The latest source code can be found in [this GitHub repository](https://github.com/veusz/veusz).
