# [Veusz 4.2](https://veusz.github.io/)

Veusz is a scientific plotting package.  It is designed to produce
publication-ready PDF or SVG output. Graphs are built-up by combining
plotting widgets. The user interface aims to be simple, consistent and
powerful.

Veusz provides GUI, Python module, command line, scripting, DBUS and
SAMP interfaces to its plotting facilities. It also allows for
manipulation and editing of datasets. Data can be captured from
external sources such as Internet sockets or other programs.

Changes in 4.2:
  * Fix for double scaled 3D point marker borders (Takuro Hosomi)
  * Allow negative offsets for some labels (Takuro Hosomi)
  * Allow iterative change of properties of multiple widgets (Takuro Hosomi)
  * Strip BOMs from Veusz and imported files
  * Fix binary import plugin
  * Fix capture dialog
  * Prefer tomllib on Python 3.11 (Alexandre Detiste)
  * Clean up text for translation and document use of weblate in README
  * Fix inactive 2D data import range
  * Update GPL address

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

## Installation
Please see the file `INSTALL.md` included in the distribution for installation details, or go to the [download page](https://veusz.github.io/download/).

## License
Veusz is Copyright (C) 2003-2025 Jeremy Sanders
 and contributors. It is licensed under the [GPL version 2 or greater](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html).

## Source code
The latest source code can be found in [this GitHub repository](https://github.com/veusz/veusz).
Translations are welcome to be made [using Weblate](https://hosted.weblate.org/projects/veusz/).
