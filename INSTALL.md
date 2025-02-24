# Veusz Installation

## Installation methods

1. Provided binaries for x86-64 Linux, Windows and MacOS - please go
   to the [download page](https://veusz.github.io/download/). See also
   below for further information.

2. Packages for your Linux distribution, provided by the
   distribution. These are often older than the current version.

3. [PPA](https://launchpad.net/~jeremysanders/+archive/ubuntu/ppa) for
   Ubuntu distribution, which we try to keep up to date.

4. [flatpak](https://flathub.org/apps/details/io.github.veusz.Veusz) runs on many linux distributions.

5. Anaconda conda-forge [package](https://anaconda.org/conda-forge/veusz).

6. Source build, download from the [download
   page](https://veusz.github.io/download/) and see below for build
   instructions.

## Provided binaries

### Windows

Simply run the setup.exe binary installer. Add the location of the
embed.py file to your `PYTHONPATH` of your Python installation if you
want to use the embedding module.

### Linux

Unpack the provided tar file and run the `veusz.exe` file inside
(please note that the `.exe` extension does not mean a Windows
executable here!):

    $ tar xf veusz-3.6-linux-x86_64.tar.xz   [change version here]
    $ cd veusz-3.6-linux-x86_64
    $ ./veusz.exe

Note that this may not work on all distributions due to glibc or other
library incompatibilities.

### MacOS

Simply drag the Veusz application into your Applications
directory. Please note that unfortunately due to signing requirements,
you will need to disable quarantine for it to run. Please see
[this github issue](https://github.com/veusz/veusz/issues/630#issuecomment-1305817737).

## Installing from source

### Requirements

* [Python](https://www.python.org/) >= 3.10
* [Qt](https://www.qt.io/developers/) >= 5.5
* [PyQt](https://www.riverbankcomputing.co.uk/software/pyqt/) >= 5.3
* [numpy](https://numpy.org/) >= 1.7

### Optional requirements

* [h5py](https://www.h5py.org/), for HDF5 file support
* [astropy](https://www.astropy.org/), for FITS file support
* [pyemf3](https://github.com/jeremysanders/pyemf3) >= 3.3, for EMF output
* [iminuit](https://github.com/iminuit/iminuit) >= 2, for better fitting
* [Ghostscript](https://www.ghostscript.com/), for EPS/PS output
* [dbus-python](https://dbus.freedesktop.org/doc/dbus-python/), for D-BUS support
* [Sphinx](https://www.sphinx-doc.org/en/master/), to rebuild the documentation

### Installation with setuptools

Veusz provides a standard setuptools `setup.py` file. If installing
this locally, it may be better to create a Python virtual environment
so that it is self-contained and does not interfere with existing
Python dependencies and packages.

### Building and running inside a virtual environment

An example use of a virtual environment to build veusz would be

    $ python3 -m venv /path/to/virtual/environment      [setup environment]
    $ source /path/to/virtual/environment/bin/activate  [activate it]
    $ pip3 install numpy qtpy PyQt6 astropy h5py        [install necessary requirements]
    $ pip3 install h5py astropy iminuit                 [install optional requirements]
    $ pip3 install https://github.com/jeremysanders/pyemf3.git [optional, for EMF output]
    $ tar xf veusz-3.6.tar.gz                           [unpack veusz source]
    $ cd veusz-3.6
    $ pip3 install -v .                                 [build and install veusz from current directory]

However, for the above to work requires a working Qt5 development
installation. This can be your Linux distribution's Qt packages,
binaries download from the Qt website, or a Qt build from source. A
quick way to install Qt binaries on different platforms can be using
the [aqtinstall](https://github.com/miurahr/aqtinstall) command line
installer.

### Installing into system Python directories

This needs to write permissions into the destination directory, so `sudo`
may be required.

    $ tar xf veusz-3.6.tar.gz                           [unpack veusz source]
    $ cd veusz-3.6
    $ pip3 install -v .                                 [build and install veusz from current directory]

On Ubuntu/Debian systems the following packages are necessary:

    $ apt install python3-all python3-astropy python3-h5py \
        python3-numpy python3-qtpy

And either

    $ apt install python3-pyqt6 python3-pyqt6.qtsvg

or

    $ apt install python3-pyqt5 python3-pyqt5.qtsvg

On Fedora the following are required:

    $ dnf install python3-setuptools python3-astropy \
        python3-numpy python3-h5py python3-qtpy

And either

    $ dnf install python3-qt6

or

    $ dnf install python3-qt5

Other Unix or Linux systems will likely contain the needed packages.

### Testing

After veusz has been installed into the Python path (in the standard
location or in `PYTHONPATH`), you can run the `runselftest.py`
executable in the `tests` directory. This will compare the generated
output of example documents with the expected output. The return code
of the `runselftest.py` script is the number of tests that have failed
(0 for success).

On Unix/Linux, Qt requires the `DISPLAY` environment to be set to an
X11 server for the self test to run. Packagers can use Xvfb in a
non-graphical environment to create a hidden X11 server:

    $ xvfb-run -a --server-args "-screen 0 640x480x24" \
        python3 tests/runselftest.py

Alternatively, the Qt platform can be switched to minimal to avoid the
use of X11:

    $ QT_QPA_PLATFORM=minimal python3 tests/runselftest.py

Please note that the environment variable `VEUSZ_INPLACE_TEST` is set,
then the `PYTHONPATH` are set to include the current working
directory, making it easier to run the self tests in automated scripts
without installation.

### Building and running in-place

If you don't want to install veusz fully or are doing development, it
can currently be run from its own directory. Before this can work, the
`helpers` modules must be compiled and copied into the appropriate
location.

    $ tar xzf veusz-3.6.tar.gz                [change version here]
    $ cd veusz-3.6/veusz/scripts              [change version here]
    $ python3 veusz

### Notes for packagers

* It is recommended to run the self test above (if possible).

* Veusz needs access to several subdirectories containing resource
  files, which are by default installed in the veusz module directory.
  These include the current version (`VERSION`), licence (`COPYING`),
  icons (`icons` subdirectory), user-interface description (`ui`
  subdirectory) and examples (`examples` subdirectory).  This location
  may not be desired by unix packagers, for example, who want to
  separate the code from the data files.

  It is possible to install these files in a different location by
  using the setup.py option `--veusz-resource-dir` (for example with
  `/usr/share/veusz`). If you do this, then you need to tell veusz
  where these resources are at runtime or when testing. This can be
  done by using a symlink `resources` in the veusz module
  directory which points to the location of these files and
  directories. Alternatively, the environment variable
  `VEUSZ_RESOURCE_DIR` can be set.

  There is an addition setup.py option `--disable-install-examples`
  which disables installation of the example files. This may be
  helpful for packagers who want to place the example files in
  `/usr/share/doc`. As veusz shows these files on the help menu, it is
  suggested that an `examples` symlink is added to the resources
  directory to point to the location of the example files.

- Veusz is a platform-independent python code and data files.

- Veusz includes a man page in `Documents/man-page/veusz.1`. This is
  not automatically installed by setuptools.

- A manual in HTML and PDF format can be found in `Documents/manual/`.
  This and the man page can be regenerated using the Makefile in
  Documents, if Sphinx is installed (`make clean; make`).

- Veusz also includes freedesktop mime, desktop and appdata files in
  the `support` subdirectory which can be installed to better
  integrate with desktop environments.

- Icons are also included in the icons directory with the names
  `veusz_16.png`, `_32`, `_48`, `_64` and `_128`. A scalable icon can
  be found in `veusz.svg`.

- Veusz will periodically (once per week) check for updates. This can
  be disabled by patching `veusz/utils/version.py` to set
  `disableVersionChecks=True`.

- Veusz will automatically send anonymous feedback (after
  confirmation) to the developers giving version information and
  counts of feature use. This can be disabled by patching
  `veusz/utils/feedback.py` to set `disableFeedback=True`.
