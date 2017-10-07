#    Copyright (C) 2014 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

# note: no future statements here for backward compatibility

import sys
import os.path
import traceback
import io
import numpy as N

from .. import qtall as qt4
from .. import setting
from .. import utils

from ..compat import cexec, cstrerror, cbytes, cexceptionuser
from .commandinterface import CommandInterface
from . import datasets

# loaded lazily
h5py = None

def _(text, disambiguation=None, context='DocumentLoader'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class LoadError(RuntimeError):
    """Error when loading document."""
    def __init__(self, text, backtrace=''):
        RuntimeError.__init__(self, text)
        self.backtrace = backtrace

def bconv(s):
    """Sometimes h5py returns non-unicode strings,
    so hack to decode strings if in wrong format."""
    if isinstance(s, cbytes):
        return s.decode('utf-8')
    return s

def _importcaller(interface, name, callbackimporterror):
    """Wrap an import statement to check for IOError."""
    def wrapped(*args, **argsk):
        while True:
            try:
                getattr(interface, name)(*args, **argsk)
            except IOError as e:
                errmsg = cexceptionuser(e)
                fnameidx = interface.import_filenamearg[name]
                assert fnameidx >= 0
                filename = args[fnameidx]
                raiseerror = True
                if callbackimporterror:
                    # used by mainwindow to show dialog and get new filename
                    fname = callbackimporterror(filename, errmsg)
                    if fname:
                        # put new filename into function argument list
                        args = list(args)
                        args[fnameidx] = fname
                        raiseerror = False
                if raiseerror:
                    # send error message back to UI
                    raise LoadError(
                        _("Error reading file '%s':\n\n%s") %
                        (filename, errmsg))
            else:
                # imported ok
                break
    return wrapped

def executeScript(thedoc, filename, script,
                  callbackunsafe=None,
                  callbackimporterror=None):
    """Execute a script for the document.

    This handles setting up the environment and checking for unsafe
    commands in the execution.

    filename: filename to supply in __filename__
    script: text to execute
    callbackunsafe: should be set to a function to ask the user whether it is
      ok to execute any unsafe commands found. Return True if ok.
    callbackimporterror(filename, error): should be set to function to return new filename in case of import error, or False if none

    User should wipe docment before calling this.
    """

    def genexception(exc):
        info = sys.exc_info()
        backtrace = ''.join(traceback.format_exception(*info))
        return LoadError(cexceptionuser(exc), backtrace=backtrace)

    # compile script and check for security (if reqd)
    unsafe = [setting.transient_settings['unsafe_mode']]
    while True:
        try:
            compiled = utils.compileChecked(
                script, mode='exec', filename=filename,
                ignoresecurity=unsafe[0])
            break
        except utils.SafeEvalException:
            if callbackunsafe is None or not callbackunsafe():
                raise LoadError(_("Unsafe command in script"))
            # repeat with unsafe mode switched on
            unsafe[0] = True
        except Exception as e:
            raise genexception(e)

    env = thedoc.evaluate.context.copy()
    interface = CommandInterface(thedoc)

    # allow safe commands as-is
    for cmd in interface.safe_commands:
        env[cmd] = getattr(interface, cmd)

    # define root node
    env['Root'] = interface.Root

    # wrap unsafe calls with a function to check whether ok
    def _unsafecaller(func):
        def wrapped(*args, **argsk):
            if not unsafe[0]:
                if callbackunsafe is None or not callbackunsafe():
                    raise LoadError(_("Unsafe command in script"))
                unsafe[0] = True
            func(*args, **argsk)
        return wrapped
    for name in interface.unsafe_commands:
        env[name] = _unsafecaller(getattr(interface, name))

    # override import commands with wrapper
    for name in interface.import_commands:
        env[name] = _importcaller(interface, name, callbackimporterror)

    # get ready for loading document
    env['__file__'] = filename
    # allow import to happen relative to loaded file
    interface.AddImportPath( os.path.dirname(os.path.abspath(filename)) )

    with thedoc.suspend():
        try:
            # actually run script text
            cexec(compiled, env)
        except LoadError:
            raise
        except Exception as e:
            raise genexception(e)

def loadHDF5Dataset1D(datagrp):
    args = {}
    # this weird usage of sets is to work around some sort of weird
    # error where h5py gives an error when doing 'a' in datagrp
    # this gives error: 'perr' in datagrp
    parts = set(datagrp) & set(('data', 'serr', 'perr', 'nerr'))
    for v in parts:
        args[v] = N.array(datagrp[v])
    return datasets.Dataset(**args)

def loadHDF5Dataset2D(datagrp):
    args = {}
    parts = set(datagrp) & set(
        ('data', 'xcent', 'xedge', 'ycent', 'yedge', 'xrange', 'yrange'))
    for v in parts:
        args[v] = N.array(datagrp[v])
    return datasets.Dataset2D(**args)

def loadHDF5DatasetDate(datagrp):
    return datasets.DatasetDateTime(data=datagrp['data'])

def loadHDF5DatasetText(datagrp):
    data = [d.decode('utf-8') for d in datagrp['data']]
    return datasets.DatasetText(data=data)

def loadHDF5Datasets(thedoc, hdffile):
    """Load all the Veusz datasets in the HDF5 file."""
    alldatagrp = hdffile['Veusz']['Data']

    datafuncs = {
        '1d': loadHDF5Dataset1D,
        '2d': loadHDF5Dataset2D,
        'date': loadHDF5DatasetDate,
        'text': loadHDF5DatasetText,
    }

    for name in alldatagrp:
        datagrp = alldatagrp[name]
        datatype = bconv(datagrp.attrs['vsz_datatype'])
        veuszname = utils.unescapeHDFDataName(bconv(name))

        dataset = datafuncs[datatype](datagrp)
        thedoc.setData(veuszname, dataset)

def tagHDF5Datasets(thedoc, hdffile):
    """Tag datasets loaded from HDF5 file."""
    tags = hdffile['Veusz']['Document']['Tags']
    for tag in tags:
        vsztag = bconv(tag)
        datasets = tags[tag]
        for name in datasets:
            dsname = name.decode('utf-8')
            thedoc.data[dsname].tags.add(vsztag)

def loadHDF5Doc(thedoc, filename,
                callbackunsafe=None,
                callbackimporterror=None):
    """Load an HDF5 of the name given."""

    try:
        global h5py
        import h5py
    except ImportError:
        raise LoadError(_("No HDF5 support as h5py module is missing"))

    with thedoc.suspend():
        thedoc.wipe()
        thedoc.filename = filename
        hdffile = h5py.File(filename, 'r')

        try:
            vszformat = hdffile['Veusz'].attrs['vsz_format']
            vszversion = hdffile['Veusz'].attrs['vsz_version']
        except KeyError:
            raise LoadError(
                _("HDF5 file '%s' is not a Veusz saved document") %
                os.path.basename(filename))

        maxformat = 1
        if vszformat > maxformat:
            raise LoadError(
                _("This document version (%i) is not supported. "
                  "It was written by Veusz %s.\n"
                  "This Veusz only supports document version %i.") %
                (vszformat, vszversion, maxformat))

        # load document
        script = hdffile['Veusz']['Document']['document'][0].decode('utf-8')
        executeScript(
            thedoc, filename, script,
            callbackunsafe=callbackunsafe,
            callbackimporterror=callbackimporterror)

        # then load datasets
        loadHDF5Datasets(thedoc, hdffile)
        # and then tag
        tagHDF5Datasets(thedoc, hdffile)

        hdffile.close()

def loadDocument(thedoc, filename, mode='vsz',
                 callbackunsafe=None,
                 callbackimporterror=None):
    """Load document from file.

    mode is 'vsz' or 'hdf5'
    """

    if mode == 'vsz':
        try:
            with io.open(filename, 'rU', encoding='utf-8') as f:
                script = f.read()
        except EnvironmentError as e:
            raise LoadError( _("Cannot open document '%s'\n\n%s") %
                             (os.path.basename(filename), cstrerror(e)) )
        except UnicodeDecodeError:
            raise LoadError( _("File '%s' is not a valid Veusz document") %
                             os.path.basename(filename) )

        thedoc.wipe()
        thedoc.filename = filename
        executeScript(
            thedoc, filename, script,
            callbackunsafe=callbackunsafe,
            callbackimporterror=callbackimporterror)

    elif mode == 'hdf5':
        loadHDF5Doc(
            thedoc, filename,
            callbackunsafe=callbackunsafe,
            callbackimporterror=callbackimporterror)

    else:
        raise RuntimeError('Invalid load mode')

    thedoc.setModified(False)
    thedoc.clearHistory()
