#    Copyright (C) 2011 Jeremy S. Sanders
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

"""Parameters for import routines."""

from __future__ import division, print_function
import sys
import copy

from ..compat import citems, cstr
from .. import utils

class ImportingError(RuntimeError):
    """Common error when import fails."""

class ImportParamsBase(object):
    """Import parameters for the various imports.

    Parameters:
     filename: filename to import from
     linked: whether to link to file
     encoding: encoding for file
     prefix: prefix for output dataset names
     suffix: suffix for output dataset names
     tags: list of tags to apply to output datasets
     renames: dict map of names to renamed datasets
    """

    defaults = {
        'filename': None,
        'linked': False,
        'encoding': 'utf_8',
        'prefix': '',
        'suffix': '',
        'tags': None,
        'renames': None,
        }

    def __init__(self, **argsv):
        """Initialise the reader to import data from filename.
        """

        #  set defaults
        for k, v in citems(self.defaults):
            setattr(self, k, v)

        # set parameters
        for k, v in citems(argsv):
            if k not in self.defaults:
                raise ValueError("Invalid parameter %s" % k)
            setattr(self, k, v)

        # extra parameters to copy besides defaults
        self._extras = []

    def copy(self):
        """Make a copy of the parameters object."""

        newp = {}
        for k in list(self.defaults.keys()) + self._extras:
            newp[k] = getattr(self, k)
        return self.__class__(**newp)

class LinkedFileBase(object):
    """A base class for linked files containing common routines."""

    def __init__(self, params):
        """Save parameters."""
        self.params = params

    def createOperation(self):
        """Return operation to recreate self."""
        return None

    @property
    def filename(self):
        """Get filename."""
        return self.params.filename

    def _saveHelper(self, fileobj, cmd, fixedparams,
                    renameparams={}, relpath=None, extraargs={}):
        """Helper to write command to reload data.

        fileobj: file object to write to
        cmd: name of command to write
        fixedparams: list of parameters to list at start of command lines
        renameparams: optional map of params to command line params
        relpath: relative path for writing filename
        extraargs: other options to add to command line
        """

        p = self.params
        args = []

        # arguments without names at command start
        for par in fixedparams:
            if par == 'filename':
                v = self._getSaveFilename(relpath)
            else:
                v = getattr(p, par)
            args.append(utils.rrepr(v))

        # parameters key, values to put in command line
        plist = sorted( [(par, getattr(p, par)) for par in p.defaults] +
                        list(citems(extraargs)) )

        for par, val in plist:
            if ( val and
                 (par not in p.defaults or p.defaults[par] != val) and
                 par not in fixedparams and
                 par != 'tags' ):

                if par in renameparams:
                    par = renameparams[par]
                args.append('%s=%s' % (par, utils.rrepr(val)))

        # write command using comma-separated list
        fileobj.write('%s(%s)\n' % (cmd, ', '.join(args)))

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the document file."""
        pass

    def _getSaveFilename(self, relpath):
        """Get filename to write to save file.
        If relpath is a string, write relative to path given
        """
        if relpath:
            f = utils.relpath(self.params.filename, relpath)
        else:
            f = self.filename
        # Here we convert backslashes in Windows to forward slashes
        # This is compatible, but also works on Unix/Mac
        if sys.platform == 'win32':
            f = f.replace('\\', '/')
        return f

    def _deleteLinkedDatasets(self, document):
        """Delete linked datasets from document linking to self.
        Returns tags for deleted datasets.
        """

        tags = {}
        for name, ds in list(document.data.items()):
            if ds.linked == self:
                tags[name] = document.data[name].tags
                document.deleteData(name)
        return tags

    def _moveReadDatasets(self, tempdoc, document, tags):
        """Move datasets from tempdoc to document if they do not exist
        in the destination.

        tags is a dict of tags for each dataset
        """

        read = []
        for name, ds in list(tempdoc.data.items()):
            if name not in document.data:
                ds.linked = self
                if name in tags:
                    ds.tags = tags[name]
                document.setData(name, ds)
                read.append(name)
        return read

    def reloadLinks(self, document):
        """Reload links using an operation"""

        # get the operation for reloading
        op = self.createOperation()(self.params)

        # load data into a temporary document
        tempdoc = document.__class__()

        try:
            tempdoc.applyOperation(op)
        except Exception as ex:
            # if something breaks, record an error and return nothing
            document.log(cstr(ex))

            # find datasets which are linked using this link object
            # return errors for them
            errors = dict([(name, 1) for name, ds in citems(document.data)
                           if ds.linked is self])
            return ([], errors)

        # delete datasets which are linked and imported here
        tags = self._deleteLinkedDatasets(document)
        # move datasets into document
        read = self._moveReadDatasets(tempdoc, document, tags)

        # return errors (if any)
        errors = op.outinvalids

        return (read, errors)

class OperationDataImportBase(object):
    """Default useful import class."""

    def __init__(self, params):
        self.params = params

    def doImport(self, document):
        """Do import, override this.
        Set outdatasets
        """

    def addCustoms(self, document, customs):
        """Optionally, add the customs return by plugins to document."""

        type_attrs = {
            'import': 'def_imports',
            'color': 'def_colors',
            'colormap': 'def_colormaps',
            'constant': 'def_definitions',
            'function': 'def_definitions',
            'definition': 'def_definitions',
            }

        if len(customs) > 0:
            doceval = document.evaluate
            self.oldcustoms = [
                copy.deepcopy(doceval.def_imports),
                copy.deepcopy(doceval.def_definitions),
                copy.deepcopy(doceval.def_colors),
                copy.deepcopy(doceval.def_colormaps)]

            # FIXME: inefficient for large number of definitions
            for item in customs:
                ctype, name, val = item
                clist = getattr(doceval, type_attrs[ctype])
                for idx, (cname, cval) in enumerate(clist):
                    if cname == name:
                        clist[idx][1] = val
                        break
                else:
                    clist.append([name, val])

            doceval.update()

    def do(self, document):
        """Do import."""

        # list of returned dataset names
        self.outnames = []
        # map of names to datasets
        self.outdatasets = {}
        # list of returned custom variables
        self.outcustoms = []
        # invalid conversions
        self.outinvalids = {}

        # remember datasets in document for undo
        self.oldcustoms = None

        # do actual import
        retn = self.doImport()

        # these are custom values returned from the plugin
        if self.outcustoms:
            self.addCustoms(document, self.outcustoms)

        # handle tagging/renaming
        for name, ds in list(citems(self.outdatasets)):
            if self.params.tags:
                ds.tags.update(self.params.tags)
            if self.params.renames and name in self.params.renames:
                del self.outdatasets[name]
                self.outdatasets[self.params.renames[name]] = ds

        # only remember the parts we need
        self.olddatasets = [ (n, document.data.get(n))
                             for n in self.outdatasets ]

        self.olddatasets = []
        for name, ds in citems(self.outdatasets):
            self.olddatasets.append( (name, document.data.get(name)) )
            document.setData(name, ds)

        self.outnames = sorted(self.outdatasets)

        return retn

    def undo(self, document):
        """Undo import."""

        # put back old datasets
        for name, ds in self.olddatasets:
            if ds is None:
                document.deleteData(name)
            else:
                document.setData(name, ds)

        # for custom definitions
        if self.oldcustoms is not None:
            doceval = document.evaluate
            doceval.def_imports = self.oldcustoms[0]
            doceval.def_definitions = self.oldcustoms[1]
            doceval.def_colors = self.oldcustoms[2]
            doceval.def_colormaps = self.oldcustoms[3]
            doceval.update()
