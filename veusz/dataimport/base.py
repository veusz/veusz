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

        args = []

        # arguments without names at command start
        for par in fixedparams:
            if par == 'filename':
                args.append( utils.rrepr(self._getSaveFilename(relpath)) )
            else:
                args.append( utils.rrepr(getattr(self.params, par)) )

        # parameters key, values to put in command line
        plist = sorted( [(p, getattr(self.params, p))
                         for p in self.params.defaults] +
                        list(citems(extraargs)) )

        for par, val in plist:
            if ( val is not None and
                 self.params.defaults[par] != val and
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
        """Delete linked datasets from document linking to self."""

        for name, ds in list(document.data.items()):
            if ds.linked == self:
                document.deleteData(name)

    def _moveReadDatasets(self, tempdoc, document):
        """Move datasets from tempdoc to document if they do not exist
        in the destination."""

        read = []
        for name, ds in list(tempdoc.data.items()):
            if name not in document.data:
                read.append(name)

                # rename any renamed datasets
                outname = name
                print(name, self.params.renames)
                if self.params.renames and name in self.params.renames:
                    outname = self.params.renames[name]

                document.setData(outname, ds)
                ds.document = document
                ds.linked = self
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
        self._deleteLinkedDatasets(document)
        # move datasets into document
        read = self._moveReadDatasets(tempdoc, document)

        # return errors (if any)
        errors = op.outinvalids

        return (read, errors)

class OperationDataImportBase(object):
    """Default useful import class."""

    def __init__(self, params):
        self.params = params

        # list of returned datasets
        self.outdatasets = []
        # map of names to datasets
        self.outdatasetsmap = {}
        # list of returned custom variables
        self.outcustoms = []
        # invalid conversions
        self.outinvalids = {}

    def doImport(self, document):
        """Do import, override this.
        Set outdatasetsmap
        """

    def addCustoms(self, document, consts):
        """Optionally, add the customs return by plugins to document."""

        if len(consts) > 0:
            self.oldconst = list(document.customs)
            cd = document.customDict()
            for item in consts:
                if item[1] in cd:
                    idx, ctype, val = cd[item[1]]
                    document.customs[idx] = item
                else:
                    document.customs.append(item)
            document.updateEvalContext()

    def do(self, document):
        """Do import."""

        # remember datasets in document for undo
        self.oldconst = None

        # do actual import
        retn = self.doImport()

        # handle tagging/renaming
        for name, ds in list(citems(self.outdatasetsmap)):
            if self.params.tags:
                ds.tags.update(self.params.tags)
            if self.params.renames and name in self.params.renames:
                del self.outdatasetsmap[name]
                self.outdatasetsmap[self.params.renames[name]] = ds

        # only remember the parts we need
        self.olddatasets = [ (n, document.data.get(n))
                             for n in self.outdatasetsmap ]

        self.olddatasets = []
        for name, ds in citems(self.outdatasetsmap):
            self.olddatasets.append( (name, document.data.get(name)) )
            ds.document = document
            document.data[name] = ds

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
        if self.oldconst is not None:
            document.customs = self.oldconst
            document.updateEvalContext()
