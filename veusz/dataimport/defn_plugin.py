#    Copyright (C) 2013 Jeremy S. Sanders
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

from __future__ import division, print_function
import traceback

from ..compat import citems, cstr, crepr
from .. import document
from .. import plugins
from .. import qtall as qt4
from . import base

def _(text, disambiguation=None, context="Import_Plugin"):
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class ImportParamsPlugin(base.ImportParamsBase):
    """Parameters for import plugins.

    Additional parameter:
     plugin: name of plugin

    Plugins have their own parameters."""

    defaults = {
        'plugin': None,
        }
    defaults.update(base.ImportParamsBase.defaults)

    def __init__(self, **argsv):
        """Initialise plugin parameters, splitting up default parameters
        and plugin parameters."""

        pluginpars = {}
        upvars = {}
        for n, v in citems(argsv):
            if n in self.defaults:
                upvars[n] = v
            else:
                pluginpars[n] = v

        base.ImportParamsBase.__init__(self, **upvars)
        self.pluginpars = pluginpars
        self._extras.append('pluginpars')

class LinkedFilePlugin(base.LinkedFileBase):
    """Represent a file linked using an import plugin."""

    def createOperation(self):
        """Return operation to recreate self."""
        return OperationDataImportPlugin

    def saveToFile(self, fileobj, relpath=None):
        """Save the link to the vsz document file."""

        p = self.params
        params = [crepr(p.plugin),
                  crepr(self._getSaveFilename(relpath)),
                  "linked=True"]
        if p.encoding != "utf_8":
            params.append("encoding=" + crepr(p.encoding))
        if p.prefix:
            params.append("prefix=" + crepr(p.prefix))
        if p.suffix:
            params.append("suffix=" + crepr(p.suffix))
        for name, val in citems(p.pluginpars):
            params.append("%s=%s" % (name, crepr(val)))

        fileobj.write("ImportFilePlugin(%s)\n" % (", ".join(params)))

class OperationDataImportPlugin(base.OperationDataImportBase):
    """Import data using a plugin."""

    descr = _('import using plugin')

    def doImport(self, doc):
        """Do import."""

        pluginnames = [p.name for p in plugins.importpluginregistry]
        plugin = plugins.importpluginregistry[
            pluginnames.index(self.params.plugin)]

        # if the plugin is a class, make an instance
        # the old API is for the plugin to be instances
        if isinstance(plugin, type):
            plugin = plugin()

        # strip out parameters for plugin itself
        p = self.params

        # stick back together the plugin parameter object
        plugparams = plugins.ImportPluginParams(
            p.filename, p.encoding,  p.pluginpars)
        results = plugin.doImport(plugparams)

        # make link for file
        LF = None
        if p.linked:
            LF = LinkedFilePlugin(p)

        customs = []

        # convert results to real datasets
        names = []
        for d in results:
            if isinstance(d, plugins.Dataset1D):
                ds = document.Dataset(data=d.data, serr=d.serr, perr=d.perr,
                                      nerr=d.nerr)
            elif isinstance(d, plugins.Dataset2D):
                ds = document.Dataset2D(data=d.data,
                                        xrange=d.rangex, yrange=d.rangey,
                                        xgrid=d.xgrid, ygrid=d.ygrid)
            elif isinstance(d, plugins.DatasetText):
                ds = document.DatasetText(data=d.data)
            elif isinstance(d, plugins.DatasetDateTime):
                ds = document.DatasetDateTime(data=d.data)
            elif isinstance(d, plugins.Constant):
                customs.append( ['constant', d.name, d.val] )
                continue
            elif isinstance(d, plugins.Function):
                customs.append( ['function', d.name, d.val] )
                continue
            else:
                raise RuntimeError("Invalid data set in plugin results")

            # set any linking
            if p.linked:
                ds.linked = LF

            # construct name
            name = p.prefix + d.name + p.suffix

            # actually make dataset
            doc.setData(name, ds)

            names.append(name)

        # add constants, functions to doc, if any
        self.addCustoms(doc, customs)

        self.outdatasets = names
        self.outcustoms = list(customs)

def ImportFilePlugin(comm, plugin, filename, **args):
    """Import file using a plugin.

    optional arguments:
    prefix: add to start of dataset name (default '')
    suffix: add to end of dataset name (default '')
    linked: link import to file (default False)
    encoding: file encoding (may not be used, default 'utf_8')
    plus arguments to plugin

    returns: list of imported datasets, list of imported customs
    """

    realfilename = comm.findFileOnImportPath(filename)
    params = ImportParamsPlugin(
        plugin=plugin, filename=realfilename, **args)

    op = OperationDataImportPlugin(params)
    try:
        comm.document.applyOperation(op)
    except:
        comm.document.log("Error in plugin %s" % plugin)
        exc =  ''.join(traceback.format_exc())
        comm.document.log(exc)
    return op.outdatasets, op.outcustoms

document.registerImportCommand('ImportFilePlugin', ImportFilePlugin)
