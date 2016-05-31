# -*- coding: utf-8 -*-
#    Copyright (C) 2016 Jeremy S. Sanders
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
###############################################################################

"""Handle piping data."""

from .. import qtall as qt4
from ..compat import cstr, citems
from .. import plugins
from . import datasets

def _(text, disambiguation=None, context='pipe'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

def convertToPluginDataset(ds):
    """Convert Veusz standard datasets to datasets to pass to a plugin."""

    name = 'xxx'

    if ds is None:
        return None

    if ds.dimensions != 1:
        raise datasets.DatasetExpception(
            'Only 1-dimensional datasets supported')

    if ds.datatype == 'numeric':
        return plugins.Dataset1D(
            name, data=ds.data,
            serr=ds.serr, perr=ds.perr, nerr=ds.nerr)
    elif ds.datatype == 'text':
        return plugin.DatasetText(
            name, data=ds.data)

def convertFromPluginDataset(ds):
    """Convert to Veusz dataset from plugin dataset."""

    if ds is None:
        return None

    if isinstance(ds, plugins.Dataset1D):
        return datasets.Dataset(
            ds.data, serr=ds.serr, perr=ds.perr, nerr=ds.nerr)
    elif isinstance(ds, plugins.DatasetText):
        return datasets.DatasetText(data=ds.data)
    else:
        raise RuntimeError("Unknown plugin dataset type")

class Pipe:
    def __init__(self, document):
        self.document = document
        self.datasets = []

        # link the function to the input datasets
        self.pipeenv = {}
        for name, info in citems(plugins.pipepluginregistry):
            fn = info[0]
            self.pipeenv[name] = fn(self.datasets)
        self.pipeenv['_DS_'] = self._evalDataset

    def _evalDataset(self, name, part):
        try:
            ds = document.data[name]
        except KeyError:
            raise datasets.DatasetExpception(
                'Dataset %s does not exist' % name)
        return convertToPluginDataset(ds)

    def evalExpr(self, expr, dsx, dsy, dslabel, dscolor, dssize):
        """Execute pipe

        expr: expression
        dsx,dsy,dslabel,dscolor,dssize: input datasets or None

        returns: tuple of output datasets.
        """

        # substitute in names of datasets as calling _DS_ function
        subsexpr = datasets.substituteDatasets(
            self.document.data, expr, None)[0]

        comp = self.document.compileCheckedExpression(
            subsexpr, origexpr=expr)
        if comp is None:
            return

        # these are the datasets passed to the pipe functions,
        # converted to the simplified plugin format
        self.datasets[:] = [
            convertToPluginDataset(dsx),
            convertToPluginDataset(dsy),
            convertToPluginDataset(dslabel),
            convertToPluginDataset(dscolor),
            convertToPluginDataset(dssize),
        ]

        env = dict(self.document.eval_context)
        env.update(self.pipeenv)

        # run the plugin
        try:
            evalout = eval(comp, env)
        except Exception as ex:
            self.document.log(
                _("Error evaluating '%s': '%s'" % (expr, cstr(ex))))
            return

        # get the results
        out = [convertFromPluginDataset(d) for d in self.datasets]

        del self.datasets[:]

        return tuple(out)
