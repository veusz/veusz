# commandinterface.py
# this module supplies the command line interface for plotting

#    Copyright (C) 2004 Jeremy S. Sanders
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

"""
Module supplies the command interface used in the program, and for
external programs.
"""

from __future__ import division, print_function
import os.path
import numpy as N

from ..compat import cbasestr
from .. import qtall as qt4
from .. import setting
from .. import embed
from .. import plugins
from .. import utils
from .. import datasets

from . import operations
from . import mime
from . import export

def _(text, disambiguation=None, context='CommandInterface'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

def registerImportCommand(name, method, filenamearg=0):
    """Add command to command interface."""
    setattr(CommandInterface, name, method)
    CommandInterface.import_commands.append(name)
    CommandInterface.import_filenamearg[name] = filenamearg

class CommandInterface(qt4.QObject):
    """Class provides command interface."""

    # commands which are safe in any script (excluding import commands)
    safe_commands = [
        'Action',
        'Add',
        'AddCustom',
        'AddImportPath',
        'CloneWidget',
        'CreateHistogram',
        'DatasetPlugin',
        'FilterDatasets',
        'Get',
        'GetChildren',
        'GetColormap',
        'GetData',
        'GetDataType',
        'GetDatasets',
        'ImportFITSFile',
        'List',
        'NodeChildren',
        'NodeType',
        'ReloadData',
        'Remove',
        'RemoveCustom',
        'Rename',
        'ResolveReference',
        'Set',
        'SetData',
        'SetData2D',
        'SetData2DExpression',
        'SetData2DExpressionXYZ',
        'SetData2DXYFunc',
        'SetDataDateTime',
        'SetDataExpression',
        'SetDataND',
        'SetDataRange',
        'SetDataText',
        'SetToReference',
        'SetVerbose',
        'SettingType',
        'TagDatasets',
        'To',
        'WidgetType',
        ]

    # commands for importing data
    import_commands = []
    # number of argument which contains filename
    import_filenamearg = {}

    # commands which can modify disk, etc
    unsafe_commands = [
        'Export',
        'Print',
        'Save',
        ]

    def __init__(self, document):
        """Initialise the interface."""
        qt4.QObject.__init__(self)

        self.document = document
        self.currentwidget = self.document.basewidget
        self.verbose = False
        self.importpath = []

        self.document.sigWiped.connect(self.slotWipedDoc)

        self.Root = embed.WidgetNode(self, 'widget', '/')

    @qt4.pyqtSlot()
    def slotWipedDoc(self):
        """When the document is wiped, we change to the root widget."""
        self.To('/')

    def findFileOnImportPath(self, filename):
        """Find file on path, returning filename, or original if not found."""
        for path in self.importpath:
            fname = os.path.join(path, filename)
            try:
                # try to open file to make sure we have access to it and
                # it exists
                opened = open(fname)
                opened.close()
                return fname
            except EnvironmentError:
                pass
        return filename

    def SetVerbose(self, v=True):
        """Specify whether we want verbose output after operations."""
        self.verbose = v

    def Add(self, widgettype, **args_opt):
        """Add a widget to the widget with the type given.

        optional arguments:
          widget: widget path to place widget in
          autoadd: if True (default), any subwidgets associated with
            widget are created automatically
          setting names, e.g. leftMargin='1cm', Border_color='red'
        """

        at = self.currentwidget
        if 'widget' in args_opt:
            at = self.document.resolveWidgetPath(
                self.currentwidget, args_opt['widget'])
            del args_opt['widget']

        op = operations.OperationWidgetAdd(at, widgettype, **args_opt)
        w = self.document.applyOperation(op)

        if self.verbose:
            print(_("Added a widget of type '%s' (%s)") % (type, w.userdescription))

        return w.name

    def AddCustom(self, ctype, name, val, mode='appendalways'):
        """Add a custom definition for evaluation of expressions.
	This can define a constant (can be in terms of other
	constants), a function of 1 or more variables, or a function
	imported from an external python module.

        ctype is "constant", "function", "definition" (either
        constant or function), "import", "color" or "colormap".

        name is name of constant, color or colormap, "function(x, y,
        ...)"  or module name.

        val is definition for constant or function (both are
        _strings_), or is a list of symbols for a module (comma
        separated items in a string). For a colormap, val is a list of
        4-item tuples containing R,G,B,alpha values from 0 to 255. For
        a color this is a string with the format '#RRGGBB' or
        '#RRGGBBAA'.

        if mode is 'appendalways', the custom value is appended to the
        end of the list even if there is one with the same name. If
        mode is 'replace', it replaces any existing definition in the
        same place in the list or is appended otherwise. If mode is
        'append', then an existing definition is deleted, and the new
        one appended to the end.

        """

        if ctype == 'colormap':
            self.document.evaluate.validateProcessColormap(val)
        else:
            if not isinstance(val, cbasestr):
                raise RuntimeError('Value should be string')

        if mode not in ('appendalways', 'append', 'replace'):
            raise RuntimeError('Invalid mode')

        try:
            attr = operations.OperationSetCustom.type_to_attr[ctype]
        except KeyError:
            raise RuntimeError('Invalid type')

        vals = list(getattr(self.document.evaluate, attr))

        item = [name.strip(), val]
        if mode == 'appendalways':
            vals.append(item)
        else:
            # find any existing item
            for i, (n, v) in enumerate(vals):
                if n == name:
                    if mode == 'append':
                        del vals[i]
                        vals.append(item)
                    else: # replace
                        vals[i] = item
                    break
            else:
                # no existing item, so append
                vals.append(item)

        op = operations.OperationSetCustom(ctype, vals)
        self.document.applyOperation(op)

    def AddImportPath(self, directory):
        """Add directory to import file path."""
        assert isinstance(directory, cbasestr)
        self.importpath.append(directory)

    def CloneWidget(self, widget, newparent, newname=None):
        """Clone the widget given, placing the copy in newparent and
        the name given.

        newname is an optional new name to give it

        Returns new widget path
        """

        widget = self.document.resolveWidgetPath(self.currentwidget, widget)
        newparent = self.document.resolveWidgetPath(self.currentwidget, newparent)
        op = mime.OperationWidgetClone(widget, newparent, newname)
        w = self.document.applyOperation(op)
        return w.path

    def CreateHistogram(self, inexpr, outbinsds, outvalsds, binparams=None,
                        binmanual=None, method='counts',
                        cumulative = 'none', errors=False):
        """Histogram an input expression.

        inexpr is input expression
        outbinds is the name of the dataset to create giving bin positions
        outvalsds is name of dataset for bin values
        binparams is None or (numbins, minval, maxval, islogbins)
        binmanual is None or a list of bin values
        method is 'counts', 'density', or 'fractions'
        cumulative is to calculate cumulative distributions which is
          'none', 'smalltolarge' or 'largetosmall'
        errors is to calculate Poisson error bars
        """
        op = operations.OperationDatasetHistogram(
            inexpr, outbinsds, outvalsds, binparams=binparams,
            binmanual=binmanual, method=method,
            cumulative=cumulative, errors=errors)
        self.document.applyOperation(op)

        if self.verbose:
            print(_('Constructed histogram of "%s", creating datasets'
                    ' "%s" and "%s"') % (inexpr, outbinsds, outvalsds))

    def DatasetPlugin(self, pluginname, fields, datasetnames={}):
        """Use a dataset plugin.

        pluginname: name of plugin to use
        fields: dict of input values to plugin
        datasetnames: dict mapping old names to new names of datasets
        if they are renamed. The new name None means dataset is deleted."""

        # lookup plugin (urgh)
        plugin = None
        for pkls in plugins.datasetpluginregistry:
            if pkls.name == pluginname:
                plugin = pkls()
                break
        if plugin is None:
            raise RuntimeError("Cannot find dataset plugin '%s'" % pluginname)

        # do the work
        op = operations.OperationDatasetPlugin(plugin, fields,
                                               datasetnames=datasetnames)
        outdatasets = self.document.applyOperation(op)

        if self.verbose:
            print(_("Used dataset plugin %s to make datasets %s") % (
                pluginname, ', '.join(outdatasets)))

    def Remove(self, name):
        """Remove a widget from the dataset."""
        w = self.document.resolveWidgetPath(self.currentwidget, name)
        op = operations.OperationWidgetDelete(w)
        self.document.applyOperation(op)
        if self.verbose:
            print(_("Removed widget '%s'") % name)

    def RemoveCustom(self, name):
        """Removes a custom-defined constant, function or import."""

        # look for definiton and delete if found
        for ctype, attr in (
                ('import', 'def_imports'),
                ('definition', 'def_definitions'),
                ('color', 'def_colors'),
                ('colormap', 'def_colormaps')):
            vals = list(getattr(self.document.evaluate, attr))
            for i, (cname, cval) in enumerate(vals):
                if name == cname:
                    del vals[i]
                    op = operations.OperationSetCustom(ctype, vals)
                    self.document.applyOperation(op)
                    return
        else:
            raise ValueError('Custom variable not defined')

    def To(self, where):
        """Change to a widget within the current widget.

        where is a path to the widget relative to the current widget
        """

        self.currentwidget = self.document.resolveWidgetPath(
            self.currentwidget,
            where)

        if self.verbose:
            print(_("Changed to widget '%s'") % self.currentwidget.path)

    def List(self, where='.'):
        """List the contents of a widget, by default the current widget."""

        widget = self.document.resolveWidgetPath(self.currentwidget, where)
        children = widget.childnames

        if len(children) == 0:
            print('%30s' % _('No children found'))
        else:
            # output format name, type
            for name in children:
                w = widget.getChild(name)
                print('%10s %10s %30s' % (name, w.typename, w.userdescription))

    def Get(self, var):
        """Get the value of a setting."""
        return self.document.resolveSettingPath(self.currentwidget, var).val

    def GetChildren(self, where='.'):
        """Return a list of widgets which are children of the widget of the
        path given."""
        return list(
            self.document.resolveWidgetPath(self.currentwidget, where).childnames )

    def GetColormap(self, name, invert=False, nvals=256):
        """Return an array of [red,green,blue,alpha] values
        representing the colormap with the name given.

        Each return value is between 0 and 255.

        The number of values to return is given by nvals
        """
        cmap = self.document.evaluate.getColormap(name, invert)
        return utils.getColormapArray(cmap, nvals)

    def GetDatasets(self):
        """Return a list of names of datasets."""
        return sorted(self.document.data)

    def ResolveReference(self, setn):
        """If the setting is set to a reference, follow the chain of
        references to return the absolute path to the real setting.

        If it is not a reference return None.
        """

        setn = self.document.resolveSettingPath(self.currentwidget, setn)
        if setn.isReference():
            real = setn.getReference().resolve(setn)
            return real.path
        else:
            return None

    def Save(self, filename, mode='vsz'):
        """Save the state to a file.

        mode can be:
         'vsz': standard veusz text format
         'hdf5': HDF5 format
        """
        self.document.save(filename, mode)

    def Set(self, setting_path, val):
        """Set the value of a setting."""
        setn = self.document.resolveSettingPath(self.currentwidget, setting_path)
        op = operations.OperationSettingSet(setn, val)
        self.document.applyOperation(op)

        if self.verbose:
            print( _("Set setting '%s' to %s") % (setting_path, repr(setn.get())) )

    def SetToReference(self, setting_path, val):
        """Set setting to a reference value."""
        setn = self.document.resolveSettingPath(self.currentwidget, setting_path)
        op = operations.OperationSettingSet(setn, setting.Reference(val))
        self.document.applyOperation(op)

        if self.verbose:
            print( _( "Set setting '%s' to %s") % (setting_path, repr(setn.get())) )

    def SetData(self, name, val, symerr=None, negerr=None, poserr=None):
        """Create/set dataset name with values (and optionally errors)."""

        data = datasets.Dataset(val, symerr, negerr, poserr)
        op = operations.OperationDatasetSet(name, data)
        self.document.applyOperation(op)

        if self.verbose:
            print(
                _("Set dataset '%s':\n"
                  " Values = %s\n"
                  " Symmetric errors = %s\n"
                  " Negative errors = %s\n"
                  " Positive errors = %s") % (
                      name, str(data.data), str(data.serr),
                      str(data.nerr), str(data.perr))
            )

    def SetDataDateTime(self, name, vals):
        """Set datetime dataset to be values given.
        vals is a list of python datetime objects
        """
        v = [utils.datetimeToFloat(x) for x in vals]
        ds = datasets.DatasetDateTime(v)
        op = operations.OperationDatasetSet(name, ds)
        self.document.applyOperation(op)

        if self.verbose:
            print(
                _("Set dataset '%s':\n"
                  " Values = %s") % (
                      name, str(ds.data))
            )

    def SetDataExpression(self, name, val, symerr=None, negerr=None, poserr=None,
                          linked=False, parametric=None):
        """Create a dataset based on text expressions.

        Expressions are functions of existing datasets.
        If evaluating the expression 'y*10' in negerr, then the negerrs of dataset y
        are used, and so on.
        To access a specific part of the dataset y, the suffixes _data, _serr, _perr,
        and _nerr can be appended.

        If linked is True then the expressions are reevaluated if the document
        is modified

        parametric: tuple of (minval, maxval, numitems) for creating parametric
                    datasets. t set to this range when evaluating expressions.

        """

        expr = {'data': val, 'serr': symerr, 'nerr': negerr, 'perr': poserr}
        op = operations.OperationDatasetCreateExpression(name, expr, linked,
                                                         parametric=parametric)

        data = self.document.applyOperation(op)

        if self.verbose:
            print(
                _("Set dataset '%s' based on expression:\n"
                  " Values = %s\n"
                  " Symmetric errors = %s\n"
                  " Negative errors = %s\n"
                  " Positive errors = %s") % (
                      name, str(data.data), str(data.serr),
                      str(data.nerr), str(data.perr))
            )
            if parametric:
                print(_(" Where t goes form %g:%g in %i steps") % parametric)
            print(_(" linked to expression = %s") % repr(linked))

    def SetDataND(self, name, val):
        """Set n-dimensional dataset name with values."""

        data = datasets.DatasetND(val)
        op = operations.OperationDatasetSet(name, data)
        self.document.applyOperation(op)

        if self.verbose:
            print(
                _("Set dataset (nD) '%s':\n"
                  " Values = %s\n") % (
                      name, str(data.data))
            )

    def SetDataRange(self, name, numsteps, val, symerr=None, negerr=None,
                     poserr=None, linked=False):
        """Create dataset based on ranges of values, e.g. 1 to 10 in 10 steps

        name: name of dataset
        numsteps: number of steps to create
        val: range in form of tuple (minval, maxval)
        symerr, negerr & poserr: ranges for errors (optional)
        """

        parts = {'data': val, 'serr': symerr, 'nerr': negerr, 'perr': poserr}
        op = operations.OperationDatasetCreateRange(name, numsteps, parts,
                                                    linked)
        self.document.applyOperation(op)

        if self.verbose:
            print(
                _("Set dataset '%s' based on range:\n"
                  " Number of steps = %i\n"
                  " Range of data = %s\n"
                  " Range of symmetric error = %s\n"
                  " Range of positive error = %s\n"
                  " Range of negative error = %s") % (
                      name, numsteps, repr(val),
                      repr(symerr), repr(poserr), repr(negerr))
              )

    def SetData2DExpression(self, name, expr, linked=False):
        """Create a 2D dataset based on expressions

        name is the new dataset name
        expr is an expression which should return a 2D array
        linked specifies whether to permanently link the dataset to the expressions
        """

        op = operations.OperationDataset2DCreateExpression(name, expr, linked)
        data = self.document.applyOperation(op)

        if self.verbose:
            print(
                _("Set 2D dataset '%s' based on expressions\n"
                  " expression = %s\n"
                  " linked to expression = %s\n"
                  " Made a dataset (%i x %i)") % (
                      name, repr(expr), repr(linked),
                      data.data.shape[0], data.data.shape[1])
            )

    def SetData2DExpressionXYZ(self, name, xexpr, yexpr, zexpr, linked=False):
        """Create a 2D dataset based on expressions in x, y and z

        xexpr is an expression which expands to an equally-spaced grid of x coordinates
        yexpr expands to equally spaced y coordinates
        zexpr expands to z coordinates.
        linked specifies whether to permanently link the dataset to the expressions
        """

        op = operations.OperationDataset2DCreateExpressionXYZ(name, xexpr, yexpr, zexpr,
                                                              linked)
        data = self.document.applyOperation(op)

        if self.verbose:
            print(
                _("Made 2D dataset '%s' based on expressions:\n"
                  " X expression = %s\n"
                  " Y expression = %s\n"
                  " Z expression = %s\n"
                  " is linked to expression = %s\n"
                  " Shape (%i x %i)") % (
                      name,
                      repr(xexpr), repr(yexpr), repr(zexpr),
                      repr(linked),
                      data.data.shape[0], data.data.shape[1])
            )

    def SetData2DXYFunc(self, name, xstep, ystep, expr, linked=False):
        """Create a 2D dataset based on expressions of a range of x and y

        xstep is a tuple(min, max, step)
        ystep is a tuple(min, max, step)
        expr is an expression of x and y
        linked specifies whether to permanently link the dataset to the expressions
        """

        op = operations.OperationDataset2DXYFunc(name, xstep, ystep,
                                                 expr, linked)
        data = self.document.applyOperation(op)

        if self.verbose:
            print(
                _("Set 2D dataset '%s' based on function of x and y\n"
                  " X steps = %s\n"
                  " Y steps = %s\n"
                  " Expression = %s\n"
                  " linked to expression = %s\n"
                  " Made a dataset (%i x %i)") % (
                      name, repr(xstep), repr(ystep),
                      repr(expr), repr(linked),
                      data.data.shape[0], data.data.shape[1])
            )

    def SetData2D(self, name, data, xrange=None, yrange=None,
                  xedge=None, yedge=None,
                  xcent=None, ycent=None):
        """Create a 2D dataset.

        name: name of dataset
        data: 2d array
        xrange: optional tuple with X range of data (min, max)
        yrange: optional tuple with Y range of data (min, max)
        xedge: x values for grid (instead of rangex)
        yedge: y values for grid (instead of rangey)
        xcent: x values for pixel centres (instead of rangex)
        ycent: y values for pixel centres (instead of rangey)
        """

        data = N.array(data)

        if ( (xedge is not None and not utils.checkAscending(xedge)) or
             (yedge is not None and not utils.checkAscending(yedge)) ):
            raise ValueError("xedge and yedge must be ascending, if given")
        if ( (xcent is not None and not utils.checkAscending(xcent)) or
             (ycent is not None and not utils.checkAscending(ycent)) ):
            raise ValueError("xcent and ycent must be ascending, if given")

        if ( (xedge is not None and len(xedge) != data.shape[1]+1) or
             (yedge is not None and len(yedge) != data.shape[0]+1) ):
            raise ValueError("xedge and yedge lengths must be data shape+1")
        if ( (xcent is not None and len(xcent) != data.shape[1]) or
             (ycent is not None and len(ycent) != data.shape[0]) ):
            raise ValueError("xcent and ycent lengths must be data shape")

        data = datasets.Dataset2D(data, xrange=xrange, yrange=yrange,
                                  xedge=xedge, yedge=yedge,
                                  xcent=xcent, ycent=ycent)
        op = operations.OperationDatasetSet(name, data)
        self.document.applyOperation(op)

        if self.verbose:
            print(_("Set 2d dataset '%s'") % name)

    def SetDataText(self, name, val):
        """Create a text dataset."""

        data = datasets.DatasetText(val)
        op = operations.OperationDatasetSet(name, data)
        self.document.applyOperation(op)

        if self.verbose:
            print(
                _("Set text dataset '%s'\n"
                  "Values = %s") % (
                      name, repr(data.data))
            )

    def GetData(self, name):
        """Return the data with the name.

        For a 1D dataset, returns a tuple (None if not defined)
            (data, serr, nerr, perr)
        For a 2D dataset, returns
            (data, xrange, yrange)
        For an nD dataset returns data array
        For a text dataset, return a list of text
        For a date dataset, return a list of python datetime objects

        Return copies, so that the original data can't be indirectly modified
        """

        d = self.document.getData(name)
        if d.displaytype == 'text':
            return d.data[:]
        elif d.displaytype == 'date':
            return [utils.floatToDateTime(x) for x in d.data]
        elif d.dimensions == 2:
            return (d.data.copy(), d.xrange, d.yrange)
        elif d.dimensions == -1:
            return d.data.copy()
        else:
            data = serr = nerr = perr = None
            if d.data is not None:
                data = d.data.copy()
            if d.serr is not None:
                serr = d.serr.copy()
            if d.nerr is not None:
                nerr = d.nerr.copy()
            if d.perr is not None:
                perr = d.perr.copy()

            return (data, serr, nerr, perr)

    def GetDataType(self, name):
        """Return the type of dataset. Returns None if no dataset.

        For a 1D dataset, returns '1d'
        For a 2D dataset, returns '2d'
        For a text dataset, returns 'text'
        For a datetime dataset, returns 'datetime'
        """

        try:
            d = self.document.getData(name)
        except KeyError:
            return None
        if d.displaytype == 'text':
            return 'text'
        elif d.displaytype == 'date':
            return 'datetime'
        elif d.dimensions == 2:
            return '2d'
        else:
            return '1d'

    def ReloadData(self):
        """Reload any linked datasets.

        Returned is a tuple (datasets, errors)
         where datasets is a list of datasets read
         errors is a dict of the datasets with the number of errors while
         converting the data
        """

        return self.document.reloadLinkedDatasets()

    def Action(self, action, widget='.'):
        """Performs action on current widget."""

        w = self.document.resolveWidgetPath(self.currentwidget, widget)

        # run action
        w.getAction(action).function()

    def Print(self):
        """Print document."""
        export.printDialog(None, self.document)

    def Export(self, filename, color=True, page=[0], dpi=100,
               antialias=True, quality=85, backcolor='#ffffff00',
               pdfdpi=150, svgtextastext=False):
        """Export plot to filename.

        color is True or False if color is requested in output file
        page is a list of page numbers to export
        dpi is the number of dots per inch for bitmap output files
        antialias antialiases output if True
        quality is a quality parameter for jpeg output
        backcolor is the background color for bitmap files, which is a name or
         a #RRGGBBAA value (red, green, blue, alpha)
        pdfdpi is the dpi to use when exporting eps or pdf files
        svgtextastext: write text in SVG as text, rather than curves
        """

        # compatibility where page was a single number
        try:
            pages = [p for p in page]
        except TypeError:
            pages = [page]

        e = export.Export(
            self.document, filename, pages, color=color,
            bitmapdpi=dpi, antialias=antialias,
            quality=quality, backcolor=backcolor,
            pdfdpi=pdfdpi, svgtextastext=svgtextastext)
        e.export()

    def Rename(self, widget, newname):
        """Rename the widget with the path given to the new name.

        eg Rename('graph1/xy1', 'scatter')
        This function does not move widgets."""

        w = self.document.resolveWidgetPath(self.currentwidget, widget)
        op = operations.OperationWidgetRename(w, newname)
        self.document.applyOperation(op)

    def NodeType(self, path):
        """This function treats the set of objects in the widget and
        setting tree as a set of nodes.

        Returns type of node given.
        Return values are: 'widget', 'settings' or 'setting'
        """
        item = self.document.resolvePath(self.currentwidget, path)

        if item.iswidget:
            return 'widget'
        elif item.issettings:
            return 'settinggroup'
        else:
            return 'setting'

    def NodeChildren(self, path, types='all'):
        """This function treats the set of objects in the widget and
        setting tree as a set of nodes.

        Returns a list of the names of the children of this node."""

        item = self.document.resolvePath(self.currentwidget, path)

        out = []
        if item.iswidget:
            if types == 'all' or types == 'widget':
                out += item.childnames
            if types == 'all' or types == 'settinggroup':
                out += [s.name for s in item.settings.getSettingsList()]
            if types == 'all' or types == 'setting':
                out += [s.name for s in item.settings.getSettingList()]
        elif item.issettings:
            if types == 'all' or types == 'settinggroup':
                out += [s.name for s in item.getSettingsList()]
            if types == 'all' or types == 'setting':
                out += [s.name for s in item.getSettingList()]
        return out

    def WidgetType(self, path):
        """Get the Veusz widget type for a widget with path given.

        Raises a ValueError if the path doesn't point to a widget."""

        item = self.document.resolvePath(self.currentwidget, path)
        if item.iswidget:
            return item.typename
        else:
            raise ValueError("Path '%s' is not a widget" % path)

    def SettingType(self, setting_path):
        """Get the type of setting (a string) for the path given.

        Raise a ValueError if path is not a setting
        """

        setn = self.document.resolveSettingPath(self.currentwidget, setting_path)
        return setn.typename

    def TagDatasets(self, tag, datasets):
        """Apply tag to list of datasets."""
        op = operations.OperationDataTag(tag, datasets)
        self.document.applyOperation(op)

        if self.verbose:
            print(_("Applied tag %s to datasets %s") % (
                tag, ' '.join(datasets)))

    def FilterDatasets(self, filterexpr, dataset_list,
                       prefix="", suffix="",
                       invert=False, replaceblanks=False):
        """Apply filter expression to list of datasets.

        filterexpr: input filter expression
        dataset_list: list of input dataset names
        prefix, suffix: output prefix/suffix to add to names (one must be set)
        invert: invert results of filter expression
        replaceblanks: replace filtered values with nan/blank in output.
        """
        op = operations.OperationDatasetsFilter(
            filterexpr, dataset_list,
            prefix=prefix, suffix=suffix,
            invert=invert, replaceblanks=replaceblanks)
        self.document.applyOperation(op)

        if self.verbose:
            print(
                _('Filtered datasets %s using expression %s. '
                  'Output prefix=%s, suffix=%s') % (
                      dataset_list, filterexpr, prefix, suffix)
            )
