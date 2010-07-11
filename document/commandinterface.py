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

# $Id$

"""
Module supplies the command interface used in the program, and for
external programs.
"""

import os.path

import veusz.qtall as qt4
import veusz.setting as setting
import veusz.embed as embed

import datasets
import operations
import dataset_histo

class CommandInterface(qt4.QObject):
    """Class provides command interface."""

    # commands which are safe in any script
    safe_commands = (
        'Action',
        'Add',
        'AddCustom',
        'AddImportPath',
        'CreateHistogram',
        'Get',
        'GetChildren',
        'GetData',
        'GetDatasets',
        'ImportFITSFile',
        'ImportFile',
        'ImportFile2D',
        'ImportFileCSV',
        'ImportFilePlugin',
        'ImportString',
        'ImportString2D',
        'List',
        'NodeChildren',
        'NodeType',
        'ReloadData',
        'Remove',
        'Rename',
        'ResolveReference',
        'Set',
        'SetToReference',
        'SetData',
        'SetData2D',
        'SetData2DExpressionXYZ',
        'SetData2DXYFunc',
        'SetDataExpression',
        'SetDataRange',
        'SetDataText',
        'SetVerbose',
        'To',
        'WidgetType',
        )

    # commands which can modify disk, etc
    unsafe_commands = (
        'Export',        
        'Print',
        'Save',
        )

    def __init__(self, document):
        """Initialise the interface."""
        qt4.QObject.__init__(self)

        self.document = document
        self.currentwidget = self.document.basewidget
        self.verbose = False
        self.importpath = []

        self.connect( self.document, qt4.SIGNAL("sigWiped"),
                      self.slotWipedDoc )

        self.Root = embed.WidgetNode(self, 'widget', '/')

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
            except IOError:
                pass
        return filename

    def SetVerbose(self, v=True):
        """Specify whether we want verbose output after operations."""
        self.verbose = v

    def Add(self, widgettype, *args, **args_opt):
        """Add a graph to the graph with the type given.

        optional argument:
          widget: widget path to place widget in

        The optional arguments are sent to construct the widget.
        """

        at = self.currentwidget
        if 'widget' in args_opt:
            at = self.document.resolve(self.currentwidget, args_opt['widget'])
            del args_opt['widget']

        op = operations.OperationWidgetAdd(at, widgettype, *args, **args_opt)
        w = self.document.applyOperation(op)

        if self.verbose:
            print "Added a graph of type '%s' (%s)" % (type, w.userdescription)

        return w.name

    def AddCustom(self, name, ctype, val):
        """Add a custom definition for evaluation of expressions.

        name is name of constant, or function(params)
        ctype is constant or function
        val is definition."""

        vals = list( self.document.customs )
        vals.append( [name, ctype, val] )
        op = operations.OperationSetCustom(vals)
        self.document.applyOperation(op)

    def AddImportPath(self, directory):
        """Add directory to import file path."""
        assert isinstance(directory, basestring)
        self.importpath.append(directory)

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
        op = dataset_histo.OperationDatasetHistogram(
            inexpr, outbinsds, outvalsds, binparams=binparams,
            binmanual=binmanual, method=method,
            cumulative=cumulative, errors=errors)
        self.document.applyOperation(op)

        if self.verbose:
            print ('Constructed histogram of "%s", creating datasets'
                   ' "%s" and "%s"') % (inexpr, outbinsds, outvalsds)

    def Remove(self, name):
        """Remove a widget from the dataset."""
        w = self.document.resolve(self.currentwidget, name)
        op = operations.OperationWidgetDelete(w)
        self.document.applyOperation(op)
        if self.verbose:
            print "Removed widget '%s'" % name

    def To(self, where):
        """Change to a graph within the current graph."""

        self.currentwidget = self.document.resolve(self.currentwidget,
                                                   where)

        if self.verbose:
            print "Changed to graph '%s'" % self.currentwidget.path

    def List(self, where='.'):
        """List the contents of a graph."""

        widget = self.document.resolve(self.currentwidget, where)
        children = widget.childnames

        if len(children) == 0:
            print '%30s' % 'No children found'
        else:
            # output format name, type
            for name in children:
                w = widget.getChild(name)
                print '%10s %10s %30s' % (name, w.typename, w.userdescription)

    def Get(self, var):
        """Get the value of a setting."""
        return self.currentwidget.prefLookup(var).val

    def GetChildren(self, where='.'):
        """Return a list of widgets which are children of the widget of the
        path given."""
        return list( self.document.resolve(self.currentwidget,
                                           where).childnames )

    def GetDatasets(self):
        """Return a list of names of datasets."""
        ds = self.document.data.keys()
        ds.sort()
        return ds

    def ResolveReference(self, setn):
        """If the setting is set to a reference, follow the chain of
        references to return the absolute path to the real setting.

        If it is not a reference return None.
        """

        pref = self.currentwidget.prefLookup(setn)
        if pref.isReference():
            real = pref.getReference().resolve(pref)
            return real.path
        else:
            return None

    def Save(self, filename):
        """Save the state to a file."""
        f = open(filename, 'w')
        self.document.saveToFile(f)

    def Set(self, var, val):
        """Set the value of a setting."""
        pref = self.currentwidget.prefLookup(var)

        op = operations.OperationSettingSet(pref, val)
        self.document.applyOperation(op)
        
        if self.verbose:
            print ( "Set setting '%s' to %s" %
                    (var, repr(pref.get())) )

    def SetToReference(self, var, val):
        """Set setting to a reference value."""

        pref = self.currentwidget.prefLookup(var)
        op = operations.OperationSettingSet(pref, setting.Reference(val))
        self.document.applyOperation(op)
        
        if self.verbose:
            print ( "Set setting '%s' to %s" %
                    (var, repr(pref.get())) )

    def SetData(self, name, val, symerr=None, negerr=None, poserr=None):
        """Set dataset with name with values (and optionally errors)."""

        data = datasets.Dataset(val, symerr, negerr, poserr)
        op = operations.OperationDatasetSet(name, data)
        self.document.applyOperation(op)
 
        if self.verbose:
            print "Set variable '%s':" % name
            print " Values = %s" % str( data.data )
            print " Symmetric errors = %s" % str( data.serr )
            print " Negative errors = %s" % str( data.nerr )
            print " Positive errors = %s" % str( data.perr )

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
            print "Set variable '%s' based on expression:" % name
            print " Values = %s" % str( data.data )
            print " Symmetric errors = %s" % str( data.serr )
            print " Negative errors = %s" % str( data.nerr )
            print " Positive errors = %s" % str( data.perr )
            if parametric:
                print " Where t goes form %g:%g in %i steps" % parametric
            print " linked to expression = %s" % repr(linked)

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
            print "Set variable '%s' based on range:" % name
            print " Number of steps = %i" % numsteps
            print " Range of data = %s" % repr(val)
            print " Range of symmetric error = %s" % repr(symerr)
            print " Range of positive error = %s" % repr(poserr)
            print " Range of negative error = %s" % repr(negerr)

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
            print "Set 2D dataset '%s' based on expressions" % name
            print " X expression = %s" % repr(xexpr)
            print " Y expression = %s" % repr(yexpr)
            print " Z expression = %s" % repr(zexpr)
            print " linked to expression = %s" % repr(linked)
            print " Made a dataset (%i x %i)" % (data.data.shape[0],
                                                 data.data.shape[1])

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
            print "Set 2D dataset '%s' based on function of x and y" % name
            print " X steps = %s" % repr(xstep)
            print " Y steps = %s" % repr(ystep)
            print " Expression = %s" % repr(expr)
            print " linked to expression = %s" % repr(linked)
            print " Made a dataset (%i x %i)" % (data.data.shape[0],
                                                 data.data.shape[1])

    def SetData2D(self, name, data, xrange=None, yrange=None):
        """Create a 2D dataset."""

        data = datasets.Dataset2D(data, xrange=xrange, yrange=yrange)
        op = operations.OperationDatasetSet(name, data)
        self.document.applyOperation(op)

        if self.verbose:
            print "Set 2d dataset '%s'" % name

    def SetDataText(self, name, val):
        """Create a text dataset."""

        data = datasets.DatasetText(val)
        op = operations.OperationDatasetSet(name, data)
        self.document.applyOperation(op)

        if self.verbose:
            print "Set text dataset '%s'" % name
            print " Values = %s" % str(data.data)

    def GetData(self, name):
        """Return the data with the name.

        Returns a tuple containing:
            (data, serr, nerr, perr)
        if 'name' is a Dataset, and 
            data
        if 'name' is a DatasetText, where data is a list.

        Values not defined are set to None

        Return copies, so that the original data can't be indirectly modified
        """

        d = self.document.getData(name)
        data = serr = nerr = perr = None
        if isinstance(d, datasets.DatasetText):
            return d.data[:]
        else:

            if d.data is not None:
                data = d.data.copy()
            if d.serr is not None:
                serr = d.serr.copy()
            if d.nerr is not None:
                nerr = d.nerr.copy()
            if d.perr is not None:
                perr = d.perr.copy()

            return (data, serr, nerr, perr)

    def ImportString(self, descriptor, string, useblocks=False):
        """Read data from the string using a descriptor.

        If useblocks is set, then blank lines or the word 'no' are used
        to split the data into blocks. Dataset names are appended with an
        underscore and the block number (starting from 1).

        Returned is a tuple (datasets, errors)
         where datasets is a list of datasets read
         errors is a dict of the datasets with the number of errors while
         converting the data
        """

        op = operations.OperationDataImport(descriptor, datastr=string, useblocks=useblocks)
        dsnames = self.document.applyOperation(op)
        errors = op.simpleread.getInvalidConversions()
            
        if self.verbose:
            print "Imported datasets %s" % (' '.join(dsnames),)
            for name, num in errors.iteritems():
                print "%i errors encountered reading dataset %s" % (num, name)

        return (dsnames, errors)

    def ImportString2D(self, datasetnames, string, xrange=None, yrange=None,
                       invertrows=None, invertcols=None, transpose=None):
        """Read two dimensional data from the string specified.
        datasetnames is a list of datasets to read from the string or a single
        dataset name


        xrange is a tuple containing the range of data in x coordinates
        yrange is a tuple containing the range of data in y coordinates
        if invertrows=True, then rows are inverted when read
        if invertcols=True, then cols are inverted when read
        if transpose=True, then rows and columns are swapped

        """
        
        if type(datasetnames) in (str, unicode):
            datasetnames = [datasetnames]

        op = operations.OperationDataImport2D(
            datasetnames, datastr=string, xrange=xrange,
            yrange=yrange, invertrows=invertrows,
            invertcols=invertcols, transpose=transpose)
        self.document.applyOperation(op)
        if self.verbose:
            print "Imported datasets %s" % (', '.join(datasetnames))

    def ImportFile2D(self, filename, datasetnames, xrange=None, yrange=None,
                     invertrows=None, invertcols=None, transpose=None,
                     prefix="", suffix="", encoding='utf_8',
                     linked=False):
        """Import two-dimensional data from a file.
        filename is the name of the file to read
        datasetnames is a list of datasets to read from the file, or a single
        dataset name

        xrange is a tuple containing the range of data in x coordinates
        yrange is a tuple containing the range of data in y coordinates
        if invertrows=True, then rows are inverted when read
        if invertcols=True, then cols are inverted when read
        if transpose=True, then rows and columns are swapped

        prefix and suffix are prepended and appended to dataset names

        encoding is encoding character set

        if linked=True then the dataset is linked to the file
        """

        # look up filename on path
        realfilename = self.findFileOnImportPath(filename)

        if type(datasetnames) in (str, unicode):
            datasetnames = [datasetnames]

        op = operations.OperationDataImport2D(
            datasetnames, filename=realfilename, xrange=xrange,
            yrange=yrange, invertrows=invertrows,
            invertcols=invertcols, transpose=transpose,
            prefix=prefix, suffix=suffix,
            linked=linked)
        self.document.applyOperation(op)
        if self.verbose:
            print "Imported datasets %s" % (', '.join(datasetnames))

    def ImportFile(self, filename, descriptor, useblocks=False, linked=False,
                   prefix='', suffix='', ignoretext=False, encoding='utf_8'):
        """Read data from file with filename using descriptor.
        If linked is True, the data won't be saved in a saved document,
        the data will be reread from the file.

        If useblocks is set, then blank lines or the word 'no' are used
        to split the data into blocks. Dataset names are appended with an
        underscore and the block number (starting from 1).

        If prefix is set, prefix is prepended to each dataset name
        Suffix is added to each dataset name
        ignoretext ignores lines of text in the file

        Returned is a tuple (datasets, errors)
         where datasets is a list of datasets read
         errors is a dict of the datasets with the number of errors while
         converting the data
        """

        realfilename = self.findFileOnImportPath(filename)

        op = operations.OperationDataImport(
            descriptor, filename=realfilename,
            useblocks=useblocks, linked=linked,
            prefix=prefix, suffix=suffix,
            ignoretext=ignoretext)
        dsnames = self.document.applyOperation(op)
        errors = op.simpleread.getInvalidConversions()
            
        if self.verbose:
            print "Imported datasets %s" % (' '.join(dsnames),)
            for name, num in errors.iteritems():
                print "%i errors encountered reading dataset %s" % (num, name)

        return (dsnames, errors)

    def ImportFileCSV(self, filename, readrows=False, prefix=None,
                      delimiter=',', textdelimiter='"',
                      encoding='utf_8',
                      dsprefix='', dssuffix='',
                      linked=False):
        """Read data from a comma separated file (CSV).

        Data are read from filename
        If readrows is True, then data are read across rather than down
        
        Dataset names are prepended and appended, by dsprefix and dssuffix,
        respectively
         (prefix is backware compatibility only, it adds an underscore
          relative to dsprefix)

        delimiter is the character for delimiting data (usually ',')
        textdelimiter is the character surrounding text (usually '"')
        encoding is the encoding used in the file

        If linked is True the data are linked with the file."""

        # backward compatibility
        if prefix:
            dsprefix = prefix + '_'

        # lookup filename
        realfilename = self.findFileOnImportPath(filename)

        op = operations.OperationDataImportCSV(
            realfilename, readrows=readrows,
            delimiter=delimiter, textdelimiter=textdelimiter,
            encoding=encoding,
            prefix=dsprefix, suffix=dssuffix,
            linked=linked)
        dsnames = self.document.applyOperation(op)
            
        if self.verbose:
            print "Imported datasets %s" % (' '.join(dsnames),)

        return dsnames

    def ImportFITSFile(self, dsname, filename, hdu,
                       datacol = None, symerrcol = None,
                       poserrcol = None, negerrcol = None,
                       linked = False):
        """Import data from a FITS file

        dsname is the name of the dataset
        filename is name of the fits file to open
        hdu is the number/name of the hdu to access

        if the hdu is a table, datacol, symerrcol, poserrcol and negerrcol
        specify the columns containing the data, symmetric error,
        positive and negative errors.

        linked specfies that the dataset is linked to the file
        """

        # lookup filename
        realfilename = self.findFileOnImportPath(filename)

        op = operations.OperationDataImportFITS(
            dsname, realfilename, hdu,
            datacol=datacol, symerrcol=symerrcol,
            poserrcol=poserrcol, negerrcol=negerrcol,
            linked=linked)
        self.document.applyOperation(op)

    def ImportFilePlugin(self, plugin, filename, **args):
        """Import file using a plugin.

        optional arguments:
        prefix: add to start of dataset name (default '')
        suffix: add to end of dataset name (default '')
        linked: link import to file (default False)
        encoding: file encoding (may not be used, default 'utf_8')
        plus arguments to plugin
        """

        realfilename = self.findFileOnImportPath(filename)
        op = operations.OperationDataImportPlugin(plugin, realfilename,
                                                  **args)
        try:
            self.document.applyOperation(op)
        except Exception, ex:
            self.document.log("Error in plugin %s: %s" % (plugin, unicode(ex)))

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

        w = self.document.resolve(self.currentwidget, widget)

        # run action
        w.getAction(action).function()

    def Print(self):
        """Print document."""
        p = qt4.QPrinter()

        if p.setup():
            p.newPage()
            self.document.printTo( p,
                                   range(self.document.getNumberPages()) )
            
    def Export(self, filename, color=True, page=0, dpi=100,
               antialias=True, quality=85, backcolor='#ffffff00'):
        """Export plot to filename.

        color is True or False if color is requested in output file
        page is the pagenumber to export
        dpi is the number of dots per inch for bitmap output files
        antialias antialiases output if True
        quality is a quality parameter for jpeg output
        backcolor is the background color for bitmap files, which is a name or
         a #RRGGBBAA value (red, green, blue, alpha)
        """
        
        self.document.export(filename, page, color=color,
                             dpi=dpi, antialias=antialias,
                             quality=quality, backcolor=backcolor)
            
    def Rename(self, widget, newname):
        """Rename the widget with the path given to the new name.

        eg Rename('graph1/xy1', 'scatter')
        This function does not move widgets."""

        w = self.document.resolve(self.currentwidget, widget)
        op = operations.OperationWidgetRename(w, newname)
        self.document.applyOperation(op)

    def NodeType(self, path):
        """This function treats the set of objects in the widget and
        setting tree as a set of nodes.

        Returns type of node given.
        Return values are: 'widget', 'settings' or 'setting'
        """
        item = self.document.resolveItem(self.currentwidget, path)

        if hasattr(item, 'isWidget') and item.isWidget():
            return 'widget'
        elif isinstance(item, setting.Settings):
            return 'settinggroup'
        else:
            return 'setting'

    def NodeChildren(self, path, types='all'):
        """This function treats the set of objects in the widget and
        setting tree as a set of nodes.

        Returns a list of the names of the children of this node."""

        item = self.document.resolveItem(self.currentwidget, path)

        out = []
        if hasattr(item, 'isWidget') and item.isWidget():
            if types == 'all' or types == 'widget':
                out += item.childnames
            if types == 'all' or types == 'settinggroup':
                out += [s.name for s in item.settings.getSettingsList()]
            if types == 'all' or types == 'setting':
                out += [s.name for s in item.settings.getSettingList()]
        elif isinstance(item, setting.Settings):
            if types == 'all' or types == 'settinggroup':
                out += [s.name for s in item.getSettingsList()]
            if types == 'all' or types == 'setting':
                out += [s.name for s in item.getSettingList()]
        return out

    def WidgetType(self, path):
        """Get the Veusz widget type for a widget with path given.

        Raises a ValueError if the path doesn't point to a widget."""

        item = self.document.resolveItem(self.currentwidget, path)
        if hasattr(item, 'isWidget') and item.isWidget():
            return item.typename
        else:
            raise ValueError, "Path '%s' is not a widget" % path
