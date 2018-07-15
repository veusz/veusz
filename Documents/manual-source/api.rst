================================================
Veusz command line and embedding interface (API)
================================================

.. _Commands:

Introduction
############

Veusz uses a common API, or set of commands, to control the program
via its command line (from the Veusz console; click View, Windows,
Console Window), the embedding interface (when Veusz is embedded in
other Python programs), from within plugins, within documents (VSZ
documents contain commands used to generate the document) or
externally from the operating system command line (using `veusz
--listen`).

As Veusz is a a Python application it uses Python as its scripting
language. You can therefore freely mix Veusz and Python commands on
the Veusz command line (Click View, Windows, Console Window to get
access to the command line). Veusz can also read in Python scripts
from files on the command line (see the :ref:`Load <Command.Load>`
command).

When commands are entered in the command prompt in the Veusz window,
Veusz supports a simplified command syntax, whereq brackets following
commands names, and commas, can replaced by spaces in Veusz commands
(not Python commands). For example, :command:`Add('graph',
name='foo')`, may be entered as :command:`Add 'graph' name='foo'`.

The :command:`numpy` package is already imported into the command line
interface (as `\*`), so you do not need to import it first.

The command prompt supports history (use the up and down cursor keys
to recall previous commands).

Most of the commands listed below can be used in the in-program
command line interface, using the embedding interface or using `veusz
--listen`. Commands specific to particular modes are documented as
such.

Veusz also includes a new object-oriented version of the API, which is
documented at new_api_.

Commands and API
################

We list the allowed set of commands below

Action
------

.. _Command.Action:

:command:`Action('actionname',
componentpath='.')`

Initiates the specified action on the widget (component)
given the action name. Actions perform certain automated
routines. These include "fit" on a fit widget, and
"zeroMargins" on grids.

Add
---

.. _Command.Add:

:command:`Add('widgettype', name='nameforwidget',
autoadd=True, optionalargs)`

The Add command adds a graph into the current widget (See the :ref:`To
<Command.To>` command to change the current position).

The first argument is the type of widget to add. These include
"graph", "page", "axis", "xy" and "grid". :command:`name` is the name
of the new widget (if not given, it will be generated from the type of
the widget plus a number). The :command:`autoadd` parameter if set,
constructs the default sub-widgets this widget has (for example, axes
in a graph).

Optionally, default values for the graph settings may be given, for
example :command:`Add('axis', name='y', direction='vertical')`.

Subsettings may be set by using double underscores, for example
:command:`Add('xy', MarkerFill__color='red',
ErrorBarLine__hide=True)`.

Returns: Name of widget added.

AddCustom
---------

.. _Command.AddCustom:

:command:`AddCustom(type, name, value)`

Add a custom definition for evaluation of expressions. This can define
a constant (can be in terms of other constants), a function of 1 or
more variables, or a function imported from an external python module.

ctype is "constant", "function" or "import".

name is name of constant, or "function(x, y, ...)" or module name.

val is definition for constant or function (both are _strings_), or is
a list of symbols for a module (comma separated items in a string).

If mode is 'appendalways', the custom value is appended to the end of
the list even if there is one with the same name. If mode is
'replace', it replaces any existing definition in the same place in
the list or is appended otherwise. If mode is 'append', then an
existing definition is deleted, and the new one appended to the end.

AddImportPath
-------------

.. _Command.AddImportPath:

:command:`AddImportPath(directory)`

Add a directory to the list of directories to try to import data from.

CloneWidget
-----------

.. _Command.CloneWidget:

:command:`CloneWidget(widget, newparent,
newname=None)`

Clone the widget given, placing the copy in newparent and the name
given.  newname is an optional new name to give it Returns new widget
path.

Close
-----

.. _Command.Close:

:command:`Close()`

Closes the plotwindow. This is only supported in embedded mode.

CreateHistogram
---------------

.. _Command.CreateHistogram:

:command:`CreateHistogram(inexpr, outbinsds,
outvalsds, binparams=None, binmanual=None,
method='counts', cumulative = 'none',
errors=False)`

Histogram an input expression.  inexpr is input expression.  outbinds
is the name of the dataset to create giving bin positions.  outvalsds
is name of dataset for bin values.  binparams is None or (numbins,
minval, maxval, islogbins).  binmanual is None or a list of bin
values.  method is 'counts', 'density', or 'fractions'.  cumulative is
to calculate cumulative distributions which is 'none', 'smalltolarge'
or 'largetosmall'.  errors is to calculate Poisson error bars.

DatasetPlugin
-------------

.. _Command.DatasetPlugin:

:command:`DatasetPlugin(pluginname, fields,
datasetnames={})>`

Use a dataset plugin.  pluginname: name of plugin to use fields: dict
of input values to plugin datasetnames: dict mapping old names to new
names of datasets if they are renamed. The new name None means dataset
is deleted

EnableToolbar
-------------

.. _Command.EnableToolbar:

:command:`EnableToolbar(enable=True)`

Enable/disable the zooming toolbar in the plotwindow. This command is
only supported in embedded mode or from `veusz --listen`.

Export
------

.. _Command.Export:

:command:`Export(filename, color=True, page=0, dpi=100,
antialias=True, quality=85, backcolor='#ffffff00', pdfdpi=150,
svgdpi=96, svgtextastext=False)`

Export the page given to the filename given. The :command:`filename`
must end with the correct extension to get the right sort of output
file. Currrenly supported extensions are '.eps', '.pdf', '.ps',
'.svg', '.jpg', '.jpeg', '.bmp' and '.png'. If :command:`color` is
True, then the output is in colour, else greyscale. :command:`page` is
the page number of the document to export (starting from 0 for the
first page!). A list of pages can be given for multipage formats (.pdf
or .ps).  :command:`dpi` is the number of dots per inch for bitmap
output files.  :command:`antialias` - antialiases output if
True. :command:`quality` is a quality parameter for jpeg
output. :command:`backcolor` is the background color for bitmap files,
which is a name or a #RRGGBBAA value (red, green, blue,
alpha). :command:`pdfdpi` is the dpi to use when exporting EPS or PDF
files. :command:`svgdpi` is the dpi to use when exporting to SVG files.
:command:`svgtextastext` says whether to export SVG text as
text, rather than curves.

FilterDatasets
--------------

.. _Command.FilterDatasets:

:command:`FilterDatasets(filterexpr, datasets, prefix="", suffix="",
invert=False, replaceblanks=False)`

Filter a list of datasets given. Creates new datasets for each with
prefix and suffix added to input dataset names. filterexpr is an input
numpy eexpression for filtering the datasets. If invert is set, the
filter condition is inverted. If replaceblanks is set, filtered values
are not removed, but replaced with a blank or NaN value. This command
only works on 1D numeric, date or text datasets.

ForceUpdate
-----------

.. _Command.ForceUpdate:

:command:`ForceUpdate()`

Force the window to be updated to reflect the current state of the
document. Often used when periodic updates have been disabled (see
SetUpdateInterval). This command is only supported in embedded mode or
from `veusz --listen`.

Get
---

.. _Command.Get:

:command:`Get('settingpath')`

Returns: The value of the setting given by the path.

.. code-block:: python

    >>> Get('/page1/graph1/x/min')
    'Auto'

GetChildren
-----------

.. _Command.GetChildren:

:command:`GetChildren(where='.')`

Returns: The names of the widgets which are children of
the path given

GetClick
--------

.. _Command.GetClick:

:command:`GetClick()`

Waits for the user to click on a graph and returns the
position of the click on appropriate axes. Command only works
in embedded mode.

Returns: A list containing tuples of the form (axispath,
val) for each axis for which the click was in range. The value
is the value on the axis for the click.

GetColormap
-----------

.. _Command.GetColormap:

:command:`GetColormap(name, invert=False, nvals=256)`

Returns a colormap as a numpy array of red, green, blue, alpha values
(ranging from 0 to 255) with the number of steps given.

GetData
-------

.. _Command.GetData:

:command:`GetData(name)`

Returns: For a 1D dataset, a tuple containing the dataset with the
name given. The value is (data, symerr, negerr, poserr), with each a
numpy array of the same size or None. data are the values of the
dataset, symerr are the symmetric errors (if set), negerr and poserr
and negative and positive asymmetric errors (if set). If a text
dataset, return a list of text elements. If the dataset is a date-time
dataset, return a list of Python datetime objects. If the dataset is a
2D dataset return the tuple (data, rangex, rangey), where data is a 2D
numpy array and rangex/y are tuples giving the range of the x and y
coordinates of the data. If it is an ND dataset, return an
n-dimensional array.

.. code-block:: python

    data = GetData('x')
    SetData('x', data[0]*0.1, \*data[1:])

GetDataType
-----------

.. _Command.GetDataType:

:command:`GetDataType(name)`

Get type of dataset with name given. Returns '1d' for a
1d dataset, '2d' for a 2d dataset, 'text' for a text dataset
and 'datetime' for a datetime dataset.

GetDatasets
-----------

.. _Command.GetDatasets:

:command:`GetDatasets()`

Returns: The names of the datasets in the current document.

GPL
---

.. _Command.GPL:

:command:`GPL()`

Print out the GNU Public Licence, which Veusz is licenced under.

ImportFile
----------

.. _Command.ImportFile:

:command:`ImportFile('filename', 'descriptor',
linked=False, prefix='', suffix='',
encoding='utf_8',
renames={})`

Imports data from a file. The arguments are the filename to load data
from and the descriptor.

The format of the descriptor is a list of variable names representing
the columns of the data. For more information see :ref:`Descriptors
<Descriptors>`.

If the linked parameter is set to True, if the document is saved, the
data imported will not be saved with the document, but will be reread
from the filename given the next time the document is opened. The
linked parameter is optional.

If prefix and/or suffix are set, then the prefix and suffix are added
to each dataset name. If set, renames maps imported dataset names to
final dataset names after import.

Returns: A tuple containing a list of the imported datasets and the
number of conversions which failed for a dataset.

Changed in version 0.5: A tuple is returned rather than just the
number of imported variables.

ImportFile2D
------------

.. _Command.ImportFile2D:

:command:`ImportFile2D('filename', datasets,
xrange=None, yrange=None, invertrows=False,
invertcols=False, transpose=False,
prefix='', suffix='', linked=False,
encoding='utf8', renames={})`

Imports two-dimensional data from a file. The required arguments are
the filename to load data from and the dataset name, or a list of
names to use.

filename is a string which contains the filename to use. datasets is
either a string (for a single dataset), or a list of strings (for
multiple datasets).

The xrange parameter is a tuple which contains the range of the X-axis
along the two-dimensional dataset, for example (-1., 1.) represents an
inclusive range of -1 to 1. The yrange parameter specifies the range
of the Y-axis similarly. If they are not specified, (0, N) is the
default, where N is the number of datapoints along a particular axis.

invertrows and invertcols if set to True, invert the rows and columns
respectively after they are read by Veusz. transpose swaps the rows
and columns.

If prefix and/or suffix are set, they are prepended or appended to
imported dataset names. If set, renames maps imported dataset names to
final dataset names after import.

If the linked parameter is True, then the datasets are linked to the
imported file, and are not saved within a saved document.

The file format this command accepts is a two-dimensional matrix of
numbers, with the columns separated by spaces or tabs, and the rows
separated by new lines. The X-coordinate is taken to be in the
direction of the columns. Comments are supported (use `#`, `!` or
`%`), as are continuation characters (`\\`). Separate datasets are
deliminated by using blank lines.

In addition to the matrix of numbers, the various optional parameters
this command takes can also be specified in the data file. These
commands should be given on separate lines before the matrix of
numbers. They are:

#. xrange A B

#. yrange C D

#. invertrows

#. invertcols

#. transpose

ImportFileCSV
-------------

.. _Command.ImportFileCSV:

:command:`ImportFileCSV('filename', readrows=False,
dsprefix='', dssuffix='', linked=False, encoding='utf_8',
renames={})`

This command imports data from a CSV format file. Data are read from
the file using the dataset names given at the top of the files in
columns. Please see the reading data section of this manual for more
information. dsprefix is prepended to each dataset name and dssuffix
is added (the prefix option is deprecated and also addeds an
underscore to the dataset name). linked specifies whether the data
will be linked to the file. renames, if set, provides new names for
datasets after import.

ImportFileFITS
--------------

.. _Command.ImportFileFITS:

:command:`ImportFileFits(filename, items, namemap={},
slices={}, twodranges={}, twod_as_oned=set(\[]),
wcsmodes={}, prefix='', suffix='', renames={},
linked=False)`

Import data from a FITS file.

items is a list of datasets to be imported.  items are formatted like
the following:

::

    '/':               import whole file
    '/hduname':        import whole HDU (image or table)
    '/hduname/column': import column from table HDU

all values in items should be lower case.

HDU names have to follow a Veusz-specific naming. If the HDU has a
standard name (e.g. primary or events), then this is used.  If the
HDU has a EXTVER keyword then this number is appended to this
name.  An extra number is appended if this name is not unique.  If
the HDU has no name, then the name used should be 'hduX', where X
is the HDU number (0 is the primary HDU).

namemap maps an input dataset (using the scheme above for items)
to a Veusz dataset name. Special suffixes can be used on the Veusz
dataset name to indicate that the dataset should be imported
specially.

::

    'foo (+)':  import as +ve error for dataset foo
    'foo (-)':  import as -ve error for dataset foo
    'foo (+-)': import as symmetric error for dataset foo

slices is an optional dict specifying slices to be selected when
importing. For each dataset to be sliced, provide a tuple of
values, one for each dimension. The values should be a single
integer to select that index, or a tuple (start, stop, step),
where the entries are integers or None.

twodranges is an optional dict giving data ranges for 2D
datasets. It maps names to (minx, miny, maxx, maxy).

twod_as_oned: optional set containing 2D datasets to attempt to
read as 1D, treating extra columns as error bars

wcsmodes is an optional dict specfying the WCS import mode for 2D
datasets in HDUs. The keys are '/hduname' and the values can be
'pixel':      number pixel range from 0 to maximum (default)
'pixel_wcs':  pixel number relative to WCS reference pixel
'linear_wcs': linear coordinate system from the WCS keywords
'fraction':   fractional values from 0 to 1.

renames is an optional dict mapping old to new dataset names, to
be renamed after importing

linked specifies that the dataset is linked to the file.

Values under the VEUSZ header keyword can be used to override defaults:

::

    'name': override name for dataset
    'slice': slice on importing (use format "start:stop:step,...")
    'range': should be 4 item array to specify x and y ranges:
        [minx, miny, maxx, maxy]
    'xrange' / 'yrange': individual ranges for x and y
    'xcent' / 'ycent': arrays giving the centres of pixels
    'xedge' / 'yedge': arrays giving the edges of pixels
    'twod_as_oned': treat 2d dataset as 1d dataset with errors
    'wcsmode': use specific WCS mode for dataset (see values above)
    These are specified under the VEUSZ header keyword in the form
        KEY=VALUE
    or for column-specific values
    COLUMNNAME: KEY=VALUE

Returns: list of imported datasets

ImportFileHDF5
--------------

.. _Command.ImportFileHDF5:

:command:`ImportFileHDF5(filename, items, namemap={},
slices={}, twodranges={}, twod_as_oned=set(\[]),
convert_datetime={}, prefix='', suffix='', renames={},
linked=False)`

Import data from a HDF5 file. items is a list of groups and
datasets which can be imported.  If a group is imported, all
child datasets are imported.  namemap maps an input dataset
to a veusz dataset name. Special suffixes can be used on the
veusz dataset name to indicate that the dataset should be
imported specially.

::

    'foo (+)': import as +ve error for dataset foo
    'foo (-)': import as -ve error for dataset foo
    'foo (+-)': import as symmetric error for dataset foo

slices is an optional dict specifying slices to be selected when
importing. For each dataset to be sliced, provide a tuple of values,
one for each dimension. The values should be a single integer to
select that index, or a tuple (start, stop, step), where the entries
are integers or None.

twodranges is an optional dict giving data ranges for 2d datasets. It
maps names to (minx, miny, maxx, maxy).  twod_as_oned: optional set
containing 2d datasets to attempt to read as 1d

convert_datetime should be a dict mapping hdf name to specify
date/time importing.  For a 1d numeric dataset: if this is set to
'veusz', this is the number of seconds since 2009-01-01, if this is
set to 'unix', this is the number of seconds since 1970-01-01.  For a
text dataset, this should give the format of the date/time,
e.g. 'YYYY-MM-DD|T|hh:mm:ss' or 'iso' for iso format.

renames is a dict mapping old to new dataset names, to be renamed
after importing.  linked specifies that the dataset is linked to the
file.

Attributes can be used in datasets to override defaults:

::

    'vsz_name': set to override name for dataset in veusz
    'vsz_slice': slice on importing (use format "start:stop:step,...")
    'vsz_range': should be 4 item array to specify x and y ranges:
        [minx, miny, maxx, maxy]
    'vsz_twod_as_oned': treat 2d dataset as 1d dataset with errors
    'vsz_convert_datetime': treat as date/time, set to one of the values
    above.

For compound datasets these attributes can be given on a per-column
basis using attribute names vsz_attributename_columnname.

Returns: list of imported datasets

ImportFileND
------------

.. _Command.ImportFileND:

:command:`def ImportFileND(comm, filename, dataset, shape=None,
transpose=False, mode='text', csvdelimiter=',', csvtextdelimiter='"',
csvlocale='en_US', prefix="", suffix="", encoding='utf_8',
linked=False)`

Import an n-dimensional dataset from a file. The file should either be
in CSV format (mode='csv') or whitespace-separated text (mode='text').
A one-dimensional dataset is given as a list of numbers on a single
line/row.  A two-dimensional dataset is given by a set of rows.  A
three-dimensional dataset is given by a set of two-dimensional
datasets, with blank lines between them. a four-dimensional dataset is
given by a set of three-dimensional datasets with two blank lines
between each. Each additional dataset increases the separating number
of blank lines by one.  Alternatively, the numbers can be given in any
form (number of numbers on each row) and "shape" is included to
reshape the data into the desired shape.

In the file, or included as parameters above, the command "shape num1
num2..." can be included to reshape the output dataset to the shape
given by the numbers in the row after "shape" (these should be in
separate columns in CSV format). If one of these numbers is -1, then
this dimension is inferred from the number of values and the other
dimensions. Also supported is the "transpose" command or optional
argument which reverses the order of the dimensions.

ImportFilePlugin
----------------

.. _Command.ImportFilePlugin:

:command:`ImportFilePlugin('pluginname', 'filename', \**pluginargs,
linked=False, encoding='utf_8', prefix='', suffix='', renames={})`

Import data from file using import plugin 'pluginname'. The arguments
to the plugin are given, plus optionally a text encoding, and prefix
and suffix to prepend or append to dataset names.  renames, if set,
provides new names for datasets after import.

ImportFITSFile
--------------

.. _Command.ImportFITSFile:

:command:`ImportFITSFile(datasetname, filename, hdu, datacol='A',
symerrcol='B', poserrcol='C', negerrcol='D', linked=True/False,
renames={})`

This command is deprecated. Please do not use in new code, but instead
use ImportFileFITS.

This command does a simple import from a FITS file. The FITS format is
used within the astronomical community to transport binary data. For a
more powerful FITS interface, you can use PyFITS within your scripts.

The datasetname is the name of the dataset to import, the filename is
the name of the FITS file to import from. The hdu parameter specifies
the HDU to import data from (numerical or a name).

If the HDU specified is a primary HDU or image extension, then a
two-dimensional dataset is loaded from the file. The optional
parameters (other than linked) are ignored. Any WCS information within
the HDU are used to provide a suitable xrange and yrange.

If the HDU is a table, then the datacol parameter must be specified
(and optionally symerrcol, poserrcol and negerrcol). The dataset is
read in from the named column in the table. Any errors are read in
from the other specified columns.

If linked is True, then the dataset is not saved with a saved
document, but is reread from the data file each time the document is
loaded.  renames, if set, provides new names for datasets after
import.

ImportString
------------

.. _Command.ImportString:

:command:`ImportString('descriptor',
'data')`

Like, :ref:`ImportFile <Command.ImportFile>`, but loads the data from
the specfied string rather than a file. This allows data to be easily
embedded within a document. The data string is usually a multi-line
Python string.

Returns: A tuple containing a list of the imported datasets and the
number of conversions which failed for a dataset.

Changed in version 0.5: A tuple is returned rather than just the
number of imported variables.

.. code-block:: python

    ImportString('x y', '''
        1   2
        2   5
        3   10
    ''')

ImportString2D
--------------

.. _Command.ImportString2D:

:command:`ImportString2D(datasets, string, xrange=None, yrange=None,
invertrows=None, invertcols=None, transpose=None)`

Imports a two-dimensional dataset from the string given. This is
similar to the :ref:`ImportFile2D <Command.ImportFile2D>` command,
with the same dataset format within the string. The optional values
are also listed there. The various controlling parameters can be set
within the string. See the :ref:`ImportFile2D <Command.ImportFile2D>`
section for details.

ImportStringND
--------------

.. _Command.ImportStringND:

:command:`ImportStringND(dataset, string, shape=None,
transpose=False)`

Imports a n-dimensional dataset from the string given. This is similar
to the :ref:`ImportFileND <Command.ImportFileND>` command. Please look
there for more detail and the description of the optional parameters
and in-stream allowed parameters.

IsClosed
--------

.. _Command.IsClosed:

:command:`IsClosed()`

Returns a boolean value telling the caller whether the plotting window
has been closed.

Note: this command is only supported in the embedding interface.

List
----

.. _Command.List:

:command:`List(where='.')`

List the widgets which are contained within the widget with the path
given, the type of widgets, and a brief description.

Load
----

.. _Command.Load:

:command:`Load('filename.vsz')`

Loads the veusz script file given. The script file can be any Python
code. The code is executed using the Veusz interpreter.

Note: this command is only supported at the command line and not in a
script. Scripts may use the python :command:`execfile` function
instead.

MoveToPage
----------

.. _Command.MoveToPage:

:command:`MoveToPage(pagenum)`

Updates window to show the page number given of the document.

Note: this command is only supported in the embedding interface or
`veusz --listen`.

ReloadData
----------

.. _Command.ReloadData:

:command:`ReloadData()`

Reload any datasets which have been linked to files.

Returns: A tuple containing a list of the imported datasets and the
number of conversions which failed for a dataset.

Rename
------

.. _Command.Rename:

:command:`Remove('widgetpath', 'newname')`

Rename the widget at the path given to a new name. This command does
not move widgets.  See :ref:`To <Command.To>` for a description of the
path syntax. '.' can be used to select the current widget.

Remove
------

.. _Command.Remove:

:command:`Remove('widgetpath')`

Remove the widget selected using the path. See :ref:`To <Command.To>`
for a description of the path syntax.

ResizeWindow
------------

.. _Command.ResizeWindow:

:command:`ResizeWindow(width, height)`

Resizes window to be width by height pixels.

Note: this command is only supported in the embedding interface or
`veusz --listen`.

Save
----

.. _Command.Save:

:command:`Save('filename.vsz')`

Save the current document under the filename
given.

Set
---

.. _Command.Set:

:command:`Set('settingpath', val)`

Set the setting given by the path to the value given. If the type of
:command:`val` is incorrect, an :command:`InvalidType` exception is
thrown. The path to the setting is the optional path to the widget the
setting is contained within, an optional subsetting specifier, and the
setting itself.

.. code-block:: python

    Set('page1/graph1/x/min', -10.)

SetAntiAliasing
---------------

.. _Command.SetAntiAliasing:

:command:`SetAntiAliasing(on)`

Enable or disable anti aliasing in the plot window, replotting the
image.

SetData
-------

.. _Command.SetData:

:command:`SetData(name, val, symerr=None, negerr=None, poserr=None)`

Set the dataset name with the values given. If None is given for an
item, it will be left blank. val is the actual data, symerr are the
symmetric errors, negerr and poserr and the getative and positive
asymmetric errors. The data can be given as lists or numpys.

SetDataExpression
-----------------

.. _Command.SetDataExpression:

:command:`SetDataExpression(name, val, symerr=None, negerr=None,
poserr=None, linked=False, parametric=None)`

Create a new dataset based on the expressions given. The expressions
are Python syntax expressions based on existing datasets.

If linked is True, the dataset will change as the datasets in the
expressions change.

Parametric can be set to a tuple of (minval, maxval,
numitems). :command:`t` in the expression will iterate from minval to
maxval in numitems values.

SetDataND
---------

.. _Command.SetDataND:

:command:`SetDataRange(name, val)`

Set a n-dimensional dataset to be the values given by val. val should
be an n-dimensional numpy array of values, or a list of lists.

SetDataRange
------------

.. _Command.SetDataRange:

:command:`SetDataRange(name, numsteps, val, symerr=None, negerr=None,
poserr=None, linked=False)`

Set dataset to be a range of values with numsteps steps. val is tuple
made up of (minimum value, maximum value). symerr, negerr and poserr
are optional tuples for the error bars.

If linked is True, the dataset can be saved in a document as a
SetDataRange, otherwise it is expanded to the values which would make
it up.

SetData2D
---------

.. _Command.SetData2D:

:command:`SetData2D('name', val, xrange=(A,B), yrange=(C,D),
xgrid=[1,2,3...], ygrid=[4,5,6...])`

Creates a two-dimensional dataset with the name given. val is either a
two-dimensional numpy array, or is a list of lists, with each list in
the list representing a row. Do not give xrange if xgrid is set and do
not give yrange if ygrid is set, and vice versa.

xrange and yrange are optional tuples giving the inclusive range of
the X and Y coordinates of the data. xgrid and ygrid are optional
lists, tuples or arrays which give the coordinates of the edges of the
pixels. There should be one more item in each array than pixels.

SetData2DExpression
-------------------

.. _Command.SetData2DExpression:

:command:`SetData2DExpression('name', expr, linked=False)`

Create a 2D dataset based on expressions.  name is the new dataset
name expr is an expression which should return a 2D array linked
specifies whether to permanently link the dataset to the expressions.

SetData2DExpressionXYZ
----------------------

.. _Command.SetData2DExpressionXYZ:

:command:`SetData2DExpressionXYZ('name', 'xexpr', 'yexpr', 'zexpr',
linked=False)`

Create a 2D dataset based on three 1D expressions. The x, y
expressions need to evaluate to a grid of x, y points, with the z
expression as the 2D value at that point. Currently only linear fixed
grids are supported. This function is intended to convert calculations
or measurements at fixed points into a 2D dataset easily. Missing
values are filled with NaN.

SetData2DXYFunc
---------------

.. _Command.SetData2DXYFunc:

:command:`SetData2DXYFunc('name', xstep, ystep, 'expr', linked=False)`

Construct a 2D dataset using a mathematical expression of "x" and
"y". The x values are specified as (min, max, step) in xstep as a
tuple, the y values similarly. If linked remains as False, then a real
2D dataset is created, where values can be modified and the data are
stored in the saved file.

SetDataDateTime
---------------

.. _Command.SetDataDateTime:

:command:`SetDataDateTime('name', vals)`

Creates a datetime dataset of name given. vals is a list of Python
datetime objects.

SetDataText
-----------

.. _Command.SetDataText:

:command:`SetDataText(name, val)`

Set the text dataset name with the values given.  :command:`val` must
be a type that can be converted into a Python list.

.. code-block:: python

    SetDataText('mylabel', ['oranges', 'apples', 'pears', 'spam'])

SetToReference
--------------

.. _Command.SetToReference:

:command:`SetToReference(setting, refval)`

Link setting given to other setting refval.

SetUpdateInterval
-----------------

.. _Command.SetUpdateInterval:

:command:`SetUpdateInterval(interval)`

Tells window to update every interval milliseconds at most. The value
0 disables updates until this function is called with a non-zero. The
value -1 tells Veusz to update the window every time the document has
changed. This will make things slow if repeated changes are made to
the document. Disabling updates and using the ForceUpdate command will
allow the user to control updates directly.

Note: this command is only supported in the embedding interface or
`veusz --listen`.

SetVerbose
----------

.. _Command.SetVerbose:

:command:`SetVerbose(v=True)`

If :command:`v` is :command:`True`, then extra information is printed
out by commands.

StartSecondView
---------------

.. _Command.StartSecondView:

:command:`StartSecondView(name = 'window title')`

In the embedding interface, this method will open a new Embedding
interface onto the same document, returning the object. This new
window provides a second view onto the document. It can, for instance,
show a different page to the primary view. name is a window title for
the new window.

Note: this command is only supported in the embedding interface.

TagDatasets
-----------

.. _Command.TagDatasets:

:command:`TagDatasets('tag', ['ds1', 'ds2'...])`

Adds the tag to the list of datasets given..

To
--

.. _Command.To:

:command:`To('widgetpath')`

The To command takes a path to a widget and moves to that widget. For
example, this may be "/", the root widget, "graph1",
"/page1/graph1/x", "../x". The syntax is designed to mimic Unix paths
for files. "/" represents the base widget (where the pages reside),
and ".." represents the widget next up the tree.

Quit
----

.. _Command.Quit:

:command:`Quit()`

Quits Veusz. This is only supported in `veusz --listen`.

WaitForClose
------------

.. _Command.WaitForClose:

:command:`WaitForClose()`

Wait until the plotting window has been closed.

Note: this command is only supported in the embedding interface.

Zoom
----

.. _Command.Zoom:

:command:`Zoom(factor)`

Sets the plot zoom factor, relative to a 1:1 scaling. factor can also
be "width", "height" or "page", to zoom to the page width, height or
page, respectively.

This is only supported in embedded mode or `veusz --listen`.

Security
########

With the 1.0 release of Veusz, input scripts and expressions are
checked for possible security risks. Only a limited subset of Python
functionality is allowed, or a dialog box is opened allowing the user
to cancel the operation. Specifically you cannot import modules, get
attributes of Python objects, access globals() or locals() or do any
sort of file reading or manipulation. Basically anything which might
break in Veusz or modify a system is not supported. In addition
internal Veusz functions which can modify a system are also warned
against, specifically Print(), Save() and Export().

If you are running your own scripts and do not want to be bothered by
these dialogs, you can run veusz with the :command:`--unsafe-mode`
option.

Using Veusz from other programs
###############################

Non-Qt Python programs
----------------------

Veusz can be used as a Python module for plotting data. There are two
ways to use the module: (1) with an older path-based Veusz commands,
used in Veusz saved document files or (2) using an object-oriented
interface. With the old style method the user uses a unix-path
inspired API to navigate the widget tree and add or manipulate
widgets. With the new style interface, the user navigates the tree
with attributes of the ``Root`` object to access Nodes. The new
interface is likely to be easier to use unless you are directly
translating saved files.

Older path-based interface
--------------------------

.. code-block:: python

    """An example embedding program. Veusz needs to be installed into
    the Python path for this to work (use setup.py)
    
    This animates a sin plot, then finishes
    """
    
    import time
    import numpy
    import veusz.embed as veusz
    
    # construct a Veusz embedded window
    # many of these can be opened at any time
    g = veusz.Embedded('window title')
    g.EnableToolbar()
    
    # construct the plot
    g.To( g.Add('page') )
    g.To( g.Add('graph') )
    g.Add('xy', marker='tiehorz', MarkerFill__color='green')
    
    # this stops intelligent axis extending
    g.Set('x/autoExtend', False)
    g.Set('x/autoExtendZero', False)
    
    # zoom out
    g.Zoom(0.8)
    
    # loop, changing the values of the x and y datasets
    for i in range(10):
        x = numpy.arange(0+i/2., 7.+i/2., 0.05)
        y = numpy.sin(x)
        g.SetData('x', x)
        g.SetData('y', y)
    
        # wait to animate the graph
        time.sleep(2)
    
    # let the user see the final result
    print "Waiting for 10 seconds"
    time.sleep(10)
    print "Done!"
    
    # close the window (this is not strictly necessary)
    g.Close()
    
The embed interface has the methods listed in the command line
interface listed in the Veusz manual
https://veusz.github.io/docs/manual.html

Multiple Windows are supported by creating more than one ``Embedded``
object. Other useful methods include:

- ``WaitForClose()`` - wait until window has closed

- ``GetClick()`` - return a list of ``(axis, value)`` tuples where the
  user clicks on a graph

- ``ResizeWndow(width, height)`` - resize window to be ``width`` x
  ``height`` pixels

- ``SetUpdateInterval(interval)`` - set update interval in ms or 0 to
  disable

- ``MoveToPage(page)`` - display page given (starting from 1)

- ``IsClosed()`` - has the page been closed

- ``Zoom(factor)`` - set zoom level (float) or 'page', 'width',
  'height'

- ``Close()`` - close window

- ``SetAntiAliasing(enable)`` - enable or disable antialiasing

- ``EnableToolbar(enable=True)`` - enable plot toolbar

- ``StartSecondView(name='Veusz')`` - start a second view onto the
  document of the current ``Embedded`` object. Returns a new
  ``Embedded`` object.

- ``Wipe()`` - wipe the document of all widgets and datasets.

.. _new_api:

New-style object interface
--------------------------

In Veusz 1.9 or late a new style of object interface is present, which
makes it easier to construct the widget tree. Each widget, group of
settings or setting is stored as a Node object, or its subclass, in a
tree. The root document widget can be accessed with the ``Root``
object. The dot operator "." finds children inside other nodes. In
Veusz some widgets can contain other widgets (Root, pages, graphs,
grids). Widgets contain setting nodes, accessed as attributes. Widgets
can also contain groups of settings, again accessed as attributes.

An example tree for a document (not complete) might look like this

::

    Root
    \-- page1                     (page widget)
        \-- graph1                (graph widget)
            \--  x                (axis widget)
            \--  y                (axis widget)
            \-- function          (function widget)
        \-- grid1                 (grid widget)
            \-- graph2            (graph widget)
                \-- xy1           (xy widget)
                    \-- xData     (setting)
                    \-- yData     (setting)
                    \-- PlotLine  (setting group)
                        \-- width (setting)
                        ...
                    ...
                \-- x             (axis widget)
                \-- y             (axis widget)
            \-- graph3            (graph widget)
                \-- contour1      (contour widget)
                \-- x             (axis widget)
                \-- y             (axis widget)
    
Here the user could access the xData setting node of the
xy1 widget using ``Root.page1.graph2.xy1.xData``. To
actually read or modify the value of a setting, you should get
or set the ``val`` property of the setting node. The line
width could be changed like this

.. code-block:: python

    graph = embed.Root.page1.graph2
    graph.xy1.PlotLine.width.val = '2pt'

For instance, this constructs a simple x-squared plot which
changes to x-cubed:

.. code-block:: python

    import veusz.embed as veusz
    import time

    #  open a new window and return a new Embedded object
    embed = veusz.Embedded('window title')
    #  make a new page, but adding a page widget to the root widget
    page = embed.Root.Add('page')
    #  add a new graph widget to the page
    graph = page.Add('graph')
    #  add a function widget to the graph. The Add() method can take a list of settings
    #  to set after widget creation. Here, "function='x**2'" is equivalent to
    #  function.function.val = 'x**2'
    function = graph.Add('function', function='x**2')

    time.sleep(2)
    function.function.val = 'x**3'
    #  this is the same if the widgets have the default names
    Root.page1.graph1.function1.function.val = 'x**3'

If the document contains a page called "page1" then ``Root.page1`` is
the object representing the page. Similarly, ``Root.page1.graph1`` is
a graph called ``graph1`` in the page. You can also use
dictionary-style indexing to get child widgets,
e.g. Root['page1']['graph1']. This style is easier to use if the names
of widgets contain spaces or if widget names shadow methods or
properties of the Node object, i.e. if you do not control the widget
names.

Widget nodes can contain as children other widgets, groups of
settings, or settings. Groups of settings can contain child
settings. Settings cannot contain other nodes. Here are the useful
operations of Nodes:

.. code-block:: python

    class Node(object):
      """properties:
        path - return path to object in document, e.g. /page1/graph1/function1
        type - type of node: "widget", "settinggroup" or "setting"
        name - name of this node, e.g. "graph1"
        children - a generator to return all the child Nodes of this Node, e.g.
          for c in Root.children:
            print c.path
        children_widgets - generator to return child widget Nodes of this Node
        children_settinggroups - generator for child setting groups of this Node
        children_settings - a generator to get the child settings
        childnames - return a list of the names of the children of this Node
        childnames_widgets - return a list of the names of the child widgets
        childnames_settinggroups - return a list of the names of the setting groups
        childnames_settings - return a list of the names of the settings
        parent - return the Node corresponding to the parent widget of this Node
    
        __getattr__ - get a child Node with name given, e.g. Root.page1
        __getitem__ - get a child Node with name given, e.g. Root['page1']
      """
    
      def fromPath(self, path):
         """Returns a new Node corresponding to the path given, e.g. '/page1/graph1'"""
    
    class SettingNode(Node):
        """A node which corresponds to a setting. Extra properties:
        val - get or set the setting value corresponding to this value, e.g.
         Root.page1.graph1.leftMargin.val = '2cm'
        """
    
    class SettingGroupNode(Node):
        """A node corresponding to a setting group. No extra properties."""
    
    class WidgetNode(Node):
        """A node corresponding to a widget.
    
           property:
             widgettype - get Veusz type of widget
    
           Methods are below."""
    
        def WalkWidgets(self, widgettype=None):
            """Generator to walk widget tree and get widgets below this
            WidgetNode of type given.
    
            widgettype is a Veusz widget type name or None to get all
            widgets."""
    
        def Add(self, widgettype, *args, **args_opt):
            """Add a widget of the type given, returning the Node instance.
            """
    
        def Rename(self, newname):
            """Renames widget to name given.
            Existing Nodes corresponding to children are no longer valid."""
    
        def Action(self, action):
            """Applies action on widget."""
    
        def Remove(self):
            """Removes a widget and its children.
            Existing Nodes corresponding to children are no longer valid."""
    
Note that Nodes are temporary objects which are created on
the fly. A real widget in Veusz can have several different
WidgetNode objects. The operators == and != can test whether
a Node points to the same widget, setting or setting group.

Here is an example to set all functions in the document to
be ``x**2``:

.. code-block:: python

    for n in Root.WalkWidgets(widgettype='function'):
        n.function.val = 'x**2'

Translating old to new style
----------------------------

Here is an example how you might translate the old to new
style interface (this is taken from the ``sin.vsz``
example).

.. code-block:: python

    # old (from saved document file)
    Add('page', name='page1')
    To('page1')
    Add('graph', name='graph1', autoadd=False)
    To('graph1')
    Add('axis', name='x')
    To('x')
    Set('label', '\\\\italic{x}')
    To('..')
    Add('axis', name='y')
    To('y')
    Set('label', 'sin \\\\italic{x}')
    Set('direction', 'vertical')
    To('..')
    Add('xy', name='xy1')
    To('xy1')
    Set('MarkerFill/color', 'cyan')
    To('..')
    Add('function', name='function1')
    To('function1')
    Set('function', 'sin(x)')
    Set('Line/color', 'red')
    To('..')
    To('..')
    To('..')

.. code-block:: python

    # new (in python)
    import veusz.embed
    embed = veusz.embed.Embedded('window title')

    page = embed.Root.Add('page')
    # note: autoAdd=False stops graph automatically adding own axes (used in saved files)
    graph = page.Add('graph', autoadd=False)
    x = graph.Add('axis', name='x')
    x.label.val = '\\\\italic{x}'
    y = graph.Add('axis', name='y')
    y.direction.val = 'vertical'
    xy = graph.Add('xy')
    xy.MarkerFill.color.val = 'cyan'
    func = graph.Add('function')
    func.function.val = 'sin(x)'
    func.Line.color.val = 'red'

PyQt programs
=============

There is no direct PyQt interface. The standard embedding interface
should work, however.

Non Python programs
===================

Support for non Python programs is available in a limited
form. External programs may execute Veusz using :command:`veusz
--listen`. Veusz will read its input from the standard input, and
write output to standard output. This is a full Python execution
environment, and supports all the scripting commands mentioned in
:ref:`Commands <Commands>`, a :command:`Quit()` command, the
:command:`EnableToolbar()` and the :command:`Zoom(factor)` command
listed above. Only one window is supported at once, but many
:command:`veusz --listen` programs may be started.

:command:`veusz --listen` may be used from the shell command line by
doing something like:

.. code-block:: bash

    veusz --listen < in.vsz

where :command:`in.vsz` contains:

.. code-block:: python

    To(Add('page') )
    To(Add('graph') )
    SetData('x', arange(20))
    SetData('y', arange(20)**2)
    Add('xy')
    Zoom(0.5)
    Export("foo.pdf")
    Quit()

A program may interface with Veusz in this way by using the
:command:`popen` C Unix function, which allows a program to be started
having control of its standard input and output. Veusz can then be
controlled by writing commands to an input pipe.
