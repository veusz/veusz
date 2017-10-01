============
Introduction
============

Veusz
=====

Veusz is a scientific plotting package. It is designed to be easy to
use, easily extensible, but powerful. The program features a graphical
user interface (GUI), which works under Unix/Linux, Windows or Mac
OS. It can also be easily scripted (the saved file formats are similar
to Python scripts) or used as module inside Python. Veusz reads data
from a number of different types of data file, it can be manually
entered, or constructed from other datasets.

In Veusz the document is built in an object-oriented fashion, where a
document is built up by a number of widgets in a hierarchy. For
example, multiple function or xy widgets can be placed inside a graph
widget, and many graphs can be placed in a grid widget.

Veusz can be extended by the user easily by adding plugins. Support
for different data file types can be added with import
plugins. Dataset plugins automate the manipulation of datasets. Tools
plugins automate the manipulation of the document.

Installation
============

Please go to the website_ of Veusz to learn more about the
program. Links to binaries, distribution packages and the source
package can be found in downloads_. For source installation, please
see the package INSTALL.

.. _website: https://veusz.github.io/
.. _downloads: https://veusz.github.io/download/

Getting started
===============

Veusz includes a built-in tutorial which starts the first time the
program is run. You can rerun it later from the Help menu. It also
includes many examples_, to show how certain kinds of plots are
produced. For more help and link to a video tutorial, see help_.

.. _examples: https://veusz.github.io/examples/
.. _help: https://veusz.github.io/help-support/

Terminology
===========

Here we define some terminology for future use.

Widget
------

A document and its graphs are built up from widgets.  These widgets
can often by placed within each other, depending on the type of the
widget. A widget has children (those widgets placed within it) and its
parent. The widgets have a number of different settings which modify
their behaviour. These settings are divided into properties, which
affect what is plotted and how it is plotted. These would include the
dataset being plotted or whether an axis is logarithmic.  There are
also formatting settings, including the font to be used and the line
thickness. In addition they have actions, which perform some sort of
activity on the widget or its children, like "fit" for a fit widget.

As an aside, using the scripting interface, widgets are
specified with a "path", like a file in Unix or Windows. These
can be relative to the current widget (do not start with a
slash), or absolute (start with a slash). Examples of
paths include, `/page1/graph1/x`, `x` and `.`.

The widget types include

#. :command:`document` - representing a complete document. A document
   can contain pages. In addition it contains a setting giving the
   page size for the document.

#. :command:`page` - representing a page in a document. One or more
   graphs can be placed on a page, or a grid.

#. :command:`graph` - defining an actual graph. A graph can be placed
   on a page or within a grid. Contained within the graph are its axes
   and plotters. A graph can be given a background fill and a border
   if required. It also has a margin, which specifies how far away
   from the edge of its parent widget to plot the body of the graph.
   A graph can contain several axes, at any position on the plot. In
   addition a graph can use axes defined in parent widgets, shared
   with other graphs.  More than one graph can be placed within in a
   page. The margins can be adjusted so that they lie within or
   besides each other.

#. :command:`grid` - containing one or more graphs. A grid plots
   graphs in a gridlike fashion. You can specify the number of rows
   and columns, and the plots are automatically replotted in the
   chosen arrangement. A grid can contain graphs or axes. If an axis
   is placed in a grid, it can be shared by the graphs in the grid.

#. :command:`axis` - giving the scale for plotting data. An axis
   translates the coordinates of the data to the screen. An axis can
   be linear or logarithmic, it can have fixed endpoints, or can
   automatically get them from the plotted data. It also has settings
   for the axis labels and lines, tick labels, and major and minor
   tick marks.  An axis may be "horizontal" or "vertical" and can
   appear anywhere on its parent graph or grid.  If an axis appears
   within a grid, then it can be shared by all the graphs which are
   contained within the grid.  The :command:`axis-broken` widget is an
   axis sub-type. It is an axis type where there are jumps in the
   scale of the axis.  The :command:`axis-function` widget allows the
   user to create an axis where the values are scaled by a monotonic
   function, allowing non-linear and non-logarithmic axis scales. The
   widget can also be linked to a different axis via the function.

#. plotters - types of widgets which plot data or add other things on
   a graph. There is no actual plotter widget which can be added, but
   several types of plotters listed below. Plotters typically take an
   axis as a setting, which is the axis used to plot the data on the
   graph (default x and y).

   #. :command:`function` - a plotter which plots a function on the
      graph. Functions can be functions of x or y (parametric
      functions are not done yet!), and are defined in Python
      expression syntax, which is very close to most other
      languages. For example `3*x**2 + 2*x - 4`. A number of functions
      are available (e.g. sin, cos, tan, exp, log...). Technically,
      Veusz imports the numpy package when evaluating, so numpy
      functions are available.  As well as the function setting, also
      settable is the line type to plot the function, and the number
      of steps to evaluate the function when plotting. Filling is
      supported above/below/left/right of the function.
   #. :command:`xy` - a plotter which plots scatter, line, or stepped
      plots. This versatile plotter takes an x and y dataset, and
      plots (optional) points, in a chosen marker and colour,
      connecting them with (optional) lines, and plotting (optional)
      error bars. An xy plotter can also plot a stepped line, allowing
      histograms to be plotted (note that it doesn't yet do the
      binning of the data).  The settings for the xy widget are the
      various attibutes for the points, line and error bars, the
      datasets to plot, and the axes to plot on.  The xy plotter can
      plot a label next to each dataset, which is either the same for
      each point or taken from a text dataset.  If you wish to leave
      gaps in a plot, the input value `nan` can be specified in the
      numeric dataset.
   #. :command:`fit` - fit a function to data. This plotter is a like
      the function plotter, but allows fitting of the function to
      data. This is achived by clicking on a "fit" button, or using
      the "fit" action of the widget. The fitter takes a function to
      fit containing the unknowns, e.g. `a*x**2 + b*x + c`, and
      initial values for the variables (here a, b and c). It then fits
      the data (note that at the moment, the fit plotter fits all the
      data, not just the data that can be seen on the graph) by
      minimising the chi-squared.  In order to fit properly, the y
      data (or x, if fitting as a function of x) must have a properly
      defined, preferably symmetric error. If there is none, Veusz
      assumes the same fractional error everywhere, or symmetrises
      asymmetric errors.  Note that more work is required in this
      widget, as if a parameter is not well defined by the data, the
      matrix inversion in the fit will fail. In addition Veusz does
      not supply estimates for the errors or the final chi-squared in
      a machine readable way.  If the fitting parameters vary
      significantly from 1, then it is worth "normalizing" them by
      adding in a factor in the fit equation to bring them to of the
      order of 1.
   #. :command:`bar` - a bar chart which plots sets of data as
      horizontal or vertical bars. Multiple datasets are supported. In
      "grouped" mode the bars are placed side-by-side for each
      dataset. In "stacked" mode the bars are placed on top of each
      other (in the appropriate direction according to the sign of the
      dataset). Bars are placed on coordinates given, or in integer
      values from 1 upward if none are given. Error bars are plotted
      for each of the datasets.  Different fill styles can be given
      for each dataset given. A separate key value can be given for
      each dataset.
   #. :command:`key` - a box which describes the data plotted. If a
      key is added to a plot, the key looks for "key" settings of the
      other data plotted within a graph. If there any it builds up a
      box containing the symbol and line for the plotter, and the text
      in the "key" setting of the widget. This allows a key to be very
      easily added to a plot.  The key may be placed in any of the
      corners of the plot, in the centre, or manually
      placed. Depending on the ordering of the widgets, the key will
      be placed behind or on top of the widget. The key can be filled
      and surrounded by a box, or not filled or surrounded.
   #. :command:`label` - a text label places on a graph. The alignment
      can be adjusted and the font changed. The position of the label
      can be specified in fractional terms of the current graph, or
      using axis coordinates.
   #. :command:`rect, ellipse` - these draw a rectangle or ellipse,
      respectively, of size and rotation given. These widgets can be
      placed directly on the page or on a graph. The centre can be
      given in axis coordinates or fractional coordinates.
   #. :command:`imagefile` - draw an external graphs file on the graph
      or page, with size and rotation given. The centre can be given
      in axis coordinates or fractional coordinates.
   #. :command:`line` - draw a line with optional arrowheads on the
      graph or page. One end can be given in axis coordinates or
      fractional coordinates.
   #. :command:`contour` - plot contours of a 2D dataset on the
      graph. Contours are automatically calculated between the minimum
      and maximum values of the graph or chosen manually. The line
      style of the contours can be chosen individually and the region
      between contours can be filled with shading or color.  2D
      datasets currently consist of a regular grid of values between
      minimum and maximum positions in x and y. They can be
      constructed from three 1D datasets of x, y and z if they form a
      regular x, y grid.
   #. :command:`image` - plot a 2D dataset as a colored
      image. Different color schemes can be chosen. The scaling
      between the values and the image can be specified as linear,
      logarithmic, square-root or square.
   #. :command:`polygon` - plot x and y points from datasets as a
      polygon. The polygon can be placed directly on the page or
      within a graph. Coordinates are either plotted using the axis or
      as fractions of the width and height of the containing widget.
   #. :command:`boxplot` - plot distribution of points in a dataset.
   #. :command:`polar` - plot polar data or functions. This is a
      non-orthogonal plot and is placed directly on the page rather
      than in a graph.
   #. :command:`ternary` - plot data of three variables which add up
      to 100 per cent.This is a non-orthogonal plot and is placed
      directly on the page rather than in a graph.

Settings: properties and formatting
-----------------------------------

The various settings of the widgets come in a number of types,
including integers (e.g. 10), floats (e.g. 3.14), dataset names
(`mydata`), expressions (`x+y`), text (`hi there!`), distances (see
above), options (`horizontal` or `vertical` for axes).

Veusz performs type checks on these parameters. If they
are in the wrong format the control to edit the setting will
turn red. In the command line, a TypeError exception is
thrown.

In the GUI, the current page is replotted if a setting
is changed when enter is pressed or the user moves to another
setting.

The settings are split up into formatting settings,
controlling the appearance of the plot, or properties,
controlling what is plotted and how it is plotted.

Default settings, including the default font and line
style, and the default settings for any graph widget, can be
modified in the "Default styles" dialog box under the "Edit"
menu. Default settings are set on a per-document basis, but
can be saved into a separate file and loaded. A default
default settings file can be given to use for new documents
(set in the preferences dialog).

Datasets
--------

Data are imported into Veusz as a dataset. A dataset is
imported from a file, entered manually, set via the command
line, or linked to other datasets via an expression or
dataset plugin. Each dataset has a unique name in the
document. They can be seen in the dataset browser panel, or
in the Data, Edit dialog box.  To choose the data to be
plotted, the user usually selects the dataset in the
appropriate setting of a widget.

Veusz supports one-dimensional (1D) datasets, which are a
list of values with optional error bars. Error bars can
either be symmetric or asymmetric. Veusz also supports
two-dimensional (2D) data. A 2D dataset is a grid of values,
with either a fixed spacing in coordinates, or with
arbitrary pixel sizes. An n-dimensional (nD) dataset is an
arbitrary matrix of values. These cannot be plotted
directly, but subsets can be plotted using python slice
syntax to convert to 1D or 2D datasets.

In addition to simple numeric datasets, Veusz also supports
date-time datasets. For details see the sections on reading
data. Also supported are text datasets, which are lists of
text strings.

Datasets can either be plain lists of values which are
stored within the document, or they can be linked to a file,
so that the values update if the file is reloaded, or they
can be linked to other datasets via expressions or dataset
plugins.

.. _TextFonts:

Text
--------------------

Veusz understands a limited set of LaTeX-like formatting
for text. There are some differences (for example, `10^23`
puts the 2 and 3 into superscript), but it is fairly
similar. You should also leave out the dollar signs. Veusz
supports superscripts (`^`), subscripts (`_`), brackets for
grouping attributes are `{` and `}`.

Supported LaTeX symbols include: \\AA, \\Alpha, \\Beta,
\\Chi, \\Delta, \\Epsilon, \\Eta, \\Gamma, \\Iota, \\Kappa, \\Lambda, \\Mu,
\\Nu, \\Omega, \\Omicron, \\Phi, \\Pi, \\Psi, \\Rho, \\Sigma, \\Tau, \\Theta,
\\Upsilon, \\Xi, \\Zeta, \\alpha, \\approx, \\ast, \\asymp, \\beta, \\bowtie,
\\bullet, \\cap, \\chi, \\circ, \\cup, \\dagger, \\dashv, \\ddagger, \\deg,
\\delta, \\diamond, \\divide, \\doteq, \\downarrow, \\epsilon, \\equiv,
\\eta, \\gamma, \\ge, \\gg, \\in, \\infty, \\int, \\iota, \\kappa, \\lambda,
\\le, \\leftarrow, \\lhd, \\ll, \\models, \\mp, \\mu, \\neq, \\ni, \\nu, \\odot,
\\omega, \\omicron, \\ominus, \\oplus, \\oslash, \\otimes, \\parallel,
\\perp, \\phi, \\pi, \\pm, \\prec, \\preceq, \\propto, \\psi, \\rhd, \\rho,
\\rightarrow, \\sigma, \\sim, \\simeq, \\sqrt, \\sqsubset, \\sqsubseteq,
\\sqsupset, \\sqsupseteq, \\star, \\stigma, \\subset, \\subseteq, \\succ,
\\succeq, \\supset, \\supseteq, \\tau, \\theta, \\times, \\umid, \\unlhd,
\\unrhd, \\uparrow, \\uplus, \\upsilon, \\vdash, \\vee, \\wedge, \\xi, \\zeta.
Please request additional characters if they are required (and exist
in the unicode character set). Special symbols can be included
directly from a character map.

Other LaTeX commands are supported. `\\\\` breaks a
line. This can be used for simple tables. For example `{a\\\\b}
{c\\\\d}` shows `a c` over `b d`. The command `\\frac{a}{b}`
shows a vertical fraction a/b.

Also supported are commands to change font. The command
`\\font{name}{text}` changes the font text is written in to
name. This may be useful if a symbol is missing from the
current font, e.g. `\\font{symbol}{g}` should produce a
gamma. You can increase, decrease, or set the size of the font
with `\\size{+2}{text}`, `\\size{-2}{text}`, or
`\\size{20}{text}`. Numbers are in points.

Various font attributes can be changed: for example,
`\\italic{some italic text}` (or use `\\textit` or `\\emph`),
`\\bold{some bold text}` (or use `\\textbf`) and `\\underline{some
underlined text}`.

Example text could include `Area / \\pi (10^{-23}
cm^{-2})`, or `\\pi\\bold{g}`.

Veusz plots these symbols with Qt's unicode support. You
can also include special characters directly, by copying and
pasting from a character map application. If your current font
does not contain these symbols then you may get a box
character.

Veusz also supports the evaluation of a Python
expression when text is written to the page. Python code is
written inside the brackets :command:`%{{ }}%`. Note
that the Python evaluation happens before the LaTeX expansion
is done. The return value of the expression is converted to
text using the Python :command:`str()` function. For
example, the expression :command:`%{{2+2}}%` would
write :command:`4`. Custom functions and constants are
supported when evaluation, in addition to the usual numpy
functions. In addition, Veusz defines the following useful
functions and values.

#. :command:`ENVIRON` is the
   :command:`os.environ` dict of environment
   variables. :command:`%{{ENVIRON['USER']}}%`
   would show the current user in unix.

#. :command:`DATE([fmt])` returns the current
   date, by default in ISO format. fmt is an optional
   format specifier using
   :command:`datetime.date.strftime` format
   specifiers.

#. :command:`TIME([fmt])` returns the current
   date/time, by default in ISO format. fmt is an optional
   format specifier using
   :command:`datetime.datetime.strftime` format
   specifiers.

#. :command:`DATA(name[, part])` returns the Veusz
   dataset with given name. For numeric datasets this is a
   numpy array. For numeric datasets with errors, part
   specifies the dataset part to return, i.e. 'data',
   'serr', 'perr', 'nerr'. For example, the mean value of
   a dataset could be shown using
   :command:`%{{mean(DATA('x'))}}%`.

#. :command:`FILENAME()` - returns the current
   document filename. This can include the
   directory/folder of the file. Note that the filename is
   escaped with ESCAPE() so that LaTeX symbols are not
   expanded when shown.

#. :command:`BASENAME()` - returns the current
   document filename, removing the directory or folder
   name Note that the filename is escaped with ESCAPE() so
   that LaTeX symbols are not expanded when shown.

#. :command:`ESCAPE(x)` - escapes any LaTeX
   symbols in x so that they are not interpreted as
   LaTeX.

#. :command:`SETTING(path)` - return the value of
   the Veusz setting given by the full path,
   e.g. :command:`%{{SETTING('/page1/width')}}%`.

Measurements
------------

Distances, widths and lengths in Veusz can be specified in a number of
different ways. These include absolute distances specified in physical
units, e.g. 1cm, 0.05m, 10mm, 5in and 10pt, and relative units, which
are relative to the largest dimension of the page, including 5%, 1/20,
0.05.

Color theme
-----------

From version 1.26, widgets are colored automatically using the color
theme. This theme is specified in the main document widget
settings. Widgets are given the colors in order given the order in a
graph widget. The default theme can be specified in the preferences
dialog box.

To override a theme, the user can manually specify the individual
colors in the custom definitions dialog box. Color `theme1` is used as
the first theme color, then `theme2`, etc.

Axis numeric scales
-------------------

The way in which numbers are formatted in axis scales is chosen
automatically. For standard numerical axes, values are shown with the
`%Vg` formatting (see below). For date axes, an appropriate date
formatting is used so that the interval shown is correct. A format can
be given for an axis in the axis number formatting panel can be given
to explicitly choose a format. Some examples are given in the drop
down axis menu. Hold the mouse over the example for detail.

C-style number formatting is used with a few Veusz specific
extensions. Text can be mixed with format specifiers, which start with
a `%` sign. Examples of C-style formatting include: `%.2f` (decimal
number with two decimal places, e.g. 2.01), `%.3e` (scientific
formatting with three decimal places, e.g. 2.123e-02), `%g` (general
formatting, switching between `%f` and `%e` as appropriate). See
`<http://opengroup.org/onlinepubs/007908799/xsh/fprintf.html>`_ for
details.

Veusz extensions include `%Ve`, which is like `%e` except it displays
scientific notation as written, e.g. 1.2x10^23, rather than
1.2e+23. `%Vg` switches between standard numbers and Veusz scientific
notation for large and small numbers. `%VE` using engineering SI
suffixes to represent large or small numbers (e.g. 1000 is 1k).

Veusz allows dates and times to be formatted using `%VDX` where `X` is
one of the formatting characters for strftime (see
`<http://opengroup.org/onlinepubs/007908799/xsh/strftime.html>`_ for
details). These include `a` for an abbreviated weekday name, `A` for
full weekday name, `b` for abbreviated month name, `B` for full month
name, `c` date and time representaiton, `d` day of month 01..31, `H`
hour as 00..23, `I` hour as 01..12, `j` as day of year 001..366, `m`
as month 01..12, `M` minute as 00..59, `p` AM/PM, `S` second 00..61,
`U` week number of year 00..53 (Sunday as first day of week), `w`
weekday as decimal number 0..6, `W` week number of year (Monday as
first day of week), `x` date representation, `X` time representation,
`y` year without century 00..99 and `Y` year. `%VDVS` is a special
Veusz addon format which shows seconds and fractions of seconds
(e.g. 12.2).

The main window
===============

You should see the main window when you run Veusz (you can just type
the veusz command in Unix).

.. image:: _images/mainwindow.png

The Veusz window is split into several sections. At the top is the
menu bar and tool bar. These work in the usual way to other
applications. Sometimes options are disabled (greyed out) if they do
not make sense to be used. If you hold your mouse over a button for a
few seconds, you will usually get an explanation for what it does
called a "tool tip".

Below the main toolbar is a second toolbar for constructing the graph
by adding widgets (on the left), and some editing buttons. The add
widget buttons add the request widget to the currently selected widget
in the selection window. The widgets are arranged in a tree-like
structure.

Below these toolbars and to the right is the plot window. This is
where the current page of the current document is shown. You can
adjust the size of the plot on the screen (the zoom factor) using the
"View" menu or the zoom tool bar button (the magnifying
glass). Initially you will not see a plot in the plot window, but you
will see the Veusz logo. At the moment you cannot do much else with
the window. In the future you will be able to click on items in the
plot to modify them.

To the left of the plot window is the selection window, and the
properties and formatting windows. The properties window lets you edit
various aspects of the selected widget (such as the minimum and
maximum values on an axis). Changing these values should update the
plot. The formatting lets you modify the appearance of the selected
widget. There are a series of tabs for choosing what aspect to modify.

The various windows can be "dragged" from the main window to "float"
by themselves on the screen.

To the bottom of the window is the console. This window is not shown
by default, but can be enabled in the View menu. The console is a
Veusz and Python command line console. To read about the commands
available see :ref:`Commands <Commands>`. As this is a Python console,
you can enter mathematical expressions (e.g. `1+2.0*cos(pi/4)`) here
and they will be evaluated when you press Enter. The usual special
functions and the operators are supported. You can also assign results
to variables (e.g. `a=1+2`) for use later. The console also supports
command history like many Unix shells. Press the up and down cursor
keys to browse through the history. Command line completion is not
available yet!

There also exists a dataset browsing window, by default to the right
of the screen. This window allows you to view the datasets currently
loaded, their dimensions and type. Hovering a mouse over the size of
the dataset will give you a preview of the data.

My first plot
=============

After opening Veusz, on the left of the main window, you will see a
Document, containing a Page, which contains a Graph with its axes. The
Graph is selected in the selection window. The toolbar above adds a
new widget to the selected widget. If a widget cannot be added to a
selected widget it is disabled. On opening a new document Veusz
automatically adds a new Page and Graph (with axes) to the document.

You will see something like this:

.. image:: _images/winwithgraph.png

Select the x axis which has been added to the document (click on `x`
in the selection window). In the properties window you will see a
variety of different properties you can modify. For instance you can
enter a label for the axis by writing `Area (cm^{2})` in the box next
to label and pressing enter. Veusz supports text in LaTeX-like form
(without the dollar signs). Other important parameters is the `log`
switch which switches between linear and logarithmic axes, and `min`
and `max` which allow the user to specify the minimum and maximum
values on the axes.

The formatting dialog lets you edit various aspects of the graph
appearance. For instance the "Line" tab allows you to edit the line of
the axis. Click on "Line", then you can then modify its colour. Enter
"green" instead of "black" and press enter. Try making the axis label
bold.

Now you can try plotting a function on the graph. If the graph, or its
children are selected, you will then be able to click the "function"
button at the top (a red curve on a graph). You will see a straight
line (y=x) added to the plot. If you select "function1", you will be
able to edit the functional form plotted and the style of its
line. Change the function to `x**2` (x-squared).

We will now try plotting data on the graph. Go to your
favourite text editor and save the following data as
test.dat:

::

    1     0.1   -0.12   1.1    0.1
    2.05  0.12  -0.14   4.08   0.12
    2.98  0.08  -0.1    2.9    0.11
    4.02  0.04  -0.1    15.3   1.0

The first three columns are the x data to plot plus its asymmetric
errors. The final two columns are the y data plus its symmetric
errors. In Veusz, go to the "Data" menu and select "Import". Type the
filename into the filename box, or use the "Browse..." button to
search for the file. You will see a preview of the data pop up in the
box below. Enter `x,+,- y,+-` into the descriptors edit box (note that
commas and spaces in the descriptor are almost interchangeable in
Veusz 1.6 or newer). This describes the format of the data which
describes dataset "x" plus its asymmetric errors, and "y" with its
symmetric errors. If you now click "Import", you will see it has
imported datasets `x` and `y`.

To plot the data you should now click on `graph1` in the tree
window. You are now able to click on the "xy" button (which looks like
points plotted on a graph). You will see your data plotted on the
graph. Veusz plots datasets `x` and `y` by default, but you can change
these in the properties of the "xy" plotter.

You are able to choose from a variety of markers to plot. You can
remove the plot line by choosing the "Plot Line" subsetting, and
clicking on the "hide" option. You can change the colour of the marker
by going to the "Marker Fill" subsetting, and entering a new colour
(e.g. red), into the colour property.
