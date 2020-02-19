# -*- coding: utf-8 -*-
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
###############################################################################

from __future__ import division
import os.path

import sip
from .. import qtall as qt
from .. import utils
from .. import setting

def _(text, disambiguation=None, context="Tutorial"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class TutorialStep(qt.QObject):

    nextStep = qt.pyqtSignal()

    def __init__(self, text, mainwin,
                 nextstep=None, flash=None,
                 disablenext=False,
                 closestep=False,
                 nextonsetting=None,
                 nextonselected=None):

        """
        nextstep is class next TutorialStep class to use
        If flash is set, flash widget
        disablenext: wait until nextStep is emitted before going to next slide
        closestep: add a close button
        nextonsetting: (setnpath, lambda val: ok) -
          check setting to go to next slide
        nextonselected: go to next if widget with name is selected
        """

        qt.QObject.__init__(self)
        self.text = text
        self.nextstep = nextstep
        self.flash = flash
        self.disablenext = disablenext
        self.closestep = closestep
        self.mainwin = mainwin

        self.nextonsetting = nextonsetting
        if nextonsetting is not None:
            mainwin.document.signalModified.connect(self.slotNextSetting)

        self.nextonselected = nextonselected
        if nextonselected is not None:
            mainwin.treeedit.widgetsSelected.connect(self.slotWidgetsSelected)

    def slotNextSetting(self, *args):
        """Check setting to emit next."""
        try:
            setn = self.mainwin.document.resolveSettingPath(
                None, self.nextonsetting[0]).get()
            if self.nextonsetting[1](setn):
                self.nextStep.emit()
        except ValueError:
            pass

    def slotWidgetsSelected(self, widgets, *args):
        """Go to next page if widget selected."""
        if len(widgets) == 1 and widgets[0].name == self.nextonselected:
            self.nextStep.emit()

##########################
## Introduction to widgets

class StepIntro(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>Welcome to Veusz!</h1>

<p>This tutorial aims to get you working with Veusz as quickly as
possible.</p>

<p>You can close this tutorial at any time using the close button to
the top-right of this panel. The tutorial can be replayed in the help
menu.</p>

<p class="usercmd">Press Next to go to the next step</p>
'''), mainwin, nextstep=StepWidgets1)

class StepWidgets1(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>Widgets</h1>

<p>Plots in Veusz are constructed from <i>widgets</i>.  Different
types of widgets are used to make different parts of a plot. For
example, there are widgets for axes, for a graph, for plotting data
and for plotting functions.</p>

<p>There are also special widgets. The grid widget arranges graphs
inside it in a grid arrangement.</p>
'''), mainwin, nextstep=StepWidgets2)

class StepWidgets2(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>Widgets can often be placed inside each other. For instance, a graph
widget is placed in a page widget or a grid widget. Plotting widgets
are placed in graph widget.</p>

<p>You can have multiple widgets of different types. For example, you
can have several graphs on the page, optionally arranged in a
grid. Several plotting widgets and axis widgets can be put in a
graph.</p>
'''), mainwin, nextstep=StepWidgetWin)

class StepWidgetWin(TutorialStep):
    def __init__(self, mainwin):
        t = mainwin.treeedit
        TutorialStep.__init__(
            self, _('''
<h1>Widget editing</h1>

<p>The flashing window is the Editing window, which shows the widgets
currently in the plot in a hierarchical tree. Each widget has a name
(the left column) and a type (the right column).</p>

<p class="usercmd">Press Next to continue.</p>
'''), mainwin,
            nextstep=StepWidgetWinExpand,
            flash=t)

class StepWidgetWinExpand(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>The graph widget is the currently selected widget.</p>

<p class="usercmd">Expand the graph widget - click the arrow or plus
to its left in the editing window - and select the x axis widget.</p>
'''), mainwin,
            disablenext=True,
            nextonselected='x',
            nextstep=StepPropertiesWin)

class StepPropertiesWin(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>Widget properties</h1>

<p>This window shows the properties of the currently selected widget,
the x axis widget of the graph.</p>

<p class="usercmd">Enter a new label for the widget, by clicking in the
text edit box to the right of "Label", typing some text and press the
Enter key.</p>
'''), mainwin,
            flash = mainwin.propdock,
            disablenext = True,
            nextonsetting = ('/page1/graph1/x/label',
                             lambda val: val != ''),
            nextstep = StepPropertiesWin2)

class StepPropertiesWin2(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>Notice that the x axis label of your plot has now been updated.
Veusz supports LaTeX style formatting for labels, so you could include
superscripts, subscripts and fractions.</p>

<p>Other important axis properties include the minimum, maximum values
of the axis and whether the axis is logarithmic.</p>

<p class="usercmd">Click Next to continue.</p>
'''), mainwin, nextstep=WidgetAdd)

class WidgetAdd(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>Adding widgets</h1>

<p>The flashing Add Widget toolbar and the Insert menu add widgets to
the document. New widgets are inserted in the currently selected
widget, if possible, or its parents.</p>

<p>Hold your mouse pointer over one of the toolbar buttons to
see a description of a widget type.</p>

<p class="usercmd">Press Next to continue.</p>
'''), mainwin,
            flash=mainwin.treeedit.addtoolbar,
            nextstep=FunctionAdd )

class FunctionAdd(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>Add a function</h1>

<p>We will now add a function plotting widget to the current
graph.</p>

<p class="usercmd">Click on the flashing icon, or go to the Insert menu
and choosing "Add function".</p>
'''), mainwin,
            flash=mainwin.treeedit.addtoolbar.widgetForAction(
                mainwin.vzactions['add.function']),
            disablenext=True,
            nextonsetting = ('/page1/graph1/function1/function',
                             lambda val: val != ''),
            nextstep=FunctionSet)

class FunctionSet(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>You have now added a function widget to the graph widget. By
default function widgets plot y=x.</p>

<p class="usercmd">Go to the Function property and change the function to
be <code>x**2</code>, plotting x squared.</p>

<p>(Veusz uses Python syntax for its functions, so the power operator
is <code>**</code>, rather than <code>^</code>)</p>
'''), mainwin,
            nextonsetting = ('/page1/graph1/function1/function',
                             lambda val: val.strip() == 'x**2'),
            disablenext = True,
            nextstep=FunctionFormatting)

class FunctionFormatting(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>Formatting</h1>

<p>Widgets have a number of formatting options. The Formatting window
(flashing) shows the options for the currently selected widget, here
the function widget.</p>

<p class="usercmd">Press Next to continue</p>
'''), mainwin,
            flash=mainwin.formatdock,
            nextstep=FunctionFormatLine)

class FunctionFormatLine(TutorialStep):
    def __init__(self, mainwin):

        tb = mainwin.formatdock.tabwidget.tabBar()
        label = qt.QLabel("  ", tb)
        tb.setTabButton(1, qt.QTabBar.LeftSide, label)

        TutorialStep.__init__(
            self, _('''
<p>Different types of formatting properties are grouped under separate
tables. The options for drawing the function line are grouped under
the flashing Line tab (%s).</p>

<p class="usercmd">Click on the Line tab to continue.</p>
''') % utils.pixmapAsHtml(utils.getPixmap('settings_plotline.svg')),
            mainwin,
            flash=label,
            disablenext=True,
            nextstep=FunctionLineFormatting)

        tb.currentChanged[int].connect(self.slotCurrentChanged)

    def slotCurrentChanged(self, idx):
        if idx == 1:
            self.nextStep.emit()

class FunctionLineFormatting(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>Veusz lets you choose a line style, thickness and color for the
function line.</p>

<p class="usercmd">Choose a new line color for the line.</p>
'''),
            mainwin,
            disablenext=True,
            nextonsetting = ('/page1/graph1/function1/Line/color',
                             lambda val: val.strip() != 'black'),
            nextstep=DataStart)

###########
## Datasets

class DataStart(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>Datasets</h1>

<p>Many widgets in Veusz plot datasets. Datasets can be imported from
files, entered manually or created from existing datasets using
operations or expressions.</p>

<p>Imported data can be linked to an external file or embedded in the
document.</p>

<p class="usercmd">Press Next to continue</p>
'''), mainwin,
            nextstep=DataImport)

class DataImport(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>Importing data</h1>

<p>Let us start by importing data.</p>

<p class="usercmd">Click the flashing Data Import icon, or choose
"Import..."  From the Data menu.</p>
'''), mainwin,
            flash=mainwin.datatoolbar.widgetForAction(
                mainwin.vzactions['data.import']),
            disablenext=True,
            nextstep=DataImportDialog)

        # make sure we have the default delimiters
        for k in ( 'importdialog_csvdelimitercombo_HistoryCombo',
                   'importdialog_csvtextdelimitercombo_HistoryCombo' ):
            if k in setting.settingdb:
                del setting.settingdb[k]

        mainwin.dialogShown.connect(self.slotDialogShown)

    def slotDialogShown(self, dialog):
        """Called when a dialog is opened in the main window."""
        from ..dialogs.importdialog import ImportDialog
        if isinstance(dialog, ImportDialog):
            # make life easy by sticking in filename
            dialog.slotReset()
            dialog.filenameedit.setText(
                os.path.join(utils.exampleDirectory, 'tutorialdata.csv'))
            # and choosing tab
            dialog.guessImportTab()
            # get rid of existing values
            self.nextStep.emit()

class DataImportDialog(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>This is the data import dialog. In this tutorial, we have selected
an example CSV (comma separated value) file for you, but you would
normally browse to find your data file.</p>

<p>This example file defines three datasets, <i>alpha</i>, <i>beta</i>
and <i>gamma</i>, entered as columns in the CSV file.</p>

<p class="usercmd">Press Next to continue</p>
'''), mainwin, nextstep=DataImportDialog2)

class DataImportDialog2(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>Veusz will try to guess the datatype - numeric, text or date - from
the data in the file or you can specify it manually.</p>

<p>Several different data formats are supported in Veusz and plugins
can be defined to import any data format. The Link option links data
to the original file.</p>

<p class="usercmd">Click the Import button in the dialog.</p>
'''), mainwin,
            nextstep=DataImportDialog3,
            disablenext=True)
        mainwin.document.signalModified.connect(self.slotDocModified)

    def slotDocModified(self):
        if 'alpha' in self.mainwin.document.data:
            self.nextStep.emit()

class DataImportDialog3(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>Notice how Veusz has loaded the three different datasets from the
file. You could carry on importing new datasets from the Import dialog
box or reopen it later.</p>

<p class="usercmd">Close the Import dialog box.</p>
'''), mainwin,
            disablenext=True,
            nextstep=DataImportDialog4)

        self.timer = qt.QTimer()
        self.timer.timeout.connect(self.slotTimeout)
        self.timer.start(200)

    def slotTimeout(self):
        from ..dialogs.importdialog import ImportDialog
        closed = True
        for dialog in self.mainwin.dialogs:
            if isinstance(dialog, ImportDialog):
                closed = False
        if closed:
            # move forward if no import dialog open
            self.nextStep.emit()

class DataImportDialog4(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>The Data viewing window (flashing) shows the currently loaded
datasets in the document.</p>

<p>Hover your mouse over datasets to get information about them. You
can see datasets in more detail in the Data Edit dialog box.</p>

<p class="usercmd">Click Next to continue</p>
'''), mainwin,
            flash=mainwin.datadock,
            nextstep=AddXYPlotter)

##############
## XY plotting

class AddXYPlotter(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>Plotting data</h1>

<p>The point plotting widget plots datasets loaded in Veusz.</p>

<p class="usercmd">The flashing icon adds a point plotting (xy)
widget. Click on this, or go to the Insert menu and choose "Add xy".</p>
'''), mainwin,
            flash=mainwin.treeedit.addtoolbar.widgetForAction(
                mainwin.vzactions['add.xy']),
            disablenext=True,
            nextonsetting = ('/page1/graph1/xy1/xData',
                             lambda val: val != ''),
            nextstep=SetXY_X)

class SetXY_X(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>The datasets to be plotted are in the widget's properties.</p>

<p class="usercmd">Change the "X data" setting to be the
<code>alpha</code> dataset. You can choose this from the drop down
menu or type it.</p>
'''), mainwin,
            disablenext=True,
            nextonsetting = ('/page1/graph1/xy1/xData',
                             lambda val: val == 'alpha'),
            nextstep=SetXY_Y)

class SetXY_Y(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p class="usercmd">Change the "Y data" setting to be the
<code>beta</code> dataset.</p>
'''), mainwin,
            disablenext=True,
            nextonsetting = ('/page1/graph1/xy1/yData',
                             lambda val: val == 'beta'),
            nextstep=SetXYLine)

class SetXYLine(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>Veusz has now plotted the data on the graph. You can manipulate how
the data are shown using the formatting settings.</p>

<p class="usercmd">Make sure that the line Formatting tab (%s) for the
widget is selected.</p>

<p class="usercmd">Click on the check box next to the Hide option at
the bottom, to hide the line plotted between the data points.</p>
''') % utils.pixmapAsHtml(utils.getPixmap('settings_plotline.svg')),
            mainwin,
            disablenext=True,
            nextonsetting = ('/page1/graph1/xy1/PlotLine/hide',
                             lambda val: val),
            nextstep=SetXYFill)

class SetXYFill(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>Now we will change the point color.</p>

<p class="usercmd">Click on the "Marker fill (%s)" formatting tab.
Change the fill color of the plotted data.</p>
''') % utils.pixmapAsHtml(utils.getPixmap('settings_plotmarkerfill.svg')),
            mainwin,
            disablenext=True,
            nextonsetting = ('/page1/graph1/xy1/MarkerFill/color',
                             lambda val: val != 'black'),
            nextstep=AddXY2nd)

class AddXY2nd(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>Adding a second dataset</h1>

<p>We will now plot dataset <code>alpha</code> against
<code>gamma</code> on the same graph.</p>

<p class="usercmd">Add a second point plotting (xy) widget using the
flashing icon, or go to the Insert menu and choose "Add xy".</p>
'''), mainwin,
            flash=mainwin.treeedit.addtoolbar.widgetForAction(
                mainwin.vzactions['add.xy']),
            disablenext=True,
            nextonsetting = ('/page1/graph1/xy2/xData',
                             lambda val: val != ''),
            nextstep=AddXY2nd_2)

class AddXY2nd_2(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p class="usercmd">Change the "X data" setting to be the
<code>alpha</code> dataset.</p>
'''), mainwin,
            disablenext=True,
            nextonsetting = ('/page1/graph1/xy2/xData',
                             lambda val: val == 'alpha'),
            nextstep=AddXY2nd_3)

class AddXY2nd_3(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p class="usercmd">Next, change the "Y data" setting to be the
<code>gamma</code> dataset.</p>
'''), mainwin,
            disablenext=True,
            nextonsetting = ('/page1/graph1/xy2/yData',
                             lambda val: val == 'gamma'),
            nextstep=AddXY2nd_4)

class AddXY2nd_4(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>We can fill regions under plots using the Fill Below Formatting tab
(%s).</p>

<p class="usercmd">Go to this tab, and unselect the "Hide edge fill"
option.</p>
''') % utils.pixmapAsHtml(utils.getPixmap('settings_plotfillbelow.png')),
            mainwin,
            disablenext=True,
            nextonsetting = ('/page1/graph1/xy2/FillBelow/hide',
                             lambda val: not val),
            nextstep=File1)

class File1(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>Saving</h1>

<p>The document can be saved under the File menu, choosing "Save
as...", or by clicking on the Save icon (flashing).</p>

<p>Veusz documents are simple text files which can be easily modified
outside the program.</p>

<p class="usercmd">Click Next to continue</p>
'''), mainwin,
            flash=mainwin.maintoolbar.widgetForAction(
                mainwin.vzactions['file.save']),
            nextstep=File2)

class File2(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>Exporting</h1>

<p>The document can be exported in scalable (EPS, PDF, SVG and EMF) or
bitmap formats.</p>

<p>The "Export..." command under the File menu exports the selected
page. Alternatively, click on the Export icon (flashing).</p>

<p class="usercmd">Click Next to continue</p>
'''), mainwin,
            flash=mainwin.maintoolbar.widgetForAction(
                mainwin.vzactions['file.export']),
            nextstep=Cut1,
            )

class Cut1(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>Cut and paste</h1>

<p>Widgets can be cut and pasted to manipulate the document.</p>

<p class="usercmd">Select the "graph1" widget in the Editing window.</p>
'''), mainwin,
            disablenext=True,
            nextonselected='graph1',
            nextstep=Cut2)

class Cut2(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p class="usercmd">Now click the Cut icon (flashing) or choose "Cut"
from the Edit menu.</p>

<p>This copies the currently selected widget to the clipboard and
deletes it from the document.</p>
'''), mainwin,
            disablenext=True,
            flash=mainwin.treeedit.edittoolbar.widgetForAction(
                mainwin.vzactions['edit.cut']),
            nextstep=AddGrid)
        mainwin.document.signalModified.connect(self.slotCheckDelete)

    def slotCheckDelete(self, *args):
        d = self.mainwin.document
        try:
            d.resolveWidgetPath(None, '/page1/graph1')
        except ValueError:
            # success!
            self.nextStep.emit()

class AddGrid(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>Adding a grid</h1>

<p>Now we will add a grid widget to paste the graph back into.</p>

<p class="usercmd">Click on the flashing Grid widget icon, or choose
"Add grid" from the Insert menu.</p>
'''), mainwin,
            flash=mainwin.treeedit.addtoolbar.widgetForAction(
                mainwin.vzactions['add.grid']),
            disablenext=True,
            nextonsetting = ('/page1/grid1/rows',
                             lambda val: val != ''),
            nextstep=Paste1)

class Paste1(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p class="usercmd">Now click the Paste icon (flashing) or choose "Paste"
from the Edit menu.</p>

<p>This pastes back the widget from the clipboard.</p>
'''), mainwin,
            disablenext=True,
            flash=mainwin.treeedit.edittoolbar.widgetForAction(
                mainwin.vzactions['edit.paste']),
            nextonsetting = ('/page1/grid1/graph1/leftMargin',
                             lambda val: val != ''),
            nextstep=Paste2)

class Paste2(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p class="usercmd">For a second time, click the Paste icon (flashing)
or choose "Paste" from the Edit menu.</p>

<p>This adds a second copy of the original graph to the grid.</p>
'''), mainwin,
            disablenext=True,
            flash=mainwin.treeedit.edittoolbar.widgetForAction(
                mainwin.vzactions['edit.paste']),
            nextonsetting = ('/page1/grid1/graph2/leftMargin',
                             lambda val: val != ''),
            nextstep=Paste3)

class Paste3(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>Having the graphs side-by-side looks a bit messy. We would like to
change the graphs to be arranged in rows.</p>

<p class="usercmd">Navigate to the grid1 widget properties. Change the
number of columns to 1.</p>
'''), mainwin,
            disablenext=True,
            nextonsetting = ('/page1/grid1/columns',
                             lambda val: val == 1),
            nextstep=Paste4)

class Paste4(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<p>We could now adjust the margins of the graphs and the grid.</p>

<p>Axes can also be shared by the graphs of the grid by moving them
into the grid widget. This shares the same axis scale for graphs.</p>

<p class="usercmd">Click Next to continue</p>
'''), mainwin, nextstep=EndStep)

class EndStep(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, _('''
<h1>The End</h1>

<p>Thank you for working through this Veusz tutorial. We hope you
enjoy using Veusz!</p>

<p>Please send comments, bug reports and suggestions to the
developers via the mailing list.</p>

<p>You can try this tutorial again from the Help menu.</p>
'''), mainwin, closestep=True, disablenext=True)

class TutorialDock(qt.QDockWidget):
    '''A dock tutorial window.'''

    def __init__(self, document, mainwin, *args):
        qt.QDockWidget.__init__(self, *args)
        self.setAttribute(qt.Qt.WA_DeleteOnClose)
        self.setMinimumHeight(300)
        self.setWindowTitle('Tutorial - Veusz')
        self.setObjectName('veusztutorialwindow')

        self.setStyleSheet('background: lightyellow;')

        self.document = document
        self.mainwin = mainwin

        self.layout = l = qt.QVBoxLayout()

        txtdoc = qt.QTextDocument(self)
        txtdoc.setDefaultStyleSheet(
            "p.usercmd { color: blue; } "
            "h1 { font-size: x-large;} "
            "code { color: green;} "
            )
        self.textedit = qt.QTextEdit(readOnly=True)
        self.textedit.setDocument(txtdoc)

        l.addWidget(self.textedit)

        self.buttonbox = qt.QDialogButtonBox()
        self.nextb = self.buttonbox.addButton(
            'Next', qt.QDialogButtonBox.ActionRole)
        self.nextb.clicked.connect(self.slotNext)

        l.addWidget(self.buttonbox)

        # have to use a separate widget as dialog already has layout
        self.widget = qt.QWidget()
        self.widget.setLayout(l)
        self.setWidget(self.widget)

        # timer for controlling flashing
        self.flashtimer = qt.QTimer(self)
        self.flashtimer.timeout.connect(self.slotFlashTimeout)
        self.flash = self.oldflash = None
        self.flashon = False
        self.flashct = 0
        self.flashtimer.start(500)

        self.changeStep(StepIntro)

    def ensureShowFlashWidgets(self):
        '''Ensure we can see the widgets flashing.'''
        w = self.flash
        while w is not None:
            w.show()
            w = w.parent()

    def changeStep(self, stepklass):
        '''Apply the next step.'''

        # this is the current text
        self.step = stepklass(self.mainwin)

        # listen to step for next step
        self.step.nextStep.connect(self.slotNext)

        # update text
        self.textedit.setHtml(self.step.text)

        # handle requests for flashing
        self.flashct = 20
        self.flashon = True
        self.flash = self.step.flash
        if self.flash is not None:
            self.ensureShowFlashWidgets()

        # enable/disable next button
        self.nextb.setEnabled(not self.step.disablenext)

        # add a close button if requested
        if self.step.closestep:
            closeb = self.buttonbox.addButton(
                'Close', qt.QDialogButtonBox.ActionRole)
            closeb.clicked.connect(self.close)

    # work around C/C++ object deleted
    @qt.pyqtSlot()
    def slotFlashTimeout(self):
        '''Handle flashing of UI components.'''

        # because we're flashing random UI components, the C++ object
        # might be deleted, so we have to check before doing things to
        # it: hence the sip.isdeleted

        if ( self.flash is not self.oldflash and self.oldflash is not None
             and not sip.isdeleted(self.oldflash) ):
            # clear any flashing on previous widget
            self.oldflash.setStyleSheet('')
            self.oldflash = None

        if self.flash is not None and not sip.isdeleted(self.flash):
            # set flash state and toggle variable
            if self.flashon:
                self.flash.setStyleSheet('background: yellow;')
            else:
                self.flash.setStyleSheet('')
            self.flashon = not self.flashon
            self.oldflash = self.flash

            # stop flashing after N iterations
            self.flashct -= 1
            if self.flashct == 0:
                self.flash = None

    def slotNext(self):
        """Move to the next page of the tutorial."""
        nextstepklass = self.step.nextstep
        if nextstepklass is not None:
            self.changeStep( nextstepklass )
