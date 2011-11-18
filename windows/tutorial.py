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

import os.path

import veusz.qtall as qt4
import veusz.utils as utils

class TutorialStep(qt4.QObject):
    def __init__(self, text, mainwin,
                 nextstep=None, flash=None, disablenext=False,
                 nextonsetting=None):

        """
        nextstep is class next TutorialStep class to use
        If flash is set, flash widget
        disablenext: wait until nextStep is emitted before going to next slide
        nextonsetting: (setnpath, lambda val: ok) -
          check setting to go to next slide
        """

        qt4.QObject.__init__(self)
        self.text = text
        self.nextstep = nextstep
        self.flash = flash
        self.disablenext = disablenext
        self.mainwin = mainwin

        self.nextonsetting = nextonsetting
        if nextonsetting is not None:
            self.connect( mainwin.document,
                          qt4.SIGNAL('sigModified'), self.slotDocModified )

    def slotDocModified(self, *args):
        """Check setting to emit next."""

        try:
            setn = self.mainwin.document.basewidget.prefLookup(
                self.nextonsetting[0]).get()
            if self.nextonsetting[1](setn):
                self.emit( qt4.SIGNAL('nextStep') )
        except ValueError:
            pass

class StepIntro(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<h1>Welcome to Veusz!</h1>

<p>This tutorial aims to get you working with Veusz as quickly as
possible.</p>

<p>You can close at any time and restart the tutorial in the help
menu. Press Next to go to the next step or complete a
requested action.</p>
''', mainwin, nextstep=StepWidgets )

class StepWidgets(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<h1>Widgets</h1>

<p>Plots in Veusz are constructed from <i>widgets</i>.  Different
types of widgets are used to make different parts of a plot. For
example, there are widgets for axes, for a graph, for plotting data
and for functions.</p>

<p>Widget can often be placed inside each other. For instance, a graph
widget is placed in a page widget or a grid widget. Plotting widgets
are placed in graph widget.</p>
''', mainwin, nextstep=StepWidgetWin)

class StepWidgetWin(TutorialStep):
    def __init__(self, mainwin):
        t = mainwin.treeedit
        TutorialStep.__init__(
            self, '''
<h1>Widget editing</h1>

<p>The flashing window is the editing window, which shows the widgets
currently in the plot in a hierarchical tree. Each widget has a name
(the left column) and a type (the right column).</p>
''', mainwin,
            nextstep=StepWidgetWinExpand,
            flash=t)

class StepWidgetWinExpand(TutorialStep):
    def __init__(self, mainwin):
        t = mainwin.treeedit
        TutorialStep.__init__(
            self, '''
<p>The graph widget is the currently selected widget.</p>

<p class="usercmd">Expand the graph widget - click the little arrow to
its left in the editing window - and select the x axis widget.</p>
''', mainwin,
            disablenext=True,
            nextstep=StepPropertiesWin)

        self.connect(t, qt4.SIGNAL('widgetsSelected'), self.slotWidgetsSelected)

    def slotWidgetsSelected(self, widgets, *args):
        if len(widgets) == 1 and widgets[0].name == 'x':
            self.emit( qt4.SIGNAL('nextStep') )

class StepPropertiesWin(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<h1>Widget properties</h1>

<p>This window shows the properties of the currently selected widget,
the x axis widget of the graph.</p>

<p class="usercmd">Enter a new label for the widget, by clicking in the
text edit box to the right of "Label", typing some text and press the
Enter key.</p>
''', mainwin,
            flash = mainwin.propdock,
            disablenext = True,
            nextonsetting = ('/page1/graph1/x/label',
                             lambda val: val != ''),
            nextstep = StepPropertiesWin2)

class StepPropertiesWin2(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<p>Notice that the x axis label of your plot has now been updated.
Veusz supports LaTeX style formatting for labels, so you could include
superscripts, subscripts and fractions.</p>

<p>Other important axis properties include the minimum, maximum values
of the axis and whether the axis is logarithmic.</p>

<p class="usercmd">Click Next to continue.</p>
''', mainwin, nextstep=WidgetAdd)

class WidgetAdd(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<h1>Adding widgets</h1>

<p>The flashing add widget toolbar, or the insert menu, is used to add
widgets.</p>

<p>If you hold your mouse pointer over one of the toolbar buttons you
will see a description of the widget type.</p>

<p>Veusz will place the new widget inside the currently selected
widget, if possible, or its parents.</p>

<p class="usercmd">Press Next to continue.</p>
''', mainwin,
            flash=mainwin.treeedit.addtoolbar,
            nextstep=FunctionAdd )

class FunctionAdd(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<h1>Add a function</h1>

<p>We will now add a function plotting widget to the current
graph.</p>

<p class="usercmd">Click on the flashing icon, or go to the Insert menu
and choosing "Add function".</p>
''', mainwin,
            flash=mainwin.treeedit.addtoolbar.widgetForAction(
                mainwin.vzactions['add.function']),
            disablenext=True,
            nextonsetting = ('/page1/graph1/function1/function',
                             lambda val: val != ''),
            nextstep=FunctionSet)

class FunctionSet(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<p>You have now added a function widget to the graph widget. By
default function widgets plot y=x.</p>

<p class="usercmd">Go to the Function property and change the function to
be <code>x**2</code>, plotting x squared.</p>

<p>(Veusz uses Python syntax for its functions, so the power operator
is <code>**</code>, rather than <code>^</code>)</p>
''', mainwin,
            nextonsetting = ('/page1/graph1/function1/function',
                             lambda val: val.strip() == 'x**2'),
            disablenext = True,
            nextstep=FunctionFormatting)

class FunctionFormatting(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<h1>Formatting</h1>

<p>Widgets have a number of formatting options. The formatting window
shows the options for the currently selected widget, here the function
widget.</p>

<p class="usercmd">Press Next to continue</p>
''', mainwin,
            flash=mainwin.formatdock,
            nextstep=FunctionFormatLine)

class FunctionFormatLine(TutorialStep):
    def __init__(self, mainwin):

        tb = mainwin.formatdock.tabwidget.tabBar()
        label = qt4.QLabel("  ")
        tb.setTabButton(1, qt4.QTabBar.LeftSide, label)

        TutorialStep.__init__(
            self, '''
<p>Different types of formatting properties are grouped under separate
tables. The options for drawing the function line are grouped under
the flashing line tab.</p>

<p class="usercmd">Click on the line tab to continue.</p>
''', mainwin,
            flash=label,
            disablenext=True,
            nextstep=FunctionLineFormatting)

        self.connect(tb, qt4.SIGNAL('currentChanged(int)'),
                     self.slotCurrentChanged)

    def slotCurrentChanged(self, idx):
        if idx == 1:
            self.emit( qt4.SIGNAL('nextStep') )

class FunctionLineFormatting(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<p>Veusz lets you choose a line style, thickness and color for the
function line.</p>

<p class="usercmd">Choose a new line color for the line.</p>
''',
            mainwin,
            disablenext=True,
            nextonsetting = ('/page1/graph1/function1/Line/color',
                             lambda val: val.strip() != 'black'),
            nextstep=DataStart)

class DataStart(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<h1>Datasets</h1>

<p>Many widgets in Veusz plot datasets. Datasets can be imported from
files, entered manually or created from existing datasets using
operations or expressions.</p>

<p>Imported data can be linked to an external file or embedded in the
document.</p>
''', mainwin,
            nextstep=DataImport)

class DataImport(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<h1>Importing data</h1>

<p>Let us start by importing data.</p>

<p class="usercmd">Click the flashing Data Import icon, or choose
"Import..."  From the Data menu.</p>
''', mainwin,
            flash=mainwin.datatoolbar.widgetForAction(
                mainwin.vzactions['data.import']),
            disablenext=True,
            nextstep=DataImportDialog)

        self.connect(mainwin, qt4.SIGNAL('dialogShown'), self.slotDialogShown )

    def slotDialogShown(self, dialog):
        """Called when a dialog is opened in the main window."""
        from veusz.dialogs.importdialog import ImportDialog
        if isinstance(dialog, ImportDialog):
            # make life easy by sticking in filename
            dialog.filenameedit.setText(
                os.path.join(utils.exampleDirectory, 'tutorialdata.csv'))
            # and choosing tab
            dialog.guessImportTab()
            # get rid of existing values
            dialog.methodtab.currentWidget().reset()
            self.emit( qt4.SIGNAL('nextStep') )

class DataImportDialog(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<p>This is the data import dialog. In this tutorial, we have selected
an example CSV (comma separated value) file for you, but you would
normally browse to find your data file.</p>

<p>This example file defines three datasets, <i>alpha</i>, <i>beta</i>
and <i>gamma</i>, entered as columns in the CSV file.</p>

<p class="usercmd">Press Next to continue</p>
''', mainwin, nextstep=DataImportDialog2)

class DataImportDialog2(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<p>Veusz will try to guess the datatype - numeric, text or date - from
the data in the file or you can specify it manually.</p>

<p>Several different data formats are supported in Veusz and plugins
can be defined to import any data format. The Link option links data
to the original file.</p>

<p class="usercmd">Click the Import button in the dialog.</p>
''', mainwin,
            nextstep=DataImportDialog3,
            disablenext=True)
        self.connect( mainwin.document,
                      qt4.SIGNAL('sigModified'), self.slotDocModified )

    def slotDocModified(self):
        if 'alpha' in self.mainwin.document.data:
            self.emit( qt4.SIGNAL('nextStep') )

class DataImportDialog3(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<p>Notice how Veusz has loaded the three different datasets from the
file. You could carry on importing new datasets from the Import dialog
box or reopen it later.</p>

<p class="usercmd">Close the Import dialog box.</p>
''', mainwin,
            disablenext=True,
            nextstep=DataImportDialog4)

        self.timer = qt4.QTimer()
        self.connect( self.timer, qt4.SIGNAL('timeout()'),
                      self.slotTimeout )
        self.timer.start(200)

    def slotTimeout(self):
        from veusz.dialogs.importdialog import ImportDialog
        closed = True
        for dialog in self.mainwin.dialogs:
            if isinstance(dialog, ImportDialog):
                closed = False
        if closed:
            # move forward if no import dialog open
            self.emit( qt4.SIGNAL('nextStep') )

class DataImportDialog4(TutorialStep):
    def __init__(self, mainwin):
        mainwin.datadock.show()
        TutorialStep.__init__(
            self, '''
<p>The dataset viewer shows the currently loaded datasets in the
document.</p>

<p>Hover your mouse over datasets to get information about them. You
can see datasets in more detail in the Data Edit dialog box.</p>

<p class="usercmd">Click Next to continue</p>
''', mainwin,
            flash=mainwin.datadock,
            nextstep=AddXYPlotter)

class AddXYPlotter(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<h1>Plotting data</h1>

<p>The point plotting widget plots datasets loaded in Veusz.</p>

<p class="usercmd">The flashing icon adds a point plotting (xy)
widget. Click on this, or go to the Add menu and choose "Add xy".</p>
''', mainwin,
            flash=mainwin.treeedit.addtoolbar.widgetForAction(
                mainwin.vzactions['add.xy']),
            disablenext=True,
            nextonsetting = ('/page1/graph1/xy1/xData',
                             lambda val: val != ''),
            nextstep=SetXY_X)

class SetXY_X(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<p>The datasets to be plotted are in the widget's properties.</p>

<p class="usercmd">Change the X Dataset setting to be the
<code>alpha</code> dataset. You can choose this from the drop down
menu or type it.</p>
''', mainwin,
            disablenext=True,
            nextonsetting = ('/page1/graph1/xy1/xData',
                             lambda val: val == 'alpha'),
            nextstep=SetXY_Y)

class SetXY_Y(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<p class="usercmd">Change the Y Dataset setting to be the
<code>beta</code> dataset.</p>
''', mainwin,
            disablenext=True,
            nextonsetting = ('/page1/graph1/xy1/yData',
                             lambda val: val == 'beta'),
            nextstep=SetXYLine)

class SetXYLine(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<p>Veusz has now plotted the data on the graph. You can manipulate how
the data are shown using the formatting settings.</p>

<p class="usercmd">Click on the Line formatting settings tab for the
xy widget.</p>

<p class="usercmd">Click on the check box next to the Hide option to
hide the line plotted between the data points.</p>
''', mainwin,
            disablenext=True,
            nextonsetting = ('/page1/graph1/xy1/PlotLine/hide',
                             lambda val: val),
            nextstep=SetXYFill)

class SetXYFill(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<p>Now we will change the point color.</p>

<p class="usercmd">Click on the point fill formatting tab (%s), and
change the fill color of the plotted data.</p>
''' % utils.pixmapAsHtml(utils.getPixmap('settings_plotmarkerfill.png')),
            mainwin,
            disablenext=True,
            nextonsetting = ('/page1/graph1/xy1/MarkerFill/color',
                             lambda val: val != 'black'),
            nextstep=AddXY2)

class AddXY2(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<h1>Adding a second dataset</h1>

<p></p>
'''

class TutorialDock(qt4.QDockWidget):
    '''A dock tutorial window.'''

    def __init__(self, document, mainwin, *args):
        qt4.QDockWidget.__init__(self, *args)
        self.setWindowTitle('Tutorial - Veusz')
        self.setObjectName('veusztutorialwindow')

        self.setStyleSheet('background: lightyellow;')

        self.document = document
        self.mainwin = mainwin

        self.layout = l = qt4.QVBoxLayout()

        txtdoc = qt4.QTextDocument(self)
        txtdoc.setDefaultStyleSheet(
            "p.usercmd { color: blue; } "
            "h1 { font-size: x-large;} "
            "code { color: green;} "
            )
        self.textedit = qt4.QTextEdit(readOnly=True)
        self.textedit.setDocument(txtdoc)

        l.addWidget(self.textedit)

        bb = qt4.QDialogButtonBox()
        closeb = bb.addButton('Close', qt4.QDialogButtonBox.ActionRole)
        self.connect(closeb, qt4.SIGNAL('clicked()'), self.slotClose)
        self.nextb = bb.addButton('Next', qt4.QDialogButtonBox.ActionRole)
        self.connect(self.nextb, qt4.SIGNAL('clicked()'), self.slotNext)

        l.addWidget(bb)

        # have to use a separate widget as dialog already has layout
        self.widget = qt4.QWidget()
        self.widget.setLayout(l)
        self.setWidget(self.widget)

        # timer for controlling flashing
        self.flashtimer = qt4.QTimer(self)
        self.connect(self.flashtimer, qt4.SIGNAL('timeout()'),
                     self.slotFlashTimeout)
        self.flash = self.oldflash = None
        self.flashon = False
        self.flashct = 0
        self.flashtimer.start(500)

        #self.changeStep(StepIntro)
        #self.changeStep(DataStart)
        self.changeStep(SetXYFill)

    def changeStep(self, stepklass):
        '''Apply the next step.'''
        self.step = stepklass(self.mainwin)
        self.connect(self.step, qt4.SIGNAL('nextStep'), self.slotNext)

        self.textedit.setHtml(self.step.text)

        self.flashct = 20
        self.flashon = True
        self.flash = self.step.flash

        self.nextb.setEnabled(not self.step.disablenext)

    def slotFlashTimeout(self):
        '''Handle flashing of UI components.'''

        if self.flash is not self.oldflash and self.oldflash is not None:
            # clear any flashing on previous widget
            self.oldflash.setStyleSheet('')
            self.oldflash = None

        if self.flash:
            if self.flashon:
                self.flash.setStyleSheet('background: lightyellow;')
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
        print "next"
        nextstepklass = self.step.nextstep
        if nextstepklass is not None:
            self.changeStep( nextstepklass )

    def slotClose(self):
        print 'close'
