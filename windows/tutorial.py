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

import veusz.qtall as qt4

class TutorialStep(qt4.QObject):
    def __init__(self, text, mainwin,
                 nextstep=None, flash=None, disablenext=False):
        qt4.QObject.__init__(self)
        self.text = text
        self.nextstep = nextstep
        self.flash = flash
        self.disablenext = disablenext
        self.mainwin = mainwin

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

<p>Expand the graph widget - click the little arrow to its left in
the editing window - and select the x axis widget.</p>
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

<p>Enter a new label for the widget, by clicking in the text edit box
to the right of "Label", typing some text and press the Enter key.</p>
''', mainwin,
            flash = mainwin.propdock,
            disablenext = True,
            nextstep = StepPropertiesWin2)

        x = mainwin.document.basewidget.getChild(
            'page1').getChild('graph1').getChild('x')
        x.settings.get('label').setOnModified( self.slotNewLabel )

    def slotNewLabel(self, *args):
        self.emit( qt4.SIGNAL('nextStep') )

class StepPropertiesWin2(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<p>Notice that the x axis label of your plot has now been updated.
Veusz supports LaTeX style formatting for labels, so you could include
superscripts, subscripts and fractions.</p>

<p>Other important axis properties include the minimum, maximum values
of the axis and whether the axis is logarithmic.</p>

<p>Click Next to continue.</p>
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

<p>Press Next to continue.</p>
''', mainwin,
            flash=mainwin.treeedit.addtoolbar,
            nextstep=FunctionAdd )

class FunctionAdd(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<h1>Add a function</h1>

<p>Add a function plotting widget to the current graph by clicking on
the flashing icon, or by going to the Insert menu and choosing "Add
function".</p>
''', mainwin,
            flash=mainwin.treeedit.addtoolbar.widgetForAction(
                mainwin.vzactions['add.function']),
            disablenext=True,
            nextstep=FunctionSet)
        self.connect( mainwin.document,
                      qt4.SIGNAL('sigModified'), self.slotDocModified )

    def slotDocModified(self, *args):
        try:
            # if widget has been added, move to next
            self.mainwin.document.basewidget.getChild(
                'page1').getChild('graph1').getChild('function1')
            self.emit( qt4.SIGNAL('nextStep') )
        except AttributeError:
            pass

class FunctionSet(TutorialStep):
    def __init__(self, mainwin):
        TutorialStep.__init__(
            self, '''
<p>You have now added a function widget to the graph widget. By
default function widgets plot y=x.</p>

<p>Go to the Function property and change the function to be
<code>x**2</code>, plotting x squared.</p>

<p>Veusz uses Python syntax for its functions, so the power operator
is <code>**</code>, rather than <code>^</code>.</p>
''', mainwin)

        func = mainwin.document.basewidget.getChild(
            'page1').getChild('graph1').getChild('function1')
        self.setn = func.settings.get('function')
        self.setn.setOnModified( self.slotNewLabel )

    def slotNewLabel(self, *args):
        if self.setn.val.strip() == 'x**2':
            self.emit( qt4.SIGNAL('nextStep') )

# template for text in text view
_texttempl = '''
<head>
<style type="text/css">
h1 { color: blue; font-size: x-large;}
code { color: blue;}
</style>
</head>
<body>
%s
</body>
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

        self.textedit = qt4.QTextEdit(readOnly=True)
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

        self.changeStep(StepIntro)

    def changeStep(self, stepklass):
        '''Apply the next step.'''
        self.step = stepklass(self.mainwin)
        self.connect(self.step, qt4.SIGNAL('nextStep'), self.slotNext)

        self.textedit.setHtml( _texttempl % self.step.text)

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
