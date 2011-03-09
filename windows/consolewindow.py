# consolewindow.py
# a python-like qt console

#    Copyright (C) 2003 Jeremy S. Sanders
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

import codeop
import traceback
import sys

import veusz.qtall as qt4

import veusz.document as document
import veusz.utils as utils
import veusz.setting as setting

# TODO - command line completion

class _Writer(object):
    """ Class to behave like an output stream. Pipes input back to
    the specified function."""

    def __init__(self, function):
        """ Set the function output is sent to."""
        self.function = function

    def write(self, text):
        """ Send text to the output function."""
        self.function(text)

    def flush(self):
        """ Does nothing as yet."""
        pass

class _CommandEdit(qt4.QLineEdit):
    """ A special class to allow entering of the command line.

    emits sigEnter if the return key is pressed, and returns command
    The edit control has a history (press up and down keys to access)
    """

    def __init__(self, *args):
        qt4.QLineEdit.__init__(self, *args)
        self.history = []
        self.history_posn = 0
        self.entered_text = ''

        qt4.QObject.connect( self, qt4.SIGNAL("returnPressed()"),
                             self.slotReturnPressed )

        self.setToolTip("Input a python expression here and press enter" )

    def slotReturnPressed(self):
        """ Called if the return key is pressed in the edit control."""

        # retrieve the text
        command = unicode( self.text() )
        self.setText("")

        # keep the command for history
        self.history.append( command )
        self.history_posn = len(self.history)
        self.entered_text = ''

        # tell the console we have a command
        self.emit( qt4.SIGNAL("sigEnter"), command)

    historykeys = (qt4.Qt.Key_Up, qt4.Qt.Key_Down)

    def keyPressEvent(self, key):
        """ Overridden to handle history. """

        qt4.QLineEdit.keyPressEvent(self, key)
        code = key.key()

        # check whether one of the "history keys" has been pressed
        if code in _CommandEdit.historykeys:

            # look for the next or previous history item which our current text
            # is a prefix of
            if self.isModified():
                text = unicode(self.text())
                self.history_posn = len(self.history)
            else:
                text = self.entered_text

            if code == qt4.Qt.Key_Up:
                step = -1
            elif code == qt4.Qt.Key_Down:
                step = 1

            newpos = self.history_posn + step

            while True:
                if newpos >= len(self.history):
                    break
                if newpos < 0:
                    return
                if self.history[newpos].startswith(text):
                    break

                newpos += step

            if newpos >= len(self.history):
                # go back to whatever the user had typed in
                self.history_posn = len(self.history)
                self.setText(self.entered_text)
                return

            # found a relevant history item
            self.history_posn = newpos

            # user has modified text since last set
            if self.isModified():
                self.entered_text = text

            # replace the text in the control
            text = self.history[ self.history_posn ]
            self.setText(text)

introtext=u'''Welcome to <b><font color="purple">Veusz</font></b> --- a scientific plotting application.<br>
Veusz version %s, Copyright \u00a9 2003-2010 Jeremy Sanders &lt;jeremy@jeremysanders.net&gt;<br>
Veusz comes with ABSOLUTELY NO WARRANTY. Veusz is Free Software, and you are<br>
welcome to redistribute it under certain conditions. Enter "GPL()" for details.<br>
This window is a Python command line console and acts as a calculator.<br>
''' % utils.version()

class ConsoleWindow(qt4.QDockWidget):
    """ A python-like qt console."""

    def __init__(self, thedocument, *args):
        qt4.QDockWidget.__init__(self, *args)
        self.setWindowTitle("Console - Veusz")
        self.setObjectName("veuszconsolewindow")

        # arrange sub-widgets in a vbox
        self.vbox = qt4.QWidget()
        self.setWidget(self.vbox)
        vlayout = qt4.QVBoxLayout(self.vbox)
        vlayout.setMargin( vlayout.margin()/4 )
        vlayout.setSpacing( vlayout.spacing()/4 )

        # start an interpreter instance to the document
        self.interpreter = document.CommandInterpreter(thedocument)
        # output from the interpreter goes to self.output_stdxxx

        self.con_stdout = _Writer(self.output_stdout)
        self.con_stderr = _Writer(self.output_stderr)

        self.interpreter.setOutputs(self.con_stdout, self.con_stderr)
        self.stdoutbuffer = ""
        self.stderrbuffer = ""

        # (mostly) hidden notification
        self._hiddennotify = qt4.QLabel()
        vlayout.addWidget(self._hiddennotify)
        self._hiddennotify.hide()

        # the output from the console goes here
        self._outputdisplay = qt4.QTextEdit()
        self._outputdisplay.setReadOnly(True)
        self._outputdisplay.insertHtml( introtext )
        vlayout.addWidget(self._outputdisplay)

        self._hbox = qt4.QWidget()
        hlayout = qt4.QHBoxLayout(self._hbox)
        hlayout.setMargin(0)
        vlayout.addWidget(self._hbox)
        
        self._prompt = qt4.QLabel(">>>")
        hlayout.addWidget(self._prompt)

        # where commands are typed in
        self._inputedit = _CommandEdit()
        hlayout.addWidget(self._inputedit)
        self._inputedit.setFocus()

        # keep track of multiple line commands
        self.command_build = ''

        # get called if enter is pressed in the input control
        self.connect( self._inputedit, qt4.SIGNAL("sigEnter"),
                      self.slotEnter )
        # called if document logs something
        self.connect( thedocument, qt4.SIGNAL("sigLog"),
                      self.slotDocumentLog )

    def _makeTextFormat(self, cursor, color):
        fmt = cursor.charFormat()
        
        if color is not None:
            brush = qt4.QBrush(color)
            fmt.setForeground(brush)
        else:
            # use the default foreground color
            fmt.clearForeground()

        return fmt

    def appendOutput(self, text, style):
        """Add text to the tail of the error log, with a specified style"""
        if style == 'error':
            color = setting.settingdb.color('error')
        elif style == 'command':
            color = setting.settingdb.color('command')
        else:
            color = None

        cursor = self._outputdisplay.textCursor()
        cursor.movePosition(qt4.QTextCursor.End)
        cursor.insertText(text, self._makeTextFormat(cursor, color))
        self._outputdisplay.setTextCursor(cursor)
        self._outputdisplay.ensureCursorVisible()

    def runFunction(self, func):
        """Execute the function within the console window, trapping
        exceptions."""

        # preserve output streams
        temp_stdout = sys.stdout
        temp_stderr = sys.stderr
        sys.stdout = _Writer(self.output_stdout)
        sys.stderr = _Writer(self.output_stderr)

        # catch any exceptions, printing problems to stderr
        try:
            func()
        except Exception, e:
            # print out the backtrace to stderr
            i = sys.exc_info()
            backtrace = traceback.format_exception( *i )
            for l in backtrace:
                sys.stderr.write(l)            

        # return output streams
        sys.stdout = temp_stdout
        sys.stderr = temp_stderr

    def checkVisible(self):
        """If this window is hidden, show it, then hide it again in a few
        seconds."""
        if self.isHidden():
            self._hiddennotify.setText("This window will shortly disappear. "
                                       "You can bring it back by selecting "
                                       "View, Windows, Console Window on the "
                                       "menu.")
            qt4.QTimer.singleShot(5000, self.hideConsole)
            self.show()
            self._hiddennotify.show()

    def hideConsole(self):
        """Hide window and notification widget."""
        self._hiddennotify.hide()
        self.hide()

    def output_stdout(self, text):
        """ Write text in stdout font to the log."""
        self.checkVisible()
        self.appendOutput(text, 'normal')

    def output_stderr(self, text):
        """ Write text in stderr font to the log."""
        self.checkVisible()
        self.appendOutput(text, 'error')

    def insertTextInOutput(self, text):
        """ Inserts the text into the log."""
        self.appendOutput(text, 'normal')

    def slotEnter(self, command):
        """ Called if the return key is pressed in the edit control."""

        newc = self.command_build + '\n' + command

        # check whether command can be compiled
        # c set to None if incomplete
        try:
            c = codeop.compile_command(newc)
        except Exception:
            # we want errors to be caught by self.interpreter.run below
            c = 1

        # which prompt?
        prompt = '>>>'
        if self.command_build != '':
            prompt = '...'

        # output the command in the log pane
        self.appendOutput('%s %s\n' % (prompt, command), 'command')

        # are we ready to run this?
        if c is None or (len(command) != 0 and
                         len(self.command_build) != 0 and
                         (command[0] == ' ' or command[0] == '\t')):
            # build up the expression
            self.command_build = newc
            # modify the prompt
            self._prompt.setText( '...' )
        else:
            # actually execute the command
            self.interpreter.run( unicode(newc) )
            self.command_build = ''
            # modify the prompt
            self._prompt.setText( '>>>' )

    def slotDocumentLog(self, text):
        """Output information if the document logs something."""
        self.output_stderr(text + '\n')
