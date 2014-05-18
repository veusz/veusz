#!/usr/bin/env python

#    Copyright (C) 2005 Jeremy S. Sanders
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
Veusz interface which listens to stdin, and receives commands.
Results are written to stdout

All commands in CommandInterface are supported, plus further commands:
Quit: exit the listening program
Zoom x: Change the zoom factor of the plot to x
"""

from __future__ import division
import sys

from . import qtall as qt4
from .compat import cstr

from .windows.simplewindow import SimpleWindow
from . import document

class ReadingThread(qt4.QThread):
    """Stdin reading thread. Emits newline signals with new data.

    We could use a QSocketNotifier on Unix, but this doesn't work on
    Windows as its stdin is a weird object
    """

    newline = qt4.pyqtSignal(cstr)

    def run(self):
        """Emit lines read from stdin."""
        while True:
            line = sys.stdin.readline()
            if line == '':
                break
            self.newline.emit(line)

class InputListener(qt4.QObject):
    """Class reads text from stdin, in order to send commands to a document."""

    def __init__(self, window):
        """Initialse the listening object to send commands to the
        document given by window."""
        
        qt4.QObject.__init__(self)

        self.window = window
        self.document = window.document
        self.plot = window.plot
        self.pickle = False

        self.ci = document.CommandInterpreter(self.document)
        self.ci.addCommand('Quit', self.quitProgram)
        self.ci.addCommand('Zoom', self.plotZoom)
        self.ci.addCommand('EnableToolbar', self.enableToolbar)
        self.ci.addCommand('Pickle', self.enablePickle)

        self.ci.addCommand('ResizeWindow', self.resizeWindow)
        self.ci.addCommand('SetUpdateInterval', self.setUpdateInterval)
        self.ci.addCommand('MoveToPage', self.moveToPage)

        # reading is done in a separate thread so as not to block
        self.readthread = ReadingThread(self)
        self.readthread.newline.connect(self.processLine)
        self.readthread.start()

    def resizeWindow(self, width, height):
        """ResizeWindow(width, height)

        Resize the window to be width x height pixels."""
        self.window.resize(width, height)

    def setUpdateInterval(self, interval):
        """SetUpdateInterval(interval)

        Set graph update interval.
        interval is in milliseconds (ms)
        set to zero to disable updates
        """
        self.plot.setTimeout(interval)

    def moveToPage(self, pagenum):
        """MoveToPage(pagenum)

        Tell window to show specified pagenumber (starting from 1).
        """
        self.plot.setPageNumber(pagenum-1)

    def quitProgram(self):
        """Exit the program."""
        self.window.close()

    def plotZoom(self, zoomfactor):
        """Set the plot zoom factor."""
        self.window.setZoom(zoomfactor)

    def enableToolbar(self, enable=True):
        """Enable plot toolbar."""
        self.window.enableToolbar(enable)
        
    def enablePickle(self, on=True):
        """Enable/disable pickling of commands to/data from veusz"""
        self.pickle = on

    def processLine(self, line):
        """Process inputted line."""
        if self.pickle:
            # line is repr form of pickled string get get rid of \n
            retn = self.ci.runPickle( eval(line.strip()) )
            sys.stdout.write('%s\n' % repr(retn))
            sys.stdout.flush()
            
        else:
            self.ci.run(line)

def openWindow(args, quiet=False):
    '''Opening listening window.
    args is a list of arguments to the program
    '''
    global _win
    global _listen

    if len(args) > 1:
        name = args[1]
    else:
        name = 'Veusz output'

    _win = SimpleWindow(name)
    if not quiet:
        _win.show()
    _listen = InputListener(_win)

def run():
    '''Actually run the program.'''
    app = qt4.QApplication(sys.argv)
    openWindow(sys.argv)
    app.exec_()

# if ran as a program
if __name__ == '__main__':
    run()
