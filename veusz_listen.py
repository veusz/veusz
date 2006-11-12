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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id$

"""
Veusz interface which listens to stdin, and receives commands.
Results are written to stdout

All commands in CommandInterface are supported, plus further commands:
Quit: exit the listening program
Zoom x: Change the zoom factor of the plot to x
"""

import sys
import os.path

# Allow veusz to be run even if not installed into PYTHONPATH
try:
    import veusz
except ImportError:
    # load in the veusz module, but change its path to
    # the veusz directory, and insert it into sys.modules    
    import __init__ as veusz
    thisdir = os.path.dirname( os.path.abspath(__file__) )
    veusz.__path__ = [thisdir]
    veusz.__name__ = 'veusz'
    sys.modules['veusz'] = veusz

import veusz.qtall as qt4

from veusz.windows.simplewindow import SimpleWindow
import veusz.document as document
from veusz.application import Application

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

        self.notifier = qt4.QSocketNotifier( sys.stdin.fileno(),
                                             qt4.QSocketNotifier.Read )
        self.connect( self.notifier, qt4.SIGNAL('activated(int)'),
                      self.dataReceived )
        self.notifier.setEnabled(True)

    def quitProgram(self):
        """Exit the program."""
        self.window.close()

    def plotZoom(self, zoomfactor):
        """Set the plot zoom factor."""
        self.plot.setZoomFactor(zoomfactor)

    def enableToolbar(self, enable=True):
        """Enable plot toolbar."""
        self.window.enableToolbar(enable)
        
    def enablePickle(self, on=True):
        """Enable/disable pickling of commands to/data from veusz"""
        self.pickle = on

    def dataReceived(self):
        """When a command is received, interpret it."""

        line = sys.stdin.readline()

        if self.pickle:
            # line is repr form of pickled string get get rid of \n
            retn = self.ci.runPickle( eval(line.strip()) )
            sys.stdout.write('%s\n' % repr(retn))
            sys.stdout.flush()
            
        else:
            self.ci.run(line)

def run():
    '''Actually run the program.'''
    app = Application(sys.argv)

    if len(sys.argv) > 1:
        name = sys.argv[1]
    else:
        name = 'Veusz output'
        
    win = SimpleWindow(name)
    win.show()
    app.connect(app, qt4.SIGNAL("lastWindowClosed()"),
                app, qt4.SLOT("quit()"))

    l = InputListener(win)

    app.exec_()

# if ran as a program
if __name__ == '__main__':
    run()
