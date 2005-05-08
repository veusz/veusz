# A module for embedding Veusz within another python program

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

"""This module allows veusz to be embedded within other Python programs.
For example:

import time
import numarray
import veusz.embed as veusz

g = veusz.Embedded('new win')
g.To( g.Add('page') )
g.To( g.Add('graph') )
g.SetData('x', numarray.arange(20))
g.SetData('y', numarray.arange(20)**2)
g.Add('xy')
g.Zoom(0.5)

time.sleep(60)
g.Close()

More than one embedded window can be opened at once
"""

import sys
import thread
import os.path
import atexit

# nasty hack to ensure we can load all the veusz components
dir = os.path.dirname(__file__)
if dir not in sys.path:
    sys.path.insert(0, dir)

class Embedded(object):
    """An embedded instance of Veusz.

    This embedded instance supports all the normal veusz functions
    """

    # a lock to sync the thread
    lock = None

    # the passed command to the Qt thread
    # command consists of tuple: (bound method, *args, **args)
    command = None

    # values returned by command from Qt thread
    retval = None

    def __init__(self, name = 'Veusz'):
        """Initialse the embedded veusz window.

        name is the name of the window to show.
        This method creates a new thread to run Qt if necessary
        """

        if not Embedded.lock:
            # a lock to ensure we don't get mixed up passing commands
            Embedded.lock = thread.allocate_lock()

            # we notify the QApplication via this pipe
            # sending a character causes the application to execute the command
            Embedded.pipetoveusz_r, w = os.pipe()
            Embedded.pipetoveusz_w = os.fdopen(w, 'w', 0)

            # this is a return pipe, to tell app when command has completed
            # if we get a character, we read the return value from retval
            r, w = os.pipe()
            Embedded.pipefromveusz_r = os.fdopen(r, 'r', 0)
            Embedded.pipefromveusz_w = os.fdopen(w, 'w', 0)

            # actually start the thread
            thread.start_new_thread(Embedded._startThread, ())

        # open window for this embedded instance
        Embedded._runCommand( self._NewWindow, name )

    def _runCommand(cmd, *args, **args2):
        """Execute the given function in the Qt thread with the arguments
        given."""

        assert Embedded.command==None and Embedded.retval==None

        # send the command to the thread
        # we write a character into the pipe to notify the thread
        Embedded.lock.acquire()
        Embedded.command = (cmd, args, args2)
        Embedded.lock.release()
        Embedded.pipetoveusz_w.write('N')

        # wait for command to be executed
        # second thread writes a character into the pipe when done
        r = Embedded.pipefromveusz_r.read(1)
        Embedded.lock.acquire()
        retval = Embedded.retval
        Embedded.command = None
        Embedded.retval = None
        Embedded.lock.release()

        if isinstance(retval, Exception):
            # if an exception happened, reraise as an exception
            raise retval
        else:
            # else return the returned value
            return retval

    _runCommand = staticmethod(_runCommand)

    def _NewWindow(self, name):
        """Start up a new window instance.

        This is called by the constructor
        """

        import windows.simplewindow
        import document

        self.window = windows.simplewindow.SimpleWindow(name)
        self.window.show()
        self.document = self.window.document
        self.plot = self.window.plot
        self.ci = document.CommandInterpreter(self.document)
        self.ci.addCommand('Close', self._Close)
        self.ci.addCommand('Zoom', self._Zoom)

    def _Close(self):
        """Close this window."""
        self.window.close()

    def _Zoom(self, zoomfactor):
        """Set the plot zoom factor."""
        self.plot.setZoomFactor(zoomfactor)

    def __getattr__(self, name):
        """If there's no name, then lookup in command interpreter."""

        class _Bind1st(object):
            """Bind the first argument of a given function to the given
            parameter. Returns a callable object."""
            
            def __init__(self, function, arg):
                """function is the function to call, arg is the first param."""
                self.function = function
                self.arg = arg

            def __call__(self, *args, **args2):
                """This makes the object appear to be a function."""
                return self.function(self.arg, *args, **args2)

        if name in self.ci.cmds:
            return _Bind1st(Embedded._runCommand, self.ci.cmds[name])
        else:
            return None

    def _startThread():
        """Start up the Qt application in a thread."""

        import qt
        import windows

        class _VeuszApp(qt.QApplication):
            """An application class which has a notifier to receive
            commands."""

            def notifier(self):
                """When something needs to be read."""
                Embedded.lock.acquire()

                # get rid of input into buffer
                os.read(Embedded.pipetoveusz_r, 1)

                # handle command
                assert Embedded.command != None
                cmd, args, args2 = Embedded.command
                try:
                    Embedded.retval = cmd(*args, **args2)
                except Exception, e:
                    Embedded.retval = e

                Embedded.lock.release()

                # notify sending thread
                Embedded.pipefromveusz_w.write('r')

        # create a QApplication instance
        argv = [ sys.argv[0] ]
        Embedded.app = _VeuszApp(argv)

        # application has a notifier to know when it needs to do something
        Embedded.notifier = qt.QSocketNotifier( Embedded.pipetoveusz_r,
                                                qt.QSocketNotifier.Read )

        # connect app to notifier
        Embedded.app.connect( Embedded.notifier,
                              qt.SIGNAL('activated(int)'),
                              Embedded.app.notifier )
        Embedded.notifier.setEnabled(True)

        # start the main Qt event loop
        Embedded.app.exec_loop()

        # need this otherwise we get a weird python thread error
        # presumably this is due to the non-Qt thread trying to delete the app
        Embedded.app = None
        
    _startThread = staticmethod(_startThread)
    
    def _exitQt():
        """Exit the Qt thread."""
        Embedded.app.quit()
    _exitQt = staticmethod(_exitQt)

    def _atExit():
        """Close the Qt thread if we're closing the program."""
        if Embedded.lock != None:
            # open window for this embedded instance
            Embedded._runCommand( Embedded._exitQt )
    _atExit = staticmethod(_atExit)
  
atexit.register(Embedded._atExit)
