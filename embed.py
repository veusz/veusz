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
import os
import os.path
import atexit

# nasty hack to ensure we can load all the veusz components
_dir = os.path.dirname(__file__)
if _dir not in sys.path:
    sys.path.insert(0, _dir)

class Bind1st(object):
    """Bind the first argument of a given function to the given
    parameter. Returns a callable object."""

    def __init__(self, function, arg):
        """function is the function to call, arg is the first param."""
        self.function = function
        self.arg = arg

    def __call__(self, *args, **args2):
        """This makes the object appear to be a function."""
        return self.function(self.arg, *args, **args2)

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

    def __del__(self):
        """Remove the window if deleted."""

        # FIXME: this never gets called. Don't know why.
        if self.window != None:
            self.Close()

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
        Embedded.pipefromveusz_r.read(1)
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

        self.document = None
        self.window = None
        self.plot = None
        self.ci = None

    def _Zoom(self, zoomfactor):
        """Set the plot zoom factor."""
        self.plot.setZoomFactor(zoomfactor)

    def __getattr__(self, name):
        """If there's no name, then lookup in command interpreter."""


        if name in self.ci.cmds:
            return Bind1st(Embedded._runCommand, self.ci.cmds[name])
        else:
            raise AttributeError, "instance has no attribute %s" % name

    def _startThread():
        """Start up the Qt application in a thread."""

        import qt

        class _App(qt.QApplication):
            """An application class which has a notifier to receive
            commands from the main thread."""

            def notification(self, i):
                """Called when the main thread wants to call something in this
                thread.
                """
                Embedded.lock.acquire()

                # get rid of input character
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

        # create a PyQt application instance
        # we fake argv to hope the real argv isn't messed up
        argv = [ sys.argv[0] ]
        app = _App(argv)

        # application has a notifier to know when it needs to do something
        notifier = qt.QSocketNotifier(Embedded.pipetoveusz_r,
                                      qt.QSocketNotifier.Read)

        # connect app to notifier
        app.connect(notifier, qt.SIGNAL('activated(int)'), app.notification)
        notifier.setEnabled(True)

        # start the main Qt event loop
        app.exec_loop()

        # exits when app.quit() is called
       
    _startThread = staticmethod(_startThread)
    
    def _exitQt():
        """Exit the Qt thread."""
        import qt
        qt.qApp.quit()
    _exitQt = staticmethod(_exitQt)

    def _atExit():
        """Close the Qt thread if we're closing the program."""
        if Embedded.lock != None:
            # open window for this embedded instance
            Embedded._runCommand( Embedded._exitQt )
    _atExit = staticmethod(_atExit)
  
atexit.register(Embedded._atExit)
