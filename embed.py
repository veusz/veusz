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
import numpy
import veusz.embed as veusz

g = veusz.Embedded('new win')
g.To( g.Add('page') )
g.To( g.Add('graph') )
g.SetData('x', numpy.arange(20))
g.SetData('y', numpy.arange(20)**2)
g.Add('xy')
g.Zoom(0.5)

time.sleep(60)
g.Close()

More than one embedded window can be opened at once
"""

import atexit
import sys
import os
import os.path
import struct
import new
import cPickle

# check for subprocess
try:
    import subprocess
    have_subprocess = True
except ImportError:
    have_subprocess = False

def Bind1st(function, arg):
    """Bind the first argument of a given function to the given
    parameter."""

    def runner(*args, **args2):
        return function( arg, *args, **args2 )

    return runner

class Embedded(object):
    """An embedded instance of Veusz.

    This embedded instance supports all the normal veusz functions
    """

    remote = None

    def __init__(self, name = 'Veusz', copyof = None):
        """Initialse the embedded veusz window.

        name is the name of the window to show.
        This method creates a new thread to run Qt if necessary
        """

        if not Embedded.remote:
            Embedded.startRemote()

        if not copyof:
            retval = self.sendCommand( (-1, '_NewWindow',
                                         (name,), {}) )
        else:
            retval = self.sendCommand( (-1, '_NewWindowCopy',
                                         (name, copyof.winno), {}) )

        self.winno, cmds = retval

        # add methods corresponding to Veusz commands
        for name, doc in cmds:
            func = Bind1st(self.runCommand, name)
            func.__doc__ = doc    # set docstring
            func.__name__ = name  # make name match what it calls
            method = new.instancemethod(func, Embedded)
            setattr(self, name, method) # assign to self

    def StartSecondView(self, name = 'Veusz'):
        """Provides a second view onto the document of this window.

        Returns an Embedded instance
        """
        return Embedded(name=name, copyof=self)

    def startRemote(cls):
        """Start remote process."""

        # create pipes to talk to and from process
        cls.from_pipe, to_pipe = os.pipe()
        from_pipe, cls.to_pipe = os.pipe()

        # command line to run remote process
        cmdline = [ sys.executable,
                    os.path.join( os.path.dirname(
                    os.path.abspath(__file__)), 'embed_remote.py' ),
                    'RunFromEmbed', 
                    str(to_pipe), str(from_pipe) ]

        # start remote process (using subprocess if it is available)
        if have_subprocess:
            cls.remote = subprocess.Popen(cmdline,
                                          shell=False, bufsize=0, close_fds=False)
        else:
            cls.remote = os.spawnl(os.P_NOWAIT, sys.executable, *cmdline)

        # packet length for command bytes
        cls.cmdlen = len(struct.pack('L', 0))
        atexit.register(cls.exitQt)
    startRemote = classmethod(startRemote)

    def readLenFromSocket(socket, length):
        """Read length bytes from socket."""
        s = ''
        while len(s) < length:
            s += os.read(socket, length-len(s))
        return s
    readLenFromSocket = staticmethod(readLenFromSocket)

    def sendCommand(kls, cmd):
        """Send the command to the remote process."""

        outs = cPickle.dumps(cmd)

        os.write( kls.to_pipe, struct.pack('L', len(outs)) )
        os.write( kls.to_pipe, outs )

        backlen = struct.unpack('L', kls.readLenFromSocket(kls.from_pipe,
                                                           kls.cmdlen))[0]
        rets = kls.readLenFromSocket( kls.from_pipe, backlen )
        retobj = cPickle.loads(rets)
        if isinstance(retobj, Exception):
            raise retobj
        else:
            return retobj
    sendCommand = classmethod(sendCommand)

    def runCommand(self, cmd, *args, **args2):
        """Execute the given function in the Qt thread with the arguments
        given."""
        return self.sendCommand( (self.winno, cmd, args[1:], args2) )

    def exitQt(kls):
        """Exit the Qt thread."""
        kls.sendCommand( (-1, '_Quit', (), {}) )
        os.close(kls.to_pipe)
        os.close(kls.from_pipe)
        kls.to_pipe, kls.from_pipe = -1, -1

    exitQt = classmethod(exitQt)

