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
import os.path
import struct
import new
import cPickle

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

    remote = False

    def __init__(self, name = 'Veusz'):
        """Initialse the embedded veusz window.

        name is the name of the window to show.
        This method creates a new thread to run Qt if necessary
        """

        if not Embedded.remote:
            Embedded.startRemote()

        self.winno, cmds = self.sendCommand( (-1, '_NewWindow',
                                               (name,), {}) )
        for cmd in cmds:
            setattr(self, cmd,
                    new.instancemethod( Bind1st(self.runCommand, cmd),
                                        Embedded ) )

    def startRemote(cls):
        cls.remote = True
        cmd = ('%s %s RunFromEmbed' %
               (sys.executable, 
                os.path.join( os.path.dirname(os.path.abspath(__file__)),
                              'embed_remote.py' )))
        cls.to_pipe, cls.from_pipe = os.popen2(cmd)
        cls.cmdlen = len(struct.pack('Q', 0))
        atexit.register(cls.exitQt)
    startRemote = classmethod(startRemote)

    def sendCommand(kls, cmd):
        """Send the command to the remote process."""

        outs = cPickle.dumps(cmd)

        kls.to_pipe.write( struct.pack('Q', len(outs)) )
        kls.to_pipe.write( outs )
        kls.to_pipe.flush()

        backlen = kls.from_pipe.read(kls.cmdlen)
        rets = kls.from_pipe.read( struct.unpack('Q', backlen)[0] )
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

    def exitQt(cls):
        """Exit the Qt thread."""
        cls.sendCommand( (-1, '_Quit', (), {}) )
    exitQt = classmethod(exitQt)

