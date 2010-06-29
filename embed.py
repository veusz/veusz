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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
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
import socket
import random
import subprocess
import time

import embed_tree

# check remote process has this API version
API_VERSION = 1

def Bind1st(function, arg):
    """Bind the first argument of a given function to the given
    parameter."""

    def runner(*args, **args2):
        return function( arg, *args, **args2 )

    return runner

def findOnPath(cmd):
    """Find a command on the system path, or None if does not exist."""
    path = os.getenv('PATH', os.path.defpath)
    pathparts = path.split(os.path.pathsep)
    for dirname in pathparts:
        cmdtry = os.path.join(dirname, cmd)
        if os.path.isfile(cmdtry):
            return cmdtry
    return None

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

        # check API version is same
        try:
            remotever = self._apiVersion()
        except AttributeError:
            remotever = 0
        if remotever != API_VERSION:
            raise RuntimeError("Remote Veusz instance reports version %i of"
                               " API. This embed.py supports version %i." %
                               (remotever, API_VERSION))
        # define root object
        self.Root = embed_tree.WidgetNode(self, 'widget', '/')

    def StartSecondView(self, name = 'Veusz'):
        """Provides a second view onto the document of this window.

        Returns an Embedded instance
        """
        return Embedded(name=name, copyof=self)

    def WaitForClose(self):
        """Wait for the window to close."""

        # this is messy, polling for closure, but cleaner than doing
        # it in the remote client
        while not self.IsClosed():
            time.sleep(0.1)

    @classmethod
    def makeSockets(cls):
        """Make socket(s) to communicate with remote process.
        Returns string to send to remote process
        """

        if ( hasattr(socket, 'AF_UNIX') and hasattr(socket, 'socketpair') ):
            # convenient interface
            cls.sockfamily = socket.AF_UNIX
            sock, socket2 = socket.socketpair(cls.sockfamily,
                                              socket.SOCK_STREAM)
            sendtext = 'unix %i\n' % socket2.fileno()
            cls.socket2 = socket2
            waitaccept = False

        else:
            # otherwise mess around with internet sockets
            # * This is required for windows, which doesn't have AF_UNIX
            # * It is required where socketpair is not supported
            cls.sockfamily = socket.AF_INET
            sock = socket.socket(cls.sockfamily, socket.SOCK_STREAM)
            sock.bind( ('localhost', 0) )
            interface, port = sock.getsockname()
            sock.listen(1)
            sendtext = 'internet %s %i\n' % (interface, port)
            waitaccept = True

        return (sock, sendtext, waitaccept)

    @classmethod
    def makeRemoteProcess(cls):
        """Try to find veusz process for remote program."""
        
        # here's where to look for embed_remote.py
        thisdir = os.path.dirname(os.path.abspath(__file__))

        # build up a list of possible command lines to start the remote veusz
        if sys.platform == 'win32':
            # windows is a special case
            # we need to run embed_remote.py under pythonw.exe, not python.exe

            # look for the python windows interpreter on path
            findpython = findOnPath('pythonw.exe')
            if not findpython:
                # if it wasn't on the path, use sys.prefix instead
                findpython = os.path.join(sys.prefix, 'pythonw.exe')

            # look for veusz executable on path
            findexe = findOnPath('veusz.exe')
            if not findexe:
                try:
                    # add the usual place as a guess :-(
                    findexe = os.path.join(os.environ['ProgramFiles'],
                                           'Veusz', 'veusz.exe')
                except KeyError:
                    pass

            # here is the list of commands to try
            possiblecommands = [
                [findpython, os.path.join(thisdir, 'embed_remote.py')],
                [findexe] ]

        else:
            # try embed_remote.py in this directory, veusz in this directory
            # or veusz on the path in order
            possiblecommands = [ [sys.executable,
                                  os.path.join(thisdir, 'embed_remote.py')],
                                 [os.path.join(thisdir, 'veusz')],
                                 [findOnPath('veusz')] ]

        # cheat and look for Veusz app for MacOS under the standard application
        # directory. I don't know how else to find it :-(
        if sys.platform == 'darwin':
            findbundle = findOnPath('Veusz.app')
            if findbundle:
                possiblecommands += [ [findbundle+'/Contents/MacOS/Veusz'] ]
            else:
                possiblecommands += [[
                    '/Applications/Veusz.app/Contents/MacOS/Veusz' ]]

        for cmd in possiblecommands:
            # only try to run commands that exist as error handling
            # does not work well when interfacing with OS (especially Windows)
            if ( None not in cmd and
                 False not in [os.path.isfile(c) for c in cmd] ):
                try:
                    # we don't use stdout below, but works around windows bug
                    # http://bugs.python.org/issue1124861
                    cls.remote = subprocess.Popen(cmd + ['--embed-remote'],
                                                  shell=False, bufsize=0,
                                                  close_fds=False,
                                                  stdin=subprocess.PIPE,
                                                  stdout=subprocess.PIPE)
                    return
                except OSError:
                    pass

        raise RuntimeError('Unable to find a veusz executable on system path')

    @classmethod
    def startRemote(cls):
        """Start remote process."""
        cls.serv_socket, sendtext, waitaccept = cls.makeSockets()

        cls.makeRemoteProcess()
        stdin = cls.remote.stdin

        # send socket number over pipe
        stdin.write( sendtext )

        # accept connection if necessary
        if waitaccept:
            cls.serv_socket, address = cls.serv_socket.accept()

        # send a secret to the remote program by secure route and
        # check it comes back
        # this is to check that no program has secretly connected
        # on our port, which isn't really useful for AF_UNIX sockets
        secret = ''.join([random.choice('ABCDEFGHUJKLMNOPQRSTUVWXYZ'
                                        'abcdefghijklmnopqrstuvwxyz'
                                        '0123456789')
                          for i in xrange(16)]) + '\n'
        stdin.write(secret)
        secretback = cls.readLenFromSocket(cls.serv_socket, len(secret))
        assert secret == secretback

        # packet length for command bytes
        cls.cmdlen = struct.calcsize('<I')
        atexit.register(cls.exitQt)

    @staticmethod
    def readLenFromSocket(socket, length):
        """Read length bytes from socket."""
        s = ''
        while len(s) < length:
            s += socket.recv(length-len(s))
        return s

    @staticmethod
    def writeToSocket(socket, data):
        count = 0
        while count < len(data):
            count += socket.send(data[count:])

    @classmethod
    def sendCommand(cls, cmd):
        """Send the command to the remote process."""

        outs = cPickle.dumps(cmd)

        cls.writeToSocket( cls.serv_socket, struct.pack('<I', len(outs)) )
        cls.writeToSocket( cls.serv_socket, outs )

        backlen = struct.unpack('<I', cls.readLenFromSocket(cls.serv_socket,
                                                            cls.cmdlen))[0]
        rets = cls.readLenFromSocket( cls.serv_socket, backlen )
        retobj = cPickle.loads(rets)
        if isinstance(retobj, Exception):
            raise retobj
        else:
            return retobj

    def runCommand(self, cmd, *args, **args2):
        """Execute the given function in the Qt thread with the arguments
        given."""
        return self.sendCommand( (self.winno, cmd, args[1:], args2) )

    @classmethod
    def exitQt(cls):
        """Exit the Qt thread."""
        cls.sendCommand( (-1, '_Quit', (), {}) )
        cls.serv_socket.shutdown(socket.SHUT_RDWR)
        cls.serv_socket.close()
        cls.serv_socket, cls.from_pipe = -1, -1
