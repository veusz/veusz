#    Copyright (C) 2008 Jeremy S. Sanders
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

from __future__ import division
import sys
import struct
import socket

from .compat import citems, pickle
from .windows.simplewindow import SimpleWindow
from . import document
from . import setting
from . import qtall as qt

"""Program to be run by embedding interface to run Veusz commands."""

# embed.py module checks this is the same as its version number
API_VERSION = 2

class EmbeddedClient(object):
    """An object for each instance of embedded window with document."""

    def __init__(self, title, doc=None, hidden=False):
        """Construct window with title given."""

        self.window = SimpleWindow(title, doc=doc)
        if not hidden:
            self.window.show()
        self.document = self.window.document
        self.plot = self.window.plot
        # use time based checking by default
        self.plot.setTimeout(250)
        self.ci = document.CommandInterpreter(self.document)
        self.ci.addCommand('Close', self.cmdClose)
        self.ci.addCommand('Zoom', self.cmdZoom)
        self.ci.addCommand('EnableToolbar', self.cmdEnableToolbar)
        self.ci.addCommand('ForceUpdate', self.cmdForceUpdate)
        self.ci.addCommand('GetClick', self.cmdGetClick)
        self.ci.addCommand('ResizeWindow', self.cmdResizeWindow)
        self.ci.addCommand('SetUpdateInterval', self.cmdSetUpdateInterval)
        self.ci.addCommand('MoveToPage', self.cmdMoveToPage)
        self.ci.addCommand('IsClosed', self.cmdIsClosed)
        self.ci.addCommand('SetAntiAliasing', self.cmdSetAntiAliasing)
        self.ci.addCommand('Wipe', self.cmdWipe)
        self.ci.addCommand('_apiVersion', self.cmd_apiVersion)

        setting.transient_settings['unsafe_mode'] = True

        self.document.sigLog.connect(self.logEmitted)

    def logEmitted(self, msg):
        """Write anything logged to stderr."""
        sys.stderr.write(msg + '\n')

    def cmdClose(self):
        """Close()

        Close this window."""
        self.window.close()

        self.document = None
        self.window = None
        self.plot = None
        self.ci = None

    def cmdIsClosed(self):
        """IsClosed()

        Return whether window is still open."""
        return not self.window.isVisible()

    def cmd_apiVersion(self):
        """Get internal API version."""
        return API_VERSION

    def cmdZoom(self, zoom):
        """Zoom(zoom)

        Set the plot zoom level:
        This is a number to for the zoom from 1:1 or
        'page': zoom to page
        'width': zoom to fit width
        'height': zoom to fit height
        """
        self.window.setZoom(zoom)

    def cmdSetAntiAliasing(self, ison):
        """SetAntiAliasing(zoom)

        Enables or disables anti aliasing.
        """
        self.window.setAntiAliasing(ison)

    def cmdEnableToolbar(self, enable=True):
        """EnableToolbar(enable=True)

        Enable the toolbar in this plotwindow.
        if enable is False, disable it.
        """
        self.window.enableToolbar(enable)

    def cmdForceUpdate(self):
        """ForceUpdate()

        Forces an update of the plot window.
        """
        self.plot.actionForceUpdate()

    def cmdGetClick(self):
        """GetClick()

        Return a clicked point. The user can click a point on the graph

        This returns a list of tuples containing items for each axis in
        the clicked region:
         (axisname, valonaxis)
        where axisname is the full name of an axis
        valonaxis is value clicked along the axis

        [] is returned if no axes span the clicked region
        """
        return self.plot.getClick()

    def cmdResizeWindow(self, width, height):
        """ResizeWindow(width, height)

        Resize the window to be width x height pixels."""
        self.window.resize(width, height)

    def cmdSetUpdateInterval(self, interval):
        """SetUpdateInterval(interval)

        Set graph update interval.
        interval is in milliseconds (ms)
        set to zero to disable updates
        set to -1 to update when document changes
        default interval is 250ms
        """
        self.plot.setTimeout(interval)

    def cmdMoveToPage(self, pagenum):
        """MoveToPage(pagenum)

        Tell window to show specified pagenumber (starting from 1).
        """
        self.plot.setPageNumber(pagenum-1)

    def cmdWipe(self):
        """Wipe the current document."""
        self.document.wipe()

class EmbedApplication(qt.QApplication):
    """Application to run remote end of embed connection.

    Commands are sent over stdin, with responses sent to stdout
    """

    # lengths of lengths sent to application
    cmdlenlen = struct.calcsize('<I')

    def __init__(self, thesocket, args):
        qt.QApplication.__init__(self, args)
        self.socket = thesocket

        # listen to commands on the socket
        self.notifier = qt.QSocketNotifier(
            self.socket.fileno(), qt.QSocketNotifier.Read)
        self.notifier.activated.connect(self.slotDataToRead)
        self.notifier.setEnabled(True)

        # keep track of clients (separate veusz documents)
        self.clients = {}
        self.clientcounter = 0

    @staticmethod
    def readLenFromSocket(thesocket, length):
        """Read length bytes from socket."""
        s = b''
        while len(s) < length:
            s += thesocket.recv(length-len(s))
        return s

    @staticmethod
    def writeToSocket(thesocket, data):
        """Write to socket until all data written."""
        count = 0
        while count < len(data):
            count += thesocket.send(data[count:])

    @staticmethod
    def readCommand(thesocket):
        # get length of packet
        length = struct.unpack('<I', EmbedApplication.readLenFromSocket(
                thesocket, EmbedApplication.cmdlenlen))[0]
        # unpickle command and arguments
        temp = EmbedApplication.readLenFromSocket(thesocket, length)
        return pickle.loads(temp)

    def makeNewClient(self, title, doc=None, hidden=False):
        """Make a new client window."""
        client = EmbeddedClient(title, doc=doc, hidden=hidden)
        self.clients[self.clientcounter] = client
        # return new number and list of commands and docstrings
        retfuncs = []
        for name, cmd in citems(client.ci.cmds):
            retfuncs.append( (name, cmd.__doc__) )

        retval = self.clientcounter, retfuncs
        self.clientcounter += 1
        return retval

    def writeOutput(self, output):
        """Send output back to embed process."""
        # format return data
        outstr = pickle.dumps(output)

        # send return data to stdout
        self.writeToSocket( self.socket, struct.pack('<I', len(outstr)) )
        self.writeToSocket( self.socket, outstr )

    def finishRemote(self):
        """Clean up on exit."""
        self.notifier.setEnabled(False)
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except socket.error:
            pass
        self.closeAllWindows()
        self.quit()

    def slotDataToRead(self, socketfd):
        """Call routine to read data from remote socket."""
        try:
            self.readFromSocket()
        except socket.error:
            # exit if problem
            self.finishRemote()

    def readFromSocket(self):
        self.notifier.setEnabled(False)
        self.socket.setblocking(1)
        
        # unpickle command and arguments
        window, cmd, args, argsv = self.readCommand(self.socket)

        if cmd == '_NewWindow':
            retval = self.makeNewClient(args[0], hidden=argsv['hidden'])
        elif cmd == '_Quit':
            # exit client
            retval = None
        elif cmd == '_NewWindowCopy':
            # sets the document of this window to be the same as the
            # one specified
            retval = self.makeNewClient( args[0],
                                         doc=self.clients[args[1]].document,
                                         hidden=argsv['hidden'] )
        else:
            interpreter = self.clients[window].ci

            # window commands
            try:
                if cmd not in interpreter.cmds:
                    raise AttributeError("No Veusz command %s" % cmd)

                retval = interpreter.cmds[cmd](*args, **argsv)
            except Exception as e:
                retval = e

        self.writeOutput(retval)

        # do quit after if requested
        if cmd == '_Quit':
            self.finishRemote()
            return

        self.socket.setblocking(0)
        self.notifier.setEnabled(True)

def runremote():
    """Run remote end of embedding module."""
    # get connection parameters
    params = sys.stdin.readline().split()

    if params[0] == 'unix':
        # talk to existing unix domain socket
        listensocket = socket.fromfd( int(params[1]),
                                      socket.AF_UNIX,
                                      socket.SOCK_STREAM )

    elif params[0] == 'internet':
        # talk to internet port
        listensocket = socket.socket( socket.AF_INET,
                                      socket.SOCK_STREAM )
        listensocket.connect( (params[1], int(params[2])) )

    # get secret from stdin and send back to socket
    # this is a security check
    secret = sys.stdin.readline().encode('ascii')
    EmbedApplication.writeToSocket(listensocket, secret)

    # finally start listening application
    app = EmbedApplication(listensocket, [])
    app.setQuitOnLastWindowClosed(False)
    app.exec_()
