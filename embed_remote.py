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

# $Id$

import sys
import struct
import cPickle
import socket

import veusz.qtall as qt4
from veusz.windows.simplewindow import SimpleWindow
import veusz.document as document

"""Program to be run by embedding interface to run Veusz commands."""

class EmbeddedClient(object):

    def __init__(self, title, doc=None):
        self.window = SimpleWindow(title, doc=doc)
        self.window.show()
        self.document = self.window.document
        self.plot = self.window.plot
        self.ci = document.CommandInterpreter(self.document)
        self.ci.addCommand('Close', self.cmdClose)
        self.ci.addCommand('Zoom', self.cmdZoom)
        self.ci.addCommand('EnableToolbar', self.cmdEnableToolbar)
        self.ci.addCommand('GetClick', self.cmdGetClick)
        self.ci.addCommand('ResizeWindow', self.cmdResizeWindow)
        self.ci.addCommand('SetUpdateInterval', self.cmdSetUpdateInterval)
        self.ci.addCommand('MoveToPage', self.cmdMoveToPage)

    def cmdClose(self):
        """Close()

        Close this window."""
        self.window.close()

        self.document = None
        self.window = None
        self.plot = None
        self.ci = None

    def cmdZoom(self, zoom):
        """Zoom(zoom)

        Set the plot zoom level:
        This is a number to for the zoom from 1:1 or
        'page': zoom to page
        'width': zoom to fit width
        'height': zoom to fit height
        """
        self.window.setZoom(zoom)

    def cmdEnableToolbar(self, enable=True):
        """EnableToolbar(enable=True)

        Enable the toolbar in this plotwindow.
        if enable is False, disable it.
        """
        self.window.enableToolbar(enable)

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
        """
        self.plot.setTimeout(interval)

    def cmdMoveToPage(self, pagenum):
        """MoveToPage(pagenum)

        Tell window to show specified pagenumber (starting from 1).
        """
        self.plot.setPageNumber(pagenum-1)

class EmbedApplication(qt4.QApplication):
    """Application to run remote end of embed connection.

    Commands are sent over stdin, with responses sent to stdout
    """

    # lengths of lengths sent to application
    cmdlenlen = len(struct.pack('L', 0))

    def __init__(self, socket, args):
        qt4.QApplication.__init__(self, args)
        self.socket = socket

        self.notifier = qt4.QSocketNotifier(self.socket.fileno(),
                                            qt4.QSocketNotifier.Read)
        self.connect(self.notifier, qt4.SIGNAL('activated(int)'),
                     self.slotDataToRead)

        self.clients = {}
        self.clientcounter = 0

    def readLenFromSocket(thesocket, length):
        """Read length bytes from socket."""
        s = ''
        while len(s) < length:
            s += thesocket.recv(length-len(s))
        return s
    readLenFromSocket = staticmethod(readLenFromSocket)

    def writeToSocket(thesocket, data):
        count = 0
        while count < len(data):
            count += thesocket.send(data[count:])
    writeToSocket = staticmethod(writeToSocket)

    def readCommand(socket):
        # get length of packet
        length = struct.unpack('L', EmbedApplication.readLenFromSocket(
                socket, EmbedApplication.cmdlenlen))[0]
        # unpickle command and arguments
        return cPickle.loads(
            EmbedApplication.readLenFromSocket(socket, length))
    readCommand = staticmethod(readCommand)

    def makeNewClient(self, title, doc=None):
        """Make a new client window."""
        client = EmbeddedClient(title, doc=doc)
        self.clients[self.clientcounter] = client
        # return new number and list of commands and docstrings
        retfuncs = []
        for name, cmd in client.ci.cmds.iteritems():
            retfuncs.append( (name, cmd.__doc__) )

        retval = self.clientcounter, retfuncs
        self.clientcounter += 1
        return retval

    def slotDataToRead(self, socketfd):
        self.notifier.setEnabled(False)
        self.socket.setblocking(1)
        
        # unpickle command and arguments
        window, cmd, args, argsv = self.readCommand(self.socket)

        doquit = False
        if cmd == '_NewWindow':
            retval = self.makeNewClient(args[0])
        elif cmd == '_Quit':
            # exits client
            doquit = True
            retval = None
        elif cmd == '_NewWindowCopy':
            # sets the document of this window to be the same as the
            # one specified
            retval = self.makeNewClient( args[0],
                                         doc=self.clients[args[1]].document )
        else:
            interpreter = self.clients[window].ci

            # window commands
            try:
                if cmd not in interpreter.cmds:
                    raise AttributeError, "No Veusz command %s" % cmd

                retval = interpreter.cmds[cmd](*args, **argsv)
            except Exception, e:
                retval = e
        
        # format return data
        outstr = cPickle.dumps(retval)

        # send return data to stdout
        self.writeToSocket( self.socket, struct.pack('L', len(outstr)) )
        self.writeToSocket( self.socket, outstr )

        # do quit after if requested
        if doquit:
            self.socket.shutdown(socket.SHUT_RDWR)
            self.closeAllWindows()
            self.quit()

        self.socket.setblocking(0)
        self.notifier.setEnabled(True)

def main():
    if len(sys.argv) != 2 or sys.argv[1] != 'RunFromEmbed':
        print >>sys.stderr, ("This program must be run from "
                             "the Veusz embedding module")
        sys.exit(1)

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
    secret = sys.stdin.readline()
    EmbedApplication.writeToSocket(listensocket, secret)

    # finally start listening application
    app = EmbedApplication(listensocket, [])
    app.setQuitOnLastWindowClosed(False)
    app.exec_()

if __name__ == '__main__':
    main()
