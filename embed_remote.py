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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id$

import sys
import os
import struct
import cPickle

import veusz.qtall as qt4
from veusz.application import Application
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
        if zoom == 'page':
            self.plot.slotViewZoomPage()
        elif zoom == 'width':
            self.plot.slotViewZoomWidth()
        elif zoom == 'height':
            self.plot.slotViewZoomHeight()
        else:
            self.plot.setZoomFactor(zoom)

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

class EmbedApplication(Application):
    """Application to run remote end of embed connection.

    Commands are sent over stdin, with responses sent to stdout
    """

    # lengths of lengths sent to application
    cmdlenlen = len(struct.pack('L', 0))

    def __init__(self, to_pipe, from_pipe, args):
        Application.__init__(self, args)
        self.to_pipe = to_pipe
        self.from_pipe = from_pipe

        self.notifier = qt4.QSocketNotifier(self.from_pipe,
                                            qt4.QSocketNotifier.Read)
        self.connect(self.notifier, qt4.SIGNAL('activated(int)'),
                     self.slotDataToRead)

        self.clients = {}
        self.clientcounter = 0

    def readLenFromSocket(socket, length):
        """Read length bytes from socket."""
        s = ''
        while len(s) < length:
            s += os.read(socket, length-len(s))
        return s
    readLenFromSocket = staticmethod(readLenFromSocket)

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

    def slotDataToRead(self, socket):
        self.notifier.setEnabled(False)
        
        # unpickle command and arguments
        window, cmd, args, argsv = self.readCommand(socket)

        if cmd == '_NewWindow':
            retval = self.makeNewClient(args[0])
        elif cmd == '_Quit':
            # exits client
            self.closeAllWindows()
            self.quit()
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
        os.write( self.to_pipe,  struct.pack('L', len(outstr)) )
        os.write( self.to_pipe, outstr )

        self.notifier.setEnabled(True)

def main():
    if len(sys.argv) < 4 or sys.argv[1] != 'RunFromEmbed':
        print >>sys.stderr, ("This program must be run from "
                             "the Veusz embedding module")
        sys.exit(1)

    app = EmbedApplication(int(sys.argv[2]), int(sys.argv[3]), [])
    app.setQuitOnLastWindowClosed(False)
    app.exec_()

if __name__ == '__main__':
    main()
