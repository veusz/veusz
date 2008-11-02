#!/usr/bin/env python

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

    def __init__(self, title):
        self.window = SimpleWindow(title)
        self.window.show()
        self.document = self.window.document
        self.plot = self.window.plot
        self.ci = document.CommandInterpreter(self.document)
        self.ci.addCommand('Close', self.cmdClose)
        self.ci.addCommand('Zoom', self.cmdZoom)
        self.ci.addCommand('EnableToolbar', self.cmdEnableToolbar)
        self.ci.addCommand('GetClick', self.cmdGetClick)

    def cmdClose(self):
        """Close this window."""
        self.window.close()

        self.document = None
        self.window = None
        self.plot = None
        self.ci = None

    def cmdZoom(self, zoomfactor):
        """Set the plot zoom factor."""
        self.plot.setZoomFactor(zoomfactor)

    def cmdEnableToolbar(self, enable=True):
        """Enable the toolbar in this plotwindow."""
        self.window.enableToolbar(enable)

    def cmdGetClick(self):
        """Return a clicked point."""
        return self.plot.getClick()

class EmbedApplication(Application):
    """Application to run remote end of embed connection.

    Commands are sent over stdin, with responses sent to stdout
    """

    # lengths of lengths sent to application
    cmdlenlen = len(struct.pack('Q', 0))

    def __init__(self, args):
        Application.__init__(self, args)
        self.notifier = qt4.QSocketNotifier(sys.stdin.fileno(),
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
        length = struct.unpack('Q', EmbedApplication.readLenFromSocket(
                socket, EmbedApplication.cmdlenlen))[0]
        # unpickle command and arguments
        return cPickle.loads(
            EmbedApplication.readLenFromSocket(socket, length))
    readCommand = staticmethod(readCommand)

    def slotDataToRead(self, socket):
        self.notifier.setEnabled(False)
        
        # unpickle command and arguments
        window, cmd, args, argsv = self.readCommand(socket)

        if cmd == '_NewWindow':
            # create new client
            client = EmbeddedClient(args[0])
            self.clients[self.clientcounter] = client
            # return new number and list of commands
            retval = self.clientcounter, client.ci.cmds.keys()
            self.clientcounter += 1
        elif cmd == '_Quit':
            # exits client
            self.closeAllWindows()
            self.quit()
            retval = None
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
        sys.stdout.write( struct.pack('Q', len(outstr)) )
        sys.stdout.write( outstr )
        sys.stdout.flush()

        self.notifier.setEnabled(True)

def main():
    if len(sys.argv) < 2 or sys.argv[1] != 'RunFromEmbed':
        print >>sys.stderr, ("This program must be run from "
                             "the Veusz embedding module")
        sys.exit(1)

    app = EmbedApplication(sys.argv)

    # stop app exiting if no windows left
    hiddenwin = qt4.QWidget()
    hiddenwin.hide()

    app.exec_()

if __name__ == '__main__':
    main()
