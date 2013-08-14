#    Copyright (C) 2009 Jeremy S. Sanders
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
import select
import subprocess
import os
import socket
import platform
import signal
import locale

from ..compat import cbasestr, cstr
from .. import qtall as qt4
from .. import utils
from . import simpleread

class CaptureFinishException(Exception):
    """An exception to say when a stream has been finished."""

class CaptureStream(simpleread.Stream):
    """A special stream for capturing data."""

    def __init__(self):
        """Initialise the stream."""

        simpleread.Stream.__init__(self)
        self.buffer = ''
        self.continuousreads = 0
        self.bytesread = 0
        self.linesread = 0
        self.maxlines = None
        self.timedout = False

    def _setTimeout(self, timeout):
        """Setter for setting timeout property."""
        if timeout:
            self.timer = qt4.QTimer.singleShot(timeout*1000,
                                               self._timedOut)
    timeout = property(None, _setTimeout, None,
                       "Time interval to stop in (seconds) or None")
            
    def _timedOut(self):
        self.timedout = True

    def getMoreData(self):
        """Override this to return more data from the source without
        blocking."""
        return ''

    def readLine(self):
        """Return a new line of data.

        Either returns new line or
        Raises StopIteration if there is no data, or more than 100 lines
        have been read."""

        while True:
            # we've reached the limit of lines or a timeout has occurred
            if self.linesread == self.maxlines:
                raise CaptureFinishException("Maximum number of lines read")
            if self.timedout:
                raise CaptureFinishException("Maximum time period occurred")

            # stop reading continous data greater than this many lines
            if self.continuousreads == 100:
                self.continuousreads = 0
                raise StopIteration

            index = self.buffer.find('\n')
            if index >= 0:
                # is there a line in the buffer?
                retn = self.buffer[:index]
                self.buffer = self.buffer[index+1:]
                self.linesread += 1
                self.continuousreads += 1
                return retn
            else:
                # if not, then read some more data
                data = self.getMoreData()

                if not data:
                    self.continuousreads = 0
                    raise StopIteration
                self.bytesread += len(data)
                self.buffer += data

    def close(self):
        """Close any allocated object."""
        pass

class FileCaptureStream(CaptureStream):
    """Capture from a file or named pipe."""

    def __init__(self, filename):
        CaptureStream.__init__(self)

        # open file
        self.fileobj = open(filename, 'rU')

        # make new thread to read file
        self.readerthread = utils.NonBlockingReaderThread(self.fileobj)
        self.readerthread.start()

        self.name = filename

    def getMoreData(self):
        """Read data from the file."""
        try:
            data, done = self.readerthread.getNewData()
            if len(data) == 0 and done:
                raise CaptureFinishException("End of file")
            return data
        except OSError as e:
            raise CaptureFinishException("OSError: %s" % cstr(e))

    def close(self):
        """Close file."""
        self.fileobj.close()

class CommandCaptureStream(CaptureStream):
    """Capture from an external program."""

    def __init__(self, commandline):
        """Capture from commandline - this is passed to the shell."""
        CaptureStream.__init__(self)

        self.name = commandline
        self.popen = subprocess.Popen(commandline, shell=True,
                                      bufsize=0, stdout=subprocess.PIPE,
                                      universal_newlines=True)

        # make new thread to read stdout
        self.readerthread = utils.NonBlockingReaderThread(self.popen.stdout)
        self.readerthread.start()

    def getMoreData(self):
        """Read data from the command."""

        retn, done = self.readerthread.getNewData()

        if not retn:
            poll = self.popen.poll()
            if poll is not None:
                # process has ended
                raise CaptureFinishException("Process ended (status code %i)" %
                                             poll)
        return retn

    def close(self):
        """Close file."""

        if self.popen.poll() is None:
            # need to kill process if it is still running
            if platform.system() == 'Windows':
                # awful code (for windows)
                # use this to not have a ctypes dependency
                os.system('TASKKILL /PID %i /F' % self.popen.pid)
            else:
                # unix
                os.kill(self.popen.pid, signal.SIGTERM)

        try:
            self.popen.stdout.close()
        except EnvironmentError:
            # problems closing stdout for some reason
            pass

class SocketCaptureStream(CaptureStream):
    """Capture from an internet host."""

    def __init__(self, host, port):
        """Connect to host and port specified."""
        CaptureStream.__init__(self)

        self.name = '%s:%i' % (host, port)
        try:
            self.socket = socket.socket( socket.AF_INET,
                                          socket.SOCK_STREAM )
            self.socket.connect( (host, port) )
        except socket.error as e:
            self._handleSocketError(e)

    def _handleSocketError(self, e):
        """Special function to reraise exceptions
        because socket exceptions have changed in python 2.6 and
        behave differently on some platforms.
        """

        # clean up
        self.socket.close()

        # re-raise
        raise e

    def getMoreData(self):
        """Read data from the socket."""
        
        # see whether there is data to be read
        i, o, e = select.select([self.socket], [], [], 0)
        if i:
            try:
                retn = self.socket.recv(1024)
            except socket.error as e:
                self._handleSocketError(e)
            if len(retn) == 0:
                raise CaptureFinishException("Remote socket closed")
            return retn.decode('utf-8', errors='ignore')
        else:
            return ''

    def close(self):
        """Close the socket."""
        self.socket.close()
