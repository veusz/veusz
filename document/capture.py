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

# $Id$

import fcntl
import errno
import select
import subprocess
import os

import veusz.qtall as qt4
import simpleread

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

        # open file without any blocking
        self.fd = os.open(filename, os.O_RDONLY | os.O_NDELAY)
        self.name = filename

    def getMoreData(self):
        """Read data from the file."""
        try:
            return os.read(self.fd, 1024)
        except OSError, e:
            if e.errno == errno.EAGAIN:
                # no data available
                return ''
            else:
                # raise exception to be caught above
                raise CaptureFinishException("OSError: %s" % unicode(e))

    def close(self):
        """Close file."""
        os.close(self.fd)

class CommandCaptureStream(CaptureStream):
    """Capture from an external program."""

    def __init__(self, commandline):
        CaptureStream.__init__(self)

        self.name = commandline
        self.popen = subprocess.Popen(commandline, shell=True,
                                      bufsize=0, stdout=subprocess.PIPE)

    def getMoreData(self):
        """Read data from the command."""

        i, o, e = select.select([self.popen.stdout.fileno()], [], [], 0)
        if i:
            retn = os.read(i[0], 1024)
        else:
            retn = ''

        if not retn:
            poll = self.popen.poll()
            if poll is not None:
                # process has ended
                raise CaptureFinishException("Process ended (status code %i)" %
                                             poll)
        return retn

    def close(self):
        """Close file."""
        self.popen.stdout.close()


