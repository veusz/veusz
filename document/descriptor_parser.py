# descriptor_parser.py
# some routines to decode the descriptor into each part

#    Copyright (C) 2003 Jeremy S. Sanders
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
###############################################################################

"""Routines for reading data from strings or files.

Module supplies two classes of object: 

DataStream which sets up an input stream for reading data from.
ReadDescriptor which reads data from a data stream using a descriptor."""

import string
import utils
import data
import sys
import numarray.ieeespecial

# parser builds up array to decode input data
# separate routine for decoding data stream into numbers

# datastream consists of a set of numbers or line endings or strings

class _EndOfFile(Exception):
    """Thrown if DataStream gets to the end of a file."""
    pass

class DataStream:
    """This class takes a file, and provides a readable stream.

    It takes account of line continuation characters (\)
    and removes comments. Newlines are also returned by the read
    function.
    """

    def __init__(self, file):
        self.file = file
        self.buffer = []
        self.oldbufsize = 5
        self.oldbuf = []

    def readBuffer(self):
        """Return the next item from the buffer."""
        v = self.buffer.pop(0)
        self.oldbuf = self.oldbuf[-1 * self.oldbufsize:] + [v]
        return v

    def read(self):
        """Get the next value in the stream."""

        # if we have anything in the buffer return it
        if len(self.buffer) != 0:
            return self.readBuffer()

        # else get the next line, and get the values in it
        while 1:
            line = self.file.readline()

            # we throw an exception if we can't read more data
            # and there's nothing in the buffer
            # (this assumes something will eventually empty the buffer so we
            #  can really throw an exception - otherwise we'd lose data)
            if line == '' and len(self.buffer) == 0:
                raise _EndOfFile()
                
            items = string.split(line)

            # set if we need to get the next line too (without a newline)
            docontinue = 0

            for i in items:
                if i[0] == '#' or i[0] == '!':
                    if i == items[0]: # read in next line if this only a comment
                        docontinue = 1
                    break
                if i[0] == '\\':
                    docontinue = 1
                    break
                docontinue = ( i[-1] == '\\' )
                if docontinue:
                    i.pop()
                self.buffer.append( i )

            # get another line!
            if docontinue:
                continue

            # tell the parser there's a line ending
            self.buffer.append( '\n' )
            break
                
        # return first item in buffer
        return self.readBuffer()

    def push(self):
        """Put back an item on the buffer."""
        if len(self.oldbuf) == 0:
            v = '\n'
        else:
            v = self.oldbuf.pop()

        self.buffer = [ v ] + self.buffer

    def isNextStartLine(self):
        """Is the next item the first on a line?."""
        return ( len(self.oldbuf) == 0 or self.oldbuf[-1] == '\n' )
            
    def moveToStartLine(self):
        """Position the stream to the start of a line (remaining there if necessary."""

        # repeat until the next character is at the start of a line
        while not self.isNextStartLine():
            tmp = self.read()

        # eliminate multiple blank lines
        while self.read() == '\n':
            pass
        self.push()

    def stop(self):
        """Stop processing by signalling end of file."""
        raise _EndOfFile()

class _DescrItems:
    """Descriptor item. Base class for parts of descriptor which are executed."""

    def __init__(self, text):
        """Initialise class."""
        pass

    def execute(self, stream, descriptor):
        """Execute item, taking input from stream."""
        
        # return values:
        #  1 - okay, used value
        # -1 - nasty error occured
        #  0 - couldn't use value
        return 0
    
class _DescrItemsData(_DescrItems):
    """Descriptor item for reading data from a stream."""
    
    def __init__(self, text):
        """Initialise class. Parameters to class are given as text."""
        self.text = text
        self.count = 0

    def execute(self, stream, descriptor):
        """Read data from stream."""
        
        # they can change each time
        evaled = data.data.expand_expression( self.text )

        val = stream.read()

        try:
            data.data.add_item( evaled, float(val) )
        except ValueError:     # can't convert to float
            # provide a lovely NaN
            data.data.add_item( evaled, numarray.ieeespecial.nan )
            stream.push()
            return 0
        except AttributeError: # can't append
            # FIXME - handle errors better
            print "Could not append value to variable '" + evaled + \
                  "'. Ignored."
            stream.push()
            return -1

        # keep count of number of items read
        if not descriptor.counts.has_key( evaled ):
            descriptor.counts[evaled] = 0
            
        descriptor.counts[evaled] += 1

class _DescrItemsStop(_DescrItems):
    """Descriptor just causes reading to stop here."""

    def __init__(self, text):
        pass
    
    def execute(self, stream, descriptor):
        stream.stop()

class _DescrItemsEol(_DescrItems):
    """Descriptor moves to the start of a line (if necessary)."""

    def __init__(self, text):
        pass

    def execute(self, stream, descriptor):
        stream.moveToStartLine()

class _DescrItemsFor(_DescrItems):
    """Descriptor loops over columns, lines or blocks."""
    
    fortypes = {'cols':0 , 'lines':1, 'blocks': 2}
    loopnumber = 0

    def __init__(self, text):
        """Initialise loop descriptor."""
        parts = utils.split_on_colons(text)
        self.fortype = _DescrItemsFor.fortypes['cols']

        self.var = '_for_' + str(_DescrItemsFor.loopnumber)
        _DescrItemsFor.loopnumber += 1

        self.var_start = '1'
        self.var_end = '10000000'
        self.block_end = ''

        parts.pop(0)

        # get type of for
        if len(parts) != 0:
            try:
                self.fortype = _DescrItemsFor.fortypes[ parts[0] ]
            except KeyError:
                print "For type '" + parts[0] + "' not valid. Ignored."
                return
            parts.pop(0)

        # get blockend if a block loop
        if self.fortype == _DescrItemsFor.fortypes['blocks'] and len(parts) != 0:
            self.block_end = parts[0]
            parts.pop(0)

        # get variable name
        if len(parts) != 0:
            try:
                data.data.set_var( parts[0], -1 )
            except:
                print "Failed to initialise variable '" + parts[0] + \
                      "'. Using default variable name."
            else:
                self.var = parts[0]
                parts.pop(0)

        # get start index
        if len(parts) != 0:
            self.var_start = parts[0]
            parts.pop(0)

        # get end index
        if len(parts) != 0:
            self.var_end = parts[0]
            parts.pop(0)

    def execute(self, stream, descriptor):
        """Start looping."""
        
        try:
            start = int( data.data.expand_expression( self.var_start ))
        except:
            print "Unable to interpret starting loop value '%s'. Ignored." % \
                  self.var_start
            return -1

        data.data.set_var( self.var, start )
        self.returnpos = descriptor.index
        descriptor.forstack.append( self )
        return 0

class _DescrItemsEnd(_DescrItems):
    """Descriptor for end of loop."""

    def __init__(self, text):
        pass

    def execute(self, stream, descriptor):
        """Jump back to the start of the loop if requested."""

        if len(descriptor.forstack) == 0:
            print "Unable to execute 'end' as not in loop. Ignored."
            return -1
        topitem = descriptor.forstack[-1]

        if topitem.fortype == _DescrItemsFor.fortypes['cols']:
            nextread = stream.read()
            # for looping over columns
            if nextread == '\n':  # got to the end of the line
                # remove self from stack of loops
                descriptor.forstack.pop()
            else:
                stream.push() # put back data
                
                # increment counter
                counter = data.data.get_var( topitem.var ) + 1
                data.data.set_var( topitem.var, counter )

                try:
                    stop = int( data.data.expand_expression( topitem.var_end ))
                except:
                    print "Unable to interpret stopping loop value '%s'." + \
                          " Ignored." % topitem.var_end
                    return -1

                if counter > stop:
                    # remove self from loop stack
                    descriptor.forstack.pop()
                else:
                    # jump back to start of loop (execution will go to next item)
                    descriptor.index = topitem.returnpos

        elif topitem.fortype == _DescrItemsFor.fortypes['lines']:
            # ignore until end of line
            stream.moveToStartLine()

            # increment counter
            counter = data.data.get_var(topitem.var) + 1
            data.data.set_var( topitem.var, counter)
            
            try:
                stop = int( data.data.expand_expression( topitem.var_end ) )
            except:
                print "Unable to interpret stopping loop value '%s'." + \
                      " Ignored." % topitem.var_end
                return -1
            
            if counter > stop:
                # remove self from loop stack
                descriptor.forstack.pop()
            else:
                # jump back to start of loop (execution will go to next item)
                descriptor.index = topitem.returnpos

        elif topitem.fortype == _DescrItemsFor.fortypes['block']:
            # FIXME - implement
            pass

class ReadDescriptor:
    """Class for reading data from a datastream with a descriptor."""

    def __init__(self):
        """Initialise object."""
        self.forstack = []   # hold looping info
        self.items = []      # the things to read
        self.index = 0       # where we are in the list of things
        self.counts = {}     # count how many items are read

    def parse(self, text):
        """Parse descriptor string."""
        for item in string.split(text):
            if item[:3] == 'for':
                self.items.append( _DescrItemsFor(item) )
            elif item == 'end':
                self.items.append( _DescrItemsEnd(item) )
            elif item == 'stop':
                self.items.append( _DescrItemsStop(item) )
            elif item == 'eol':
                self.items.append( _DescrItemsEol(item) )
            else:
                self.items.append( _DescrItemsData(item) )

    def read(self, datastream):
        """Read from the datastream with the descriptor."""
        
        # we loop for ever until an EOF exception is thrown
        # is this good practice?
        try:
            while 1:
                # execute item
                self.items[self.index].execute( datastream, self )
            
                # go to next position (wrapping if necessary)
                self.index += 1
                if self.index >= len(self.items):
                    self.index = 0
                    datastream.moveToStartLine()
                    
        except _EndOfFile:
            # situation normal
            pass

    def report(self):
        """Report after reading what data we read."""
        
        vars = self.counts.keys()
        vars.sort()
        print "Items read:"

        count = 0
        for i in vars:
            sys.stdout.write('%7s: %-5i ' % (i, self.counts[i]) )
            count += 1
            if count % 5 == 0:
                sys.stdout.write('\n')
        sys.stdout.write('\n')


# a test
#f = open('test.txt', 'r')
#ds = DataStream(f)
#rd = ReadDescriptor()
#rd.parse('for:lines:i:1:2 for:cols x_$i$ end end for:lines:j:1:2 y end stop')
#rd.read(ds)

#print data.x_1
#print data.x_2
#print data.y
#print data.z
#print data.a
#print data.b
#print data.c

