# document.py
# A module to handle documents

#    Copyright (C) 2004 Jeremy S. Sanders
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

import string
import numarray
import qt

import widgets
import utils

def _cnvt_numarray(a):
    """Convert to a numarray if possible."""
    if a == None:
        return None
    elif type(a) != type(numarray.arange(1)):
        return numarray.array(a, type=numarray.Float64)
    else:
        return a.astype(numarray.Float64)

class Document( qt.QObject ):
    """Document class for holding the graph data.

    Emits: sigModified when the document has been modified
           sigResize when the document size has changed.
    """

    def __init__(self):
        """Initialise the document."""
        qt.QObject.__init__( self )

        self.prefs = utils.Preferences( 'Document', self )
        self.prefs.addPref('size', 'list', ['20cm', '20cm'])
        self.prefs.read()

        self.data = {}
        self.basewidget = widgets.Widget(None)
        self.basewidget.setDocument( self )

        self.setModified()

    def setData(self, name, val, symerr = None, negerr = None, poserr = None):
        """Set data to val, with symmetric or negative and positive errors."""
        t = type(numarray.arange(1))

        self.data[name] = _cnvt_numarray(val)
        self.data[name + '_SYMERR_'] = _cnvt_numarray(symerr)
        self.data[name + '_NEGERR_'] = _cnvt_numarray(negerr)
        self.data[name + '_POSERR_'] = _cnvt_numarray(poserr)

        self.setModified()

    def getBaseWidget(self):
        """Return the base widget."""
        return self.basewidget

    def getData(self, name):
        """Get data with name."""
        return self.data[name]

    def getSymErr(self, name):
        """Get data symmetric error."""
        return self.data[name + '_SYMERR_']

    def getNegErr(self, name):
        """Get data negative error."""
        return self.data[name + '_NEGERR_']

    def getPosErr(self, name):
        """Get data positive error."""
        return self.data[name + '_POSERR_']

    def getDataAll(self, name):
        """Get (val, sym, neg, pos) data."""
        return ( self.data[name], self.data[name + '_SYMERR_'],
                 self.data[name + '_NEGERR_'],
                 self.data[name + '_POSERR_'] )

    def setModified(self, ismodified=True):
        """Set the modified flag on the data, and inform views."""
        self.modified = ismodified
        self.emit( qt.PYSIGNAL("sigModified"), ( ismodified, ) )

    def setSize(self, width, height):
        """Set the document size."""
        self.size = [ width, height ]
        self.emit( qt.PYSIGNAL("sigResize"), self.size )

    def getSize(self):
        """Get the document size."""
        return tuple(self.size)

    def isModifed(self):
        """Return whether modified flag set."""
        return self.modified
    
    def printTo(self, printer):
        """Print onto printing device."""
        metrics = qt.QPaintDeviceMetrics( printer )
        
        painter = qt.QPainter()
        painter.begin( printer )

        # work out how many pixels correspond to the given size
        width, height = utils.cnvtDists( self.size, painter )
        self.basewidget.draw( (0, 0, width, height), painter )
        painter.end()
