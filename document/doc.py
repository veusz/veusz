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

import widgets

def _cnvt_numarray(a):
    """Convert to a numarray if possible."""
    if a == None:
        return None
    elif type(a) != type(numarray.arange(1)):
        return numarray.array(a, type=numarray.Float64)
    else:
        return a.astype(numarray.Float64)

class Document:
    def __init__(self):
        """Initialise the document."""
        self.data = {}
        self.basewidget = widgets.Widget(None)
        self.basewidget.setDocument( self )

        self.modified_callbacks = []
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

    def addModifiedCallback(self, callback):
        """Register a callback function to be called if doc is modified."""
        self.modified_callbacks.append(callback)

    def setModified(self, ismodified=True):
        """Set the modified flag on the data, and inform views."""
        self.modified = ismodified
        for c in self.modified_callbacks:
            c(ismodified)

    def isModifed(self):
        """Return whether modified flag set."""
        return self.modified
    
