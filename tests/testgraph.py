#!/usr/bin/env python

# testgraph.py
# little test module

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
##############################################################################

# $Id$

import sys
import qt
from numarray import *

sys.path.append('./')

import widgets
import document

class PlotWidget(qt.QWidget):
    def __init__(self, *args):
        qt.QWidget.__init__(self, *args)
        self.setBackgroundMode( qt.Qt.PaletteBase )
 
        x=arange(50)*1+4.
        y = sin(x/10.*pi)*5 + 6
        z = cos(x/10.*pi)*5 + 6

        self.doc = document.Document()
        self.doc.setData( 'x', x, symerr=3 )
        self.doc.setData( 'y', y, symerr=0.5 )
        self.doc.setData( 'z', z, symerr=0.3 )

        reg = self.doc.getBaseWidget()

        #self.paxis1 = paxis.PAxis(reg)
        #self.paxis1.log()
        #self.paxis1.min = 0
        #self.paxis1.max = 10
        #self.paxis1.GridLines.hide = False
        #self.paxis1.AxisLabel.label = "Test label"
        #self.pregion.addChild( self.paxis1, name='x' )

        #self.paxis2 = paxis.PAxis()
        #self.paxis2.setOutputPositions(dirn=1)
        #self.paxis2.log()
        #self.paxis2.min = 0
        #self.paxis2.max = 10
        #self.paxis2.GridLines.hide = False
        #self.paxis2.AxisLabel.label = "This is a test (cm^{-1} s^{2})"
        #self.pregion.addChild( self.paxis2, name='y' )

        #self.paxis3 = paxis.PAxis()
        #self.paxis3.setOutputPositions(dirn=0, fother=1)
        #self.paxis3.min = 0
        #self.paxis3.max = 100
        #self.paxis3.reflect = True
        #self.pregion.addChild( self.paxis3, name='x2' )

        fnplot = widgets.FunctionPlotter( reg )
        fnplot.function = '(x/10.)**2'

        dplot = widgets.PointPlotter(reg, 'x', 'y')
        dplot.PlotMarker = 'O'
        dplot.MarkerSize = 5

        dplot2 = widgets.PointPlotter(reg, 'x', 'z', axis1='x')
        dplot2.PlotMarker = 'X'
        dplot2.MarkerSize = 5

    def paintEvent(self, ev):
        p = qt.QPainter(self)
        reg = self.doc.getBaseWidget()
        reg.draw( (150,150,550,550), p )

app=qt.QApplication(sys.argv)
w=PlotWidget(None)
app.setMainWidget(w)
w.show()
app.exec_loop()


