# axis.py
# package to handle an axis, and the conversion of data -> coordinates

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
import numarray

import widget
import axisticks
import widgetfactory

import utils

class Axis(widget.Widget):
    """Manages and draws an axis."""

    typename = 'axis'

    def __init__(self, parent, name=None):
        """Initialise axis."""

        widget.Widget.__init__(self, parent, name=name)

        self.addPref( 'numTicks', 'int', 5 )
        self.addPref( 'numMinorTicks', 'int', 40 )
        self.addPref( 'autoExtend', 'int', True )
        self.addPref( 'autoExtendZero', 'int', True )
        self.addPref( 'autoMirror', 'int', True )
        self.addPref( 'min', 'double', None ) # automatic
        self.addPref( 'max', 'double', None ) # automatic
        self.addPref( 'reflect', 'int', False )
        self.addPref( 'log', 'int', False )
        self.readPrefs()

        self.setOutputPositions()

        self.minorticks = None  # automatic
        self.majorticks = None  # automatic

        # create sub-preferences
        self.Line = utils.PreferencesLine( 'Line' )
        self.MajorTicks = utils.PreferencesMajorTick( 'AxisMajorTick' )
        self.MinorTicks = utils.PreferencesMinorTick( 'AxisMinorTick' )
        self.TickLabels = utils.PreferencesTickLabel( 'TickLabels' )
        self.GridLines = utils.PreferencesGridLine( 'GridLines' )
        self.Label = utils.PreferencesAxisLabel( 'Label' )

        self.addSubPref('Line', self.Line)
        self.addSubPref('MajorTicks', self.MajorTicks)
        self.addSubPref('MinorTicks', self.MinorTicks)
        self.addSubPref('TickLabels', self.TickLabels)
        self.addSubPref('GridLines', self.GridLines)
        self.addSubPref('Label', self.Label)

        # we recompute the axis later
        self._setModified()
        
    def _getAxisDirection(self):
        """Return direction (0=horz, 1=vert)."""
        return self.dirn

    def _setModified(self, modified=True):
        """Set internal modified bit."""
        self.__dict__['modified'] = modified

    def __setattr__(self, name, value):
        """Overrides default to detect whether parameters have changed."""
        self.__dict__[name] = value

        # this is so we can recompute the axis range
        # try statement reqd as self.prefs.prefnames won't be defined
        try:
            if self.hasPref(name):
                self._setModified()
        except AttributeError:
            pass

    def _autoLookupRange(self):
        """Automatically look up the plotable range from widgets that use this axis."""
        # get the name of this axis from the parent
        # design flaw?
        ourname = self.parent.getChildName( self )

        # iterate over siblings to get axis range
        autorange = [1e99, -1e99]
        changed = False

        for c in self.parent.getChildren():
            range = c.autoAxis( ourname )
            # if range is wider, expand
            if range != None:
                autorange[0] = min( range[0], autorange[0] )
                autorange[1] = max( range[1], autorange[1] )
                changed = True

        # return a default range if nobody gives us one
        if changed:
            return autorange
        else:
            if self.log:
                return [1e-2, 1.]
            else:
                return [0., 1.]
                
    def _computePlottedRange(self):
        """Convert the range requested into a plotted range."""

        self.plottedrange = [self.min, self.max]

        # automatic lookup of minimum
        if self.min == None or self.max == None:
            autorange = self._autoLookupRange()

            if self.min == None:
                self.plottedrange[0] = autorange[0]

            if self.max == None:
                self.plottedrange[1] = autorange[1]

        # work out tick values and expand axes if necessary
        
        as = axisticks.AxisTicks( self.plottedrange[0], self.plottedrange[1],
                                  self.numTicks, self.numMinorTicks,
                                  extendbounds = self.autoExtend,
                                  extendzero = self.autoExtendZero,
                                  logaxis = self.log )

        (self.plottedrange[0],self.plottedrange[1],
         self.majortickscalc, self.minortickscalc) =  as.getTicks()

        if self.majorticks != None:
            self.majortickscalc = numarray.array(self.majorticks)

        if self.minorticks != None:
            self.minortickscalc = numarray.array(self.minorticks)

        self._setModified(False)

    def setOutputPositions(self, dirn=0, f1=0., f2=1., fother=0.):
        """Set where the axis appears on the output graph.
        
        dirn = 0 (Horizontal) non-zero (vertical)
        f1 = fractional position of axis along graph
        f2 = ending fractional position
        fother = fractional position in the other direction
        """
        self.posns = [float(f1), float(f2), float(fother)]
        self.dirn = dirn
    
    def _updatePlotRange(self, bounds):
        """Calculate coordinates on plotter of axis."""

        x1, y1, x2, y2 = bounds
        dx = x2 - x1
        dy = y2 - y1
        p1, p2, pp = self.posns

        if self.dirn == 0: # horizontal
            self.coordParr1 = x1 + int(dx * p1)
            self.coordParr2 = x1 + int(dx * p2)

            # other axis coordinates
            self.coordPerp  = y2 - int(dy * pp)
            self.coordPerp1 = y2 - int(dy * p1)
            self.coordPerp2 = y2 - int(dy * p2)

        else: # vertical
            self.coordParr1 = y2 - int(dy * p1)
            self.coordParr2 = y2 - int(dy * p2)

            # other axis coordinates
            self.coordPerp  = x1 + int(dx * pp)
            self.coordPerp1 = x1 + int(dx * p1)
            self.coordPerp2 = x1 + int(dx * p2)
     
    def graphToPlotterCoords(self, bounds, vals):
        """Convert graph coordinates to plotter coordinates on this axis.

        bounds specifies the plot bounds
        vals is numarray of coordinates
        Returns positions as numarray of integers
        """

        # if the axis was modified, recompute the range
        if self.modified:
            self._computePlottedRange()

        self._updatePlotRange(bounds)

        return self._graphToPlotter(vals)

    def _graphToPlotter(self, vals):
        """Convert the coordinates assuming the machinery is in place."""
        
        # work out fractional posistions, then convert to pixels
        if self.log:
            fracposns = self.logConvertToPlotter( vals )
        else:
            fracposns = self.linearConvertToPlotter( vals )

        # rounds to nearest integer
        out = numarray.floor( 0.5 +
                              self.coordParr1 +
                              fracposns*(self.coordParr2-self.coordParr1) )
        out = out.astype(numarray.Int32)
        return out
    
    def plotterToGraphCoords(self, bounds, vals):
        """Convert plotter coordinates on this axis to graph coordinates.
        
        bounds specifies the plot bounds
        vals is a numarray of coordinates
        returns a numarray of floats
        """
        # if the axis was modified, recompute the range

        if self.modified:
            self._computePlottedRange()

        self._updatePlotRange( bounds )

        # work out fractional positions of the plotter coords
        frac = ( vals.astype(numarray.Float64) - self.coordParr1 ) / \
               ( self.coordParr2 - self.coordParr1 )

        # scaling...
        if self.log:
            return self.logConvertFromPlotter( frac )
        else:
            return self.linearConvertFromPlotter( frac )
        
    def linearConvertToPlotter(self, v):
        """Convert graph coordinates to fractional plotter units for linear scale.
        """
        return ( v - self.plottedrange[0] ) / \
               ( self.plottedrange[1]-self.plottedrange[0] )
    
    def linearConvertFromPlotter(self, v):
        """Convert from (fractional) plotter coords to graph coords.
        """
        return self.plottedrange[0] + v * \
               (self.plottedrange[1]-self.plottedrange[0] )
    
    def logConvertToPlotter(self, v):
        """Convert graph coordinates to fractional plotter units for log10 scale.
        """

        log1 = numarray.log(self.plottedrange[0])
        log2 = numarray.log(self.plottedrange[1])
        return ( numarray.log(v) - log1 )/( log2 - log1 )
    
    def logConvertFromPlotter(self, v):
        """Convert from fraction plotter coords to graph coords with log scale.
        """
        return self.plottedrange[0] * \
               ( self.plottedrange[1]/self.plottedrange[0] )**v
    
    def swapline(self, painter, a1, b1, a2, b2):
        """ Draw line, but swap x & y coordinates if vertical axis."""
        if self.dirn == 0:
            painter.drawLine(a1, b1, a2, b2)
        else:
            painter.drawLine(b1, a1, b2, a2)

    _ticklabel_alignments = (
        (         # horizontal axis
        (0, 1),    # normal
        (1, 0)     # rotated
        ),(       # vertical axis
        (1, 0),    # normal
        (0, -1)    # rotated
        ))

    _axislabel_alignments = (
        (         # horizontal axis
        (0, 1),    # normal
        (1, 0)     # rotated
        ),(       # vertical axis
        (0, -1),   # normal
        (1, 0)     # rotated
        ))

    def _drawGridLines(self, painter, coordticks):
        """Draw grid lines on the plot."""
        
        painter.setPen( self.GridLines.makeQPen(painter) )
        for t in coordticks:
            self.swapline( painter,
                           t, self.coordPerp1,
                           t, self.coordPerp2 )

    def _drawAxisLine(self, painter):
        """Draw the line of the axis."""

        painter.setPen( self.Line.makeQPen(painter) )
        self.swapline( painter,
                       self.coordParr1, self.coordPerp,
                       self.coordParr2, self.coordPerp )        

    def _drawMinorTicks(self, painter):
        """Draw minor ticks on plot."""
        
        painter.setPen( self.MinorTicks.makeQPen(painter) )
        delta = int( self.MinorTicks.length * self._pixperpt )
        minorticks = self._graphToPlotter(self.minortickscalc)

        if self.dirn != 0: delta *= -1   # vertical
        if self.reflect: delta *= -1     # reflection
        for t in minorticks:
            self.swapline( painter,
                           t, self.coordPerp,
                           t, self.coordPerp - delta )

    def _drawMajorTicks(self, painter, tickcoords):
        """Draw major ticks on the plot."""

        painter.setPen( self.MajorTicks.makeQPen(painter) )
        delta = int( self.MajorTicks.length * self._pixperpt )

        if self.dirn != 0: delta *= -1   # vertical
        if self.reflect: delta *= -1     # reflection
        for t in tickcoords:
            self.swapline( painter,
                           t, self.coordPerp,
                           t, self.coordPerp - delta )

        # acount for ticks if they are in the direction of the label
        if delta < 0:
            self._delta_axis += abs(delta)

    def _drawTickLabels(self, painter, coordticks, sign):
        """Draw tick labels on the plot."""

        painter.setPen( self.TickLabels.makeQPen() )
        font = self.TickLabels.makeQFont()
        painter.setFont(font)
        tl_spacing = painter.fontMetrics().leading() + \
                     painter.fontMetrics().descent()
        tl_ascent  = painter.fontMetrics().ascent()

        # work out font alignment
        ax, ay = Axis._ticklabel_alignments[self.dirn] \
                 [self.TickLabels.rotate]
        angle = 0
        if self.TickLabels.rotate: angle = 270

        # if reflected, we want the opposite alignment
        if self.reflect:
            ax, ay = ( -ax, -ay )

        # plot numbers
        f = self.TickLabels.format
        maxwidth = 0
        for t, val in zip(coordticks, self.majortickscalc):
            x, y = t, self.coordPerp + sign*(self._delta_axis+tl_spacing)
            if self.dirn != 0:   x, y = y, x

            num = utils.formatNumber(val, f)
            w = utils.render(painter, font,
                             x, y, num,
                             ax, ay, angle)
            maxwidth = max(maxwidth, w)

        # keep track of where we are
        self._delta_axis += tl_spacing
        if (self.dirn == 0 and angle == 0) or \
           (self.dirn != 0 and angle != 0):
            self._delta_axis += tl_ascent
        else:
            self._delta_axis += maxwidth

    def _drawAxisLabel(self, painter, sign):
        """Draw an axis label on the plot."""
        
        painter.setPen( self.Label.makeQPen() )
        font = self.Label.makeQFont()
        painter.setFont(font)
        al_spacing = painter.fontMetrics().leading() + \
                     painter.fontMetrics().descent()

        # work out font alignment
        ax, ay = Axis._axislabel_alignments[self.dirn] \
                 [self.Label.rotate]

        # if reflected, we want the opposite alignment
        if self.reflect:
            ax, ay = ( -ax, -ay )

        # angle of text
        if (self.dirn == 0 and not self.Label.rotate) or \
           (self.dirn != 0 and self.Label.rotate):
            angle = 0
        else:
            angle = 270

        x = (self.coordParr1 + self.coordParr2)/2
        y = self.coordPerp + sign*(self._delta_axis+al_spacing)
        if self.dirn != 0:
            x, y = y, x

        utils.render(painter, font, x, y,
                     self.Label.label,
                     ax, ay, angle)

    def _autoMirrorDraw(self, posn, painter, coordticks):
        """Mirror axis to opposite side of graph if there isn't an axis there already."""

        countaxis = 0
        siblings = self.parent.getChildren()
        for s in siblings:
            try:
                if self.dirn == s._getAxisDirection():
                    countaxis += 1
                
            except AttributeError:
                # if it's not an axis we get here
                pass

        # another axis in the same direction, so we don't mirror it
        if countaxis != 1:
            return

        # swap axis to other side
        other = self.posns[2]
        if other < 0.5:
            next = 1.
        else:
            next = 0.
        self.posns[2] = next

        self.reflect = not self.reflect
        self._updatePlotRange(posn)
        self._drawAxisLine(painter)
        self._drawMajorTicks(painter, coordticks)
        self._drawMinorTicks(painter)
        self.reflect = not self.reflect

        # put axis back
        self.posns[2] = other

    def draw(self, parentposn, painter):
        """Plot the axis on the painter."""

        # do plotting of children (does that make sense?)
        posn = widget.Widget.draw(self, parentposn, painter)

        # recompute if modified
        if self.modified: self._computePlottedRange()

        self._updatePlotRange(posn)

        # get tick vals
        coordticks = self._graphToPlotter(self.majortickscalc)

        # save the state of the painter for later
        painter.save()

        # calc length of ticks
        self._pixperpt = utils.getPixelsPerPoint(painter)

        # multiplication factor if reflection on the axis is requested
        sign = 1
        if self.dirn != 0: sign *= -1
        if self.reflect:   sign *= -1

        # plot gridlines
        if self.GridLines.notHidden():
            self._drawGridLines(painter, coordticks)

        # plot the line along the axis
        if self.Line.notHidden():
            self._drawAxisLine(painter)

        # plot minor ticks
        if self.MinorTicks.notHidden():
            self._drawMinorTicks(painter)

        # keep track of distance from axis
        self._delta_axis = 0

        # plot major ticks
        if self.MajorTicks.notHidden():
            self._drawMajorTicks(painter, coordticks)

        # plot tick labels
        if self.TickLabels.notHidden():
            self._drawTickLabels(painter, coordticks, sign)

        # draw an axis label
        if self.Label.notHidden():
            self._drawAxisLabel(painter, sign)

        # mirror axis at other side of plot
        if self.autoMirror:
            self._autoMirrorDraw(posn, painter, coordticks)

        # restore the state of the painter
        painter.restore()

# allow the factory to instantiate an axis
widgetfactory.thefactory.register( Axis )
