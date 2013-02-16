#    Copyright (C) 2010 Jeremy S. Sanders
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

"""Non orthogonal point plotting."""

import itertools
import numpy as N

import veusz.qtall as qt4
import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import pickable

from nonorthgraph import NonOrthGraph, FillBrush
from widget import Widget
from point import MarkerFillBrush

def _(text, disambiguation=None, context='NonOrthPoint'):
    """Translate text."""
    return unicode( 
        qt4.QCoreApplication.translate(context, text, disambiguation))

class NonOrthPoint(Widget):
    '''Widget for plotting points in a non-orthogonal plot.'''

    typename = 'nonorthpoint'
    allowusercreation = True
    description = _('Plot points on a graph with non-orthogonal axes')

    allowedparenttypes = [NonOrthGraph]

    def __init__(self, parent, name=None):
        """Initialise plotter."""
        Widget.__init__(self, parent, name=name)
        if type(self) == NonOrthPoint:
            self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        '''Settings for widget.'''
        Widget.addSettings(s)

        s.add( setting.DatasetOrFloatList(
                'data1', 'x',
                descr=_('Dataset containing 1st dataset or list of values'),
                usertext=_('Dataset 1')) )
        s.add( setting.DatasetOrFloatList(
                'data2', 'y',
                descr=_('Dataset containing 2nd dataset or list of values'),
                usertext=_('Dataset 2')) )
        s.add( setting.DatasetOrFloatList(
                'scalePoints', '',
                descr = _('Scale size of plotted markers by this dataset or'
                          ' list of values'),
                usertext=_('Scale markers')) )
        s.add( setting.DatasetOrStr('labels', '',
                                    descr=_('Dataset or string to label points'),
                                    usertext=_('Labels'), datatype='text') )

        s.add( setting.DistancePt('markerSize',
                                  '3pt',
                                  descr = _('Size of marker to plot'),
                                  usertext=_('Marker size'), formatting=True), 0 )
        s.add( setting.Marker('marker',
                              'circle',
                              descr = _('Type of marker to plot'),
                              usertext=_('Marker'), formatting=True), 0 )
        s.add( setting.Line('PlotLine',
                            descr = _('Plot line settings'),
                            usertext = _('Plot line')),
               pixmap = 'settings_plotline' )
        s.add( setting.Line('MarkerLine',
                            descr = _('Line around the marker settings'),
                            usertext = _('Marker border')),
               pixmap = 'settings_plotmarkerline' )
        s.add( MarkerFillBrush('MarkerFill',
                               descr = _('Marker fill settings'),
                               usertext = _('Marker fill')),
               pixmap = 'settings_plotmarkerfill' )
        s.add( FillBrush('Fill1',
                         descr = _('Fill settings (1)'),
                         usertext = _('Area fill 1')),
               pixmap = 'settings_plotfillbelow' )
        s.add( FillBrush('Fill2',
                         descr = _('Fill settings (2)'),
                         usertext = _('Area fill 2')),
               pixmap = 'settings_plotfillbelow' )
        s.add( setting.PointLabel('Label',
                                  descr = _('Label settings'),
                                  usertext=_('Label')),
               pixmap = 'settings_axislabel' )

    def updateDataRanges(self, inrange):
        '''Extend inrange to range of data.'''

        d1 = self.settings.get('data1').getData(self.document)
        if d1:
            inrange[0] = min( N.nanmin(d1.data), inrange[0] )
            inrange[1] = max( N.nanmax(d1.data), inrange[1] )
        d2 = self.settings.get('data2').getData(self.document)
        if d2:
            inrange[2] = min( N.nanmin(d2.data), inrange[2] )
            inrange[3] = max( N.nanmax(d2.data), inrange[3] )

    def pickPoint(self, x0, y0, bounds, distance = 'radial'):
        p = pickable.DiscretePickable(self, 'data1', 'data2',
                lambda v1, v2: self.parent.graphToPlotCoords(v1, v2))
        return p.pickPoint(x0, y0, bounds, distance)

    def pickIndex(self, oldindex, direction, bounds):
        p = pickable.DiscretePickable(self, 'data1', 'data2',
                lambda v1, v2: self.parent.graphToPlotCoords(v1, v2))
        return p.pickIndex(oldindex, direction, bounds)

    def plotMarkers(self, painter, plta, pltb, scaling, markersize, clip):
        '''Draw markers in widget.'''
        s = self.settings
        if not s.MarkerLine.hide or not s.MarkerFill.hide:
            painter.setBrush( s.MarkerFill.makeQBrushWHide() )
            painter.setPen( s.MarkerLine.makeQPenWHide(painter) )

            utils.plotMarkers(painter, plta, pltb, s.marker, markersize,
                              scaling=scaling, clip=clip)

    def drawLabels(self, painter, xplotter, yplotter,
                   textvals, markersize):
        """Draw labels for the points.

        This is copied from the xy (point) widget class, so it
        probably should be somehow be shared.

        FIXME: sane automatic placement of labels
        """

        s = self.settings
        lab = s.get('Label')

        # work out offset an alignment
        deltax = markersize*1.5*{'left':-1, 'centre':0, 'right':1}[lab.posnHorz]
        deltay = markersize*1.5*{'top':-1, 'centre':0, 'bottom':1}[lab.posnVert]
        alignhorz = {'left':1, 'centre':0, 'right':-1}[lab.posnHorz]
        alignvert = {'top':-1, 'centre':0, 'bottom':1}[lab.posnVert]

        # make font and len
        textpen = lab.makeQPen()
        painter.setPen(textpen)
        font = lab.makeQFont(painter)
        angle = lab.angle

        # iterate over each point and plot each label
        for x, y, t in itertools.izip(xplotter+deltax, yplotter+deltay,
                                      textvals):
            utils.Renderer( painter, font, x, y, t,
                            alignhorz, alignvert, angle ).render()

    def draw(self, parentposn, phelper, outerbounds=None):
        '''Plot the data on a plotter.'''

        posn = self.computeBounds(parentposn, phelper)
        s = self.settings
        d = self.document

        # exit if hidden
        if s.hide:
            return

        d1 = s.get('data1').getData(d)
        d2 = s.get('data2').getData(d)
        dscale = s.get('scalePoints').getData(d)
        text = s.get('labels').getData(d, checknull=True)
        if not d1 or not d2:
            return

        x1, y1, x2, y2 = posn
        cliprect = qt4.QRectF( qt4.QPointF(x1, y1), qt4.QPointF(x2, y2) )
        painter = phelper.painter(self, posn)
        with painter:
            self.parent.setClip(painter, posn)

            # split parts separated by NaNs
            for v1, v2, scalings, textitems in document.generateValidDatasetParts(
                d1, d2, dscale, text):
                # convert data (chopping down length)
                v1d, v2d = v1.data, v2.data
                minlen = min(v1d.shape[0], v2d.shape[0])
                v1d, v2d = v1d[:minlen], v2d[:minlen]
                px, py = self.parent.graphToPlotCoords(v1d, v2d)

                # do fill1 (if any)
                if not s.Fill1.hide:
                    self.parent.drawFillPts(painter, s.Fill1, cliprect, px, py)
                # do fill2
                if not s.Fill2.hide:
                    self.parent.drawFillPts(painter, s.Fill2, cliprect, px, py)

                # plot line
                if not s.PlotLine.hide:
                    painter.setBrush( qt4.QBrush() )
                    painter.setPen(s.PlotLine.makeQPen(painter))
                    pts = qt4.QPolygonF()
                    utils.addNumpyToPolygonF(pts, px, py)
                    utils.plotClippedPolyline(painter, cliprect, pts)

                # markers
                markersize = s.get('markerSize').convert(painter)
                pscale = None
                if scalings:
                    pscale = scalings.data
                self.plotMarkers(painter, px, py, pscale, markersize, cliprect)

                # finally plot any labels
                if textitems and not s.Label.hide:
                    self.drawLabels(painter, px, py, textitems, markersize)

# allow the factory to instantiate plotter
document.thefactory.register( NonOrthPoint )
