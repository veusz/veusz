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
###############################################################################

# $Id$

"""For plotting one or more text labels on a graph."""

import itertools

import numpy as N

import veusz.qtall as qt4
import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

import plotters
import graph
import widget
import controlgraph

class TextLabel(plotters.GenericPlotter):

    """Add a text label to a graph."""

    typename = 'label'
    description = "Text label"
    allowedparenttypes = [graph.Graph]
    allowusercreation = True

    def __init__(self, parent, name=None):
        plotters.GenericPlotter.__init__(self, parent, name=name)
        s = self.settings

        # text labels don't need key symbols
        s.remove('key')

        s.add( setting.DatasetOrStr('label', '',
                                    descr='Text to show or text dataset',
                                    usertext='Label', datatype='text'), 0 )
        s.add( setting.DatasetOrFloatList('xPos', 0.5,
                                          descr='List of X coordinates or dataset',
                                          usertext='X position',
                                          formatting=False), 1 )
        s.add( setting.DatasetOrFloatList('yPos', 0.5,
                                          descr='List of Y coordinates or dataset',
                                          usertext='Y position',
                                          formatting=False), 2 )

        s.add( setting.Choice('positioning',
                              ['axes', 'relative'], 'relative',
                              descr='Use axes or fractional position to '
                              'place label',
                              usertext='Position mode',
                              formatting=False), 6)

        s.add( setting.Choice('alignHorz',
                              ['left', 'centre', 'right'], 'left',
                              descr="Horizontal alignment of label",
                              usertext='Horz alignment',
                              formatting=True), 7)
        s.add( setting.Choice('alignVert',
                              ['top', 'centre', 'bottom'], 'bottom',
                              descr='Vertical alignment of label',
                              usertext='Vert alignment',
                              formatting=True), 8)

        s.add( setting.Float('angle', 0.,
                             descr='Angle of the label in degrees',
                             usertext='Angle',
                             formatting=True), 9 )

        s.add( setting.Text('Text',
                            descr = 'Text settings',
                            usertext='Text'),
               pixmap = 'axislabel' )

        if type(self) == TextLabel:
            self.readDefaults()

    # convert text to alignments used by Renderer
    cnvtalignhorz = { 'left': -1, 'centre': 0, 'right': 1 }
    cnvtalignvert = { 'top': 1, 'centre': 0, 'bottom': -1 }

    def draw(self, parentposn, painter, outerbounds = None):
        """Draw the text label."""

        posn = plotters.GenericPlotter.draw(self, parentposn, painter,
                                            outerbounds=outerbounds)

        s = self.settings
        d = self.document

        # exit if hidden
        if s.hide or s.Text.hide:
            return

        text = s.get('label').getData(d)
        pointsX = s.get('xPos').getFloatArray(d)
        pointsY = s.get('yPos').getFloatArray(d)

        if pointsX is None or pointsY is None:
            return

        if s.positioning == 'axes':
            # translate xPos and yPos to plotter coordinates

            axes = self.parent.getAxes( (s.xAxis, s.yAxis) )
            if None in axes:
                return
            xp = axes[0].graphToPlotterCoords(posn, pointsX)
            yp = axes[1].graphToPlotterCoords(posn, pointsY)
        else:
            # work out fractions inside pos
            xp = posn[0] + (posn[2]-posn[0])*pointsX
            yp = posn[3] + (posn[1]-posn[3])*pointsY

        painter.beginPaintingWidget(self, parentposn)
        painter.save()
        textpen = s.get('Text').makeQPen()
        painter.setPen(textpen)
        font = s.get('Text').makeQFont(painter)

        self.controlgraphitems = []
        isnotdataset = ( not s.get('xPos').isDataset(d) and 
                         not s.get('yPos').isDataset(d) )

        for index, (x, y, t) in enumerate(itertools.izip(
                xp, yp, itertools.cycle(text))):
            tbounds = utils.Renderer( painter, font, x, y, t,
                                      TextLabel.cnvtalignhorz[s.alignHorz],
                                      TextLabel.cnvtalignvert[s.alignVert],
                                      s.angle ).render()
            if isnotdataset:
                cgi = controlgraph.ControlGraphMovableBox(self, tbounds,
                                                          crosspos = (x, y))
                cgi.labelpt = (x, y)
                cgi.widgetposn = posn
                cgi.index = index
                self.controlgraphitems.append(cgi)

        painter.restore()
        painter.endPaintingWidget()

    def updateControlItem(self, cgi):
        """Update position of point given new name and vals."""

        s = self.settings
        pointsX = list(s.xPos)   # make a copy here so original is not modifed
        pointsY = list(s.yPos)
        bounds = cgi.widgetposn
        ind = cgi.index

        # convert movement of bounds into movement of label itself
        x = cgi.labelpt[0] + cgi.pos().x()
        y = cgi.labelpt[1] + cgi.pos().y()

        if s.positioning == 'axes':
            # positions in axes coordinates
            axes = self.parent.getAxes( (s.xAxis, s.yAxis) )
            if not (None in axes):
                pointsX[ind] = axes[0].plotterToGraphCoords(bounds, N.array(x))
                pointsY[ind] = axes[1].plotterToGraphCoords(bounds, N.array(y))
        else:
            # positions in graph relative coordinates
            pointsX[ind] = (x - bounds[0]) / (bounds[2]-bounds[0])
            pointsY[ind] = (y - bounds[3]) / (bounds[1]-bounds[3])

        operations = (
            document.OperationSettingSet(s.get('xPos'), pointsX),
            document.OperationSettingSet(s.get('yPos'), pointsY)
            )
        self.document.applyOperation(
            document.OperationMultiple(operations, descr='move label') )

# allow the factory to instantiate a text label
document.thefactory.register( TextLabel )
