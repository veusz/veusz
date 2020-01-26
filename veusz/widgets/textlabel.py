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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
###############################################################################

"""For plotting one or more text labels on a graph."""

from __future__ import division
import itertools

from ..compat import czip
from .. import document
from .. import setting
from .. import utils
from .. import qtall as qt

from . import plotters
from . import controlgraph

def _(text, disambiguation=None, context='TextLabel'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class BorderLine(setting.Line):
    '''Plot line around text.'''

    def __init__(self, name, **args):
        setting.Line.__init__(self, name, **args)
        self.get('hide').newDefault(True)

class TextLabel(plotters.FreePlotter):

    """Add a text label to a graph."""

    typename = 'label'
    description = _('Text label')
    allowusercreation = True

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        plotters.FreePlotter.addSettings(s)

        s.add( setting.DatasetOrStr('label', '',
                                    descr=_('Text to show or text dataset'),
                                    usertext=_('Label')), 0 )

        s.add( setting.AlignHorz('alignHorz',
                                 'left',
                                 descr=_('Horizontal alignment of label'),
                                 usertext=_('Horz alignment'),
                                 formatting=True), 7)
        s.add( setting.AlignVert('alignVert',
                                 'bottom',
                                 descr=_('Vertical alignment of label'),
                                 usertext=_('Vert alignment'),
                                 formatting=True), 8)

        s.add( setting.Float('angle', 0.,
                             descr=_('Angle of the label in degrees'),
                             usertext=_('Angle'),
                             formatting=True), 9 )

        s.add( setting.DistancePt(
                'margin',
                '4pt',
                descr = _('Margin of fill/border'),
                usertext=_('Margin'),
                formatting=True), 10 )

        s.add( setting.Bool('clip', False,
                            descr=_('Clip text to its container'),
                            usertext=_('Clip'),
                            formatting=True), 11 )

        s.add( setting.Text('Text',
                            descr = _('Text settings'),
                            usertext=_('Text')),
               pixmap = 'settings_axislabel' )
        s.add( setting.ShapeFill(
                'Background',
                descr=_('Fill behind text'),
                usertext=_('Background')),
               pixmap = 'settings_bgfill' )
        s.add( BorderLine(
                'Border',
                descr=_('Border around text'),
                usertext=_('Border')),
               pixmap = 'settings_border' )

    # convert text to alignments used by Renderer
    cnvtalignhorz = { 'left': -1, 'centre': 0, 'right': 1 }
    cnvtalignvert = { 'top': 1, 'centre': 0, 'bottom': -1 }

    @property
    def userdescription(self):
        """User friendly description."""
        s = self.settings
        return _("text='%s'") % s.label

    def draw(self, posn, phelper, outerbounds = None):
        """Draw the text label."""

        s = self.settings
        d = self.document

        # exit if hidden
        if s.hide or s.Text.hide:
            return

        text = s.get('label').getData(d)

        xp, yp = self._getPlotterCoords(posn)
        if xp is None or yp is None:
            # we can't calculate coordinates
            return

        clip = None
        if s.clip:
            clip = qt.QRectF(
                qt.QPointF(posn[0], posn[1]), qt.QPointF(posn[2], posn[3]))

        borderorfill = not s.Border.hide or not s.Background.hide

        painter = phelper.painter(self, posn, clip=clip)
        with painter:
            textpen = s.get('Text').makeQPen(painter)
            painter.setPen(textpen)
            font = s.get('Text').makeQFont(painter)
            margin = s.get('margin').convert(painter)

            # we should only be able to move non-dataset labels
            isnotdataset = ( not s.get('xPos').isDataset(d) and
                             not s.get('yPos').isDataset(d) )

            controlgraphitems = []
            for index, (x, y, t) in enumerate(czip(
                    xp, yp, itertools.cycle(text))):
                # render the text

                dx = dy = 0
                if borderorfill:
                    dx = -TextLabel.cnvtalignhorz[s.alignHorz]*margin
                    dy =  TextLabel.cnvtalignvert[s.alignVert]*margin

                r = utils.Renderer(
                    painter, font, x+dx, y+dy, t,
                    TextLabel.cnvtalignhorz[s.alignHorz],
                    TextLabel.cnvtalignvert[s.alignVert],
                    s.angle,
                    doc=d)

                tbounds = r.getBounds()
                if borderorfill:
                    tbounds = [
                        tbounds[0]-margin, tbounds[1]-margin,
                        tbounds[2]+margin, tbounds[3]+margin ]
                    rect = qt.QRectF(
                        qt.QPointF(tbounds[0], tbounds[1]),
                        qt.QPointF(tbounds[2], tbounds[3]))
                    path = qt.QPainterPath()
                    path.addRect(rect)
                    pen = s.get('Border').makeQPenWHide(painter)
                    utils.brushExtFillPath(painter, s.Background, path,
                                           stroke=pen)

                r.render()

                # add cgi for adjustable positions
                if isnotdataset:
                    cgi = controlgraph.ControlMovableBox(self, tbounds, phelper,
                                                         crosspos = (x, y))
                    cgi.labelpt = (x, y)
                    cgi.widgetposn = posn
                    cgi.index = index
                    controlgraphitems.append(cgi)

        phelper.setControlGraph(self, controlgraphitems)

    def updateControlItem(self, cgi):
        """Update position of point given new name and vals."""

        s = self.settings
        pointsX = list(s.xPos)   # make a copy here so original is not modifed
        pointsY = list(s.yPos)
        ind = cgi.index

        # calculate new position coordinate for item
        xpos, ypos = self._getGraphCoords(cgi.widgetposn,
                                          cgi.deltacrosspos[0]+cgi.posn[0],
                                          cgi.deltacrosspos[1]+cgi.posn[1])
        # this is a small distance away to get delta
        xposd, yposd = self._getGraphCoords(cgi.widgetposn,
                                            cgi.deltacrosspos[0]+cgi.posn[0]+1,
                                            cgi.deltacrosspos[1]+cgi.posn[1]+1)
        if xpos is None or ypos is None:
            return

        roundx = utils.round2delt(xpos, xposd)
        roundy = utils.round2delt(ypos, yposd)

        pointsX[ind], pointsY[ind] = roundx, roundy
        operations = (
            document.OperationSettingSet(s.get('xPos'), pointsX),
            document.OperationSettingSet(s.get('yPos'), pointsY)
            )
        self.document.applyOperation(
            document.OperationMultiple(operations, descr=_('move label')) )

# allow the factory to instantiate a text label
document.thefactory.register( TextLabel )
