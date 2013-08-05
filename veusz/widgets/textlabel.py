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
import math
import numpy as N

from .. import document
from .. import setting
from .. import utils
from .. import qtall as qt4

from . import plotters
from . import controlgraph

def _(text, disambiguation=None, context='TextLabel'):
    """Translate text."""
    return unicode( 
        qt4.QCoreApplication.translate(context, text, disambiguation))

class TextLabel(plotters.FreePlotter):

    """Add a text label to a graph."""

    typename = 'label'
    description = _('Text label')
    allowusercreation = True

    def __init__(self, parent, name=None):
        plotters.FreePlotter.__init__(self, parent, name=name)

        if type(self) == TextLabel:
            self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        plotters.FreePlotter.addSettings(s)

        s.add( setting.DatasetOrStr('label', '',
                                    descr=_('Text to show or text dataset'),
                                    usertext=_('Label'), datatype='text'), 0 )

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

        s.add( setting.Bool('clip', False,
                            descr=_('Clip text to its container'),
                            usertext=_('Clip'),
                            formatting=True), 10 )

        s.add( setting.Text('Text',
                            descr = _('Text settings'),
                            usertext=_('Text')),
               pixmap = 'settings_axislabel' )

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
            clip = qt4.QRectF( qt4.QPointF(posn[0], posn[1]),
                               qt4.QPointF(posn[2], posn[3]) )

        painter = phelper.painter(self, posn, clip=clip)
        with painter:
            textpen = s.get('Text').makeQPen()
            painter.setPen(textpen)
            font = s.get('Text').makeQFont(painter)

            # we should only be able to move non-dataset labels
            isnotdataset = ( not s.get('xPos').isDataset(d) and
                             not s.get('yPos').isDataset(d) )

            controlgraphitems = []
            for index, (x, y, t) in enumerate(itertools.izip(
                    xp, yp, itertools.cycle(text))):
                # render the text
                tbounds = utils.Renderer( painter, font, x, y, t,
                                          TextLabel.cnvtalignhorz[s.alignHorz],
                                          TextLabel.cnvtalignvert[s.alignVert],
                                          s.angle ).render()

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
        if xpos is None or ypos is None:
            return

        pointsX[ind], pointsY[ind] = xpos, ypos
        operations = (
            document.OperationSettingSet(s.get('xPos'), pointsX),
            document.OperationSettingSet(s.get('yPos'), pointsY)
            )
        self.document.applyOperation(
            document.OperationMultiple(operations, descr=_('move label')) )

# allow the factory to instantiate a text label
document.thefactory.register( TextLabel )
