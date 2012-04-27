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

import itertools

import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils
import veusz.qtall as qt4

import plotters
import controlgraph

mmlsupport = True
try:
    import veusz.helpers.qtmml as qtmml
    import veusz.helpers.recordpaint as recordpaint
except ImportError:
    mmlsupport = False

class TextLabel(plotters.FreePlotter):

    """Add a text label to a graph."""

    typename = 'label'
    description = "Text label"
    allowusercreation = True

    def __init__(self, parent, name=None):
        plotters.FreePlotter.__init__(self, parent, name=name)
        self.mmltextcache = self.mmldoccache = None

        if type(self) == TextLabel:
            self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        plotters.FreePlotter.addSettings(s)

        s.add( setting.DatasetOrStr('label', '',
                                    descr='Text to show or text dataset',
                                    usertext='Label', datatype='text'), 0 )

        s.add( setting.Choice('type',
                              ('standard', 'mathml'),
                              'standard',
                              descr='Type of text to plot',
                              usertext='Type'), 4 )

        s.add( setting.AlignHorz('alignHorz',
                                 'left',
                                 descr="Horizontal alignment of label",
                                 usertext='Horz alignment',
                                 formatting=True), 7)
        s.add( setting.AlignVert('alignVert',
                                 'bottom',
                                 descr='Vertical alignment of label',
                                 usertext='Vert alignment',
                                 formatting=True), 8)

        s.add( setting.Float('angle', 0.,
                             descr='Angle of the label in degrees',
                             usertext='Angle',
                             formatting=True), 9 )

        s.add( setting.Bool('clip', False,
                            descr='Clip text to its container',
                            usertext='Clip',
                            formatting=True), 10 )

        s.add( setting.Text('Text',
                            descr = 'Text settings',
                            usertext='Text'),
               pixmap = 'settings_axislabel' )

    # convert text to alignments used by Renderer
    cnvtalignhorz = { 'left': -1, 'centre': 0, 'right': 1 }
    cnvtalignvert = { 'top': 1, 'centre': 0, 'bottom': -1 }

    def drawMML(self, painter, phelper, font, x, y, t):
        """Draw text in MathML mode."""

        s = self.settings
        if not mmlsupport:
            self.document.log("Error: MathML support not compiled")
            return [x, y, x, y]

        if t != self.mmltextcache:
            self.mmltextcache = t
            self.mmldoccache = qtmml.QtMmlDocument()
            try:
                self.mmldoccache.setContent(t)
            except ValueError, e:
                self.document.log("Error interpreting MathML: %s" %
                                  unicode(e))

        self.mmldoccache.setFontName( qtmml.QtMmlWidget.NormalFont,
                                      font.family() )
        self.mmldoccache.setBaseFontPointSize(
            s.Text.get('size').convertPts(painter) * phelper.dpi[0]/72. )
        size = self.mmldoccache.size()

        # do alignment
        if s.alignHorz == 'centre':
            x -= size.width()/2
        elif s.alignHorz == 'right':
            x -= size.width()
        if s.alignVert == 'centre':
            y -= size.height()/2
        elif s.alignVert == 'bottom':
            y -= size.height()

        # do painting to a device with screen resolution
        # this is assumed by the mml rendering code
        screendev = qt4.QApplication.desktop()
        rc = recordpaint.RecordPaintDevice(
            1024, 1024, screendev.logicalDpiX(), screendev.logicalDpiY())
        rcpaint = qt4.QPainter(rc)
        self.mmldoccache.paint(rcpaint, qt4.QPoint(x, y))
        rcpaint.end()

        # replay back to painter
        rc.play(painter)

        # bounds for control
        tbounds = [x, y, x+size.width(), y+size.height()]
        return tbounds

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
        textpen = s.get('Text').makeQPen()
        painter.setPen(textpen)
        font = s.get('Text').makeQFont(painter)
        ttype = s.type

        # we should only be able to move non-dataset labels
        isnotdataset = ( not s.get('xPos').isDataset(d) and 
                         not s.get('yPos').isDataset(d) )

        controlgraphitems = []
        for index, (x, y, t) in enumerate(itertools.izip(
                xp, yp, itertools.cycle(text))):
            # render the text
            if ttype == 'standard':
                tbounds = utils.Renderer( painter, font, x, y, t,
                                          TextLabel.cnvtalignhorz[s.alignHorz],
                                          TextLabel.cnvtalignvert[s.alignVert],
                                          s.angle ).render()
            elif ttype == 'mathml':
                tbounds = self.drawMML(painter, phelper, font, x, y, t)

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
            document.OperationMultiple(operations, descr='move label') )

# allow the factory to instantiate a text label
document.thefactory.register( TextLabel )
