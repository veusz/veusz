#    Copyright (C) 2005 Jeremy S. Sanders
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

"""Collections of predefined settings for common settings."""

from __future__ import division
from .. import qtall as qt4

from . import setting
from .settings import Settings
from .reference import Reference

def _(text, disambiguation=None, context="Setting"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class Line(Settings):
    '''For holding properities of a line.'''

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.Color(
            'color',
            setting.Reference('/StyleSheet/Line/color'),
            descr = _('Color of line'),
            usertext = _('Color')) )
        self.add( setting.DistancePt(
            'width',
            setting.Reference('/StyleSheet/Line/width'),
            descr = _('Width of line'),
            usertext = _('Width')) )
        self.add( setting.LineStyle(
            'style', 'solid',
            descr = _('Line style'),
            usertext = _('Style')) )
        self.add( setting.Int(
            'transparency', 0,
            descr = _('Transparency percentage'),
            usertext = _('Transparency'),
            minval = 0,
            maxval = 100 ) )
        self.add( setting.Bool(
            'hide', False,
            descr = _('Hide the line'),
            usertext = _('Hide')) )

    def makeQPen(self, painter):
        '''Make a QPen from the settings (ignoring hide).
        '''

        color = self.get('color').color(painter)
        if self.transparency > 0:
            color.setAlphaF((100-self.transparency) / 100.)
        width = self.get('width').convert(painter)
        style, dashpattern = setting.LineStyle._linecnvt[self.style]
        pen = qt4.QPen(color, width, style)

        if dashpattern:
            pen.setDashPattern(dashpattern)

        return pen

    def makeQPenWHide(self, painter):
        """Make a pen, taking account of hide attribute."""
        if self.hide:
            return qt4.QPen(qt4.Qt.NoPen)
        else:
            return self.makeQPen(painter)

class XYPlotLine(Line):
    '''A plot line for plotting data, allowing histogram-steps
    to be plotted.'''

    def __init__(self, name, **args):
        Line.__init__(self, name, **args)

        self.add( setting.Choice(
            'steps',
            [
                'off', 'left', 'centre', 'right',
                'left-shift-points', 'right-shift-points',
                'vcentre'
            ],
            'off',
            descr = _('Plot 90 degree steps instead of a line'),
            usertext = _('Steps')), 0 )
        self.add( setting.Bool(
            'bezierJoin', False,
            descr=_('Connect points with a cubic Bezier curve'),
            usertext = _('Bezier join')), 1 )
        self.get('color').newDefault( Reference('../color') )

class MarkerLine(Line):
    '''A line for marker border.'''

    def __init__(self, name, **args):
        Line.__init__(self, name, **args)

        self.add( setting.Bool(
            'scaleLine', True,
            descr=_('Scale line width with marker if scaling'
                    ' enabled'),
            usertext=_('Scale')), 4 )

class ErrorBarLine(Line):
    '''A line style for error bar plotting.'''

    def __init__(self, name, **args):
        Line.__init__(self, name, **args)

        self.add( setting.Float(
            'endsize', 1.0,
            minval = 0.,
            descr = _('Scale ends of error bars by this factor'),
            usertext = _('End size')) )
        self.add( setting.Bool(
            'hideHorz', False,
            descr = _('Hide horizontal errors'),
            usertext = _('Hide horz.')) )
        self.add( setting.Bool(
            'hideVert', False,
            descr = _('Hide vertical errors'),
            usertext=_('Hide vert.')) )

class Brush(Settings):
    '''Settings of a fill.'''

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.Color(
            'color', 'foreground',
            descr = _('Fill colour'),
            usertext = _('Color')) )
        self.add( setting.FillStyle(
            'style', 'solid',
            descr = _('Fill style'),
            usertext = _('Style')) )
        self.add( setting.Int(
            'transparency', 0,
            descr = _('Transparency percentage'),
            usertext = _('Transparency'),
            minval = 0,
            maxval = 100 ) )
        self.add( setting.Bool(
            'hide', False,
            descr = _('Hide the fill'),
            usertext = _('Hide')) )

    def makeQBrush(self, painter):
        '''Make a QBrush from the settings.'''
        color = self.get('color').color(painter)
        if self.transparency > 0:
            color.setAlphaF((100-self.transparency) / 100.)
        return qt4.QBrush(color, self.get('style').qtStyle())

    def makeQBrushWHide(self, painter):
        """Make a brush, taking account of hide attribute."""
        if self.hide:
            return qt4.QBrush()
        else:
            return self.makeQBrush(painter)

class BrushExtended(Settings):
    '''Extended brush style.'''

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.Color(
                'color', 'foreground',
                descr = _('Fill colour'),
                usertext=_('Color')) )
        self.add( setting.FillStyleExtended(
                'style', 'solid',
                descr = _('Fill style'),
                usertext=_('Style')) )

        self.add( setting.Bool(
                'hide', False,
                descr = _('Hide the fill'),
                usertext=_('Hide')) )

        self.add( setting.Int(
                'transparency', 0,
                descr = _('Transparency percentage'),
                usertext = _('Transparency'),
                minval = 0,
                maxval = 100 ) )
        self.add( setting.DistancePt(
                'linewidth', '0.5pt',
                descr = _('Width of hatch or pattern line'),
                usertext=_('Line width')) )
        self.add( setting.LineStyle(
                'linestyle', 'solid',
                descr = _('Hatch or pattern line style'),
                usertext=_('Line style')) )
        self.add( setting.DistancePt(
                'patternspacing', '5pt',
                descr = _('Hatch or pattern spacing'),
                usertext = _('Spacing')) )
        self.add( setting.Color(
                'backcolor', 'background',
                descr = _('Hatch or pattern background color'),
                usertext = _('Back color') ) )
        self.add( setting.Int(
                'backtransparency', 0,
                descr = _('Hatch or pattern background transparency percentage'),
                usertext = _('Back trans.'),
                minval = 0,
                maxval = 100 ) )
        self.add( setting.Bool(
                'backhide', True,
                descr = _('Hide hatch or pattern background'),
                usertext=_('Back hide')) )

class KeyBrush(BrushExtended):
    '''Fill used for back of key.'''

    def __init__(self, name, **args):
        BrushExtended.__init__(self, name, **args)

        self.get('color').newDefault('background')

class BoxPlotMarkerFillBrush(Brush):
    '''Fill used for points on box plots.'''

    def __init__(self, name, **args):
        Brush.__init__(self, name, **args)

        self.get('color').newDefault('background')

class GraphBrush(BrushExtended):
    '''Fill used for back of graph.'''

    def __init__(self, name, **args):
        BrushExtended.__init__(self, name, **args)

        self.get('color').newDefault('background')

class PlotterFill(BrushExtended):
    '''Filling used for filling on plotters.'''

    def __init__(self, name, **args):
        BrushExtended.__init__(self, name, **args)

        self.get('hide').newDefault(True)

class PointFill(BrushExtended):
    '''Filling used for filling above/below line or inside error
    region for xy-point plotters.
    '''

    def __init__(self, name, **args):
        BrushExtended.__init__(self, name, **args)

        hide = self.get('hide')
        hide.newDefault(True)
        hide.usertext = _('Hide edge fill')
        hide.descr = _('Hide the filled region to the edge of the plot')
        self.get('color').newDefault('grey')

        self.add( setting.Choice(
            'fillto',
            ['top', 'bottom', 'left', 'right'],
            'top',
            descr=_('Edge to fill towards'),
            usertext=_('Fill to')), 0)

        self.add( setting.Bool(
            'hideerror', False,
            descr = _('Hide the filled region inside the error bars'),
            usertext=_('Hide error fill')) )

class ShapeFill(BrushExtended):
    '''Filling used for filling shapes.'''

    def __init__(self, name, **args):
        BrushExtended.__init__(self, name, **args)

        self.get('hide').newDefault(True)
        self.get('color').newDefault('background')

class ArrowFill(Brush):
    """Brush for filling arrow heads"""
    def __init__(self, name, **args):
        Brush.__init__(self, name, **args)

        self.get('color').newDefault( setting.Reference(
                '../Line/color') )

class Text(Settings):
    '''Text settings.'''

    # need to examine font table to see what's available
    # this is set on app startup
    defaultfamily = None
    families = None

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.FontFamily(
            'font',
            setting.Reference('/StyleSheet/Font/font'),
            descr = _('Font name'),
            usertext = _('Font')) )
        self.add( setting.DistancePt(
            'size',
            setting.Reference('/StyleSheet/Font/size'),
            descr = _('Font size'),
            usertext = _('Size') ) )
        self.add( setting.Color(
            'color',
            setting.Reference('/StyleSheet/Font/color'),
            descr = _('Font color'),
            usertext = _('Color') ) )
        self.add( setting.Bool(
            'italic', False,
            descr = _('Italic font'),
            usertext = _('Italic') ) )
        self.add( setting.Bool(
            'bold', False,
            descr = _('Bold font'),
            usertext = _('Bold') ) )
        self.add( setting.Bool(
            'underline', False,
            descr = _('Underline font'),
            usertext = _('Underline') ) )
        self.add( setting.Bool(
            'hide', False,
            descr = _('Hide the text'),
            usertext = _('Hide')) )

    def copy(self):
        """Make copy of settings."""
        c = Settings.copy(self)
        c.defaultfamily = self.defaultfamily
        c.families = self.families
        return c

    def makeQFont(self, painthelper):
        '''Return a qt4.QFont object corresponding to the settings.'''

        size = self.get('size').convertPts(painthelper)
        weight = qt4.QFont.Normal
        if self.bold:
            weight = qt4.QFont.Bold

        f = qt4.QFont(self.font, size,  weight, self.italic)
        if self.underline:
            f.setUnderline(True)
        f.setStyleHint(qt4.QFont.Times)

        return f

    def makeQPen(self, painter):
        """ Return a qt4.QPen object for the font pen """
        color = self.get('color').color(painter)
        return qt4.QPen(color)

class PointLabel(Text):
    """For labelling points on plots."""

    def __init__(self, name, **args):
        Text.__init__(self, name, **args)

        self.add( setting.Float(
            'angle', 0.,
            descr = _('Angle of the labels in degrees'),
            usertext = _('Angle'),
            formatting = True), 0 )
        self.add( setting.AlignVert(
            'posnVert',
            'centre',
            descr = _('Vertical position of label'),
            usertext = _('Vert position'),
            formatting = True), 0 )
        self.add( setting.AlignHorz(
            'posnHorz',
            'right',
            descr = _('Horizontal position of label'),
            usertext = _('Horz position'),
            formatting = True), 0 )

class MarkerColor(Settings):
    """Settings for a coloring points using data values."""

    def __init__(self, name):
        Settings.__init__(self, name, setnsmode='groupedsetting')
        self.add( setting.DatasetExtended(
                'points', '',
                descr = _('Use color value (0-1) in dataset to paint points'),
                usertext=_('Color markers')), 7 )
        self.add( setting.Float(
                'min', 0.,
                descr = _('Minimum value of color dataset'),
                usertext = _('Min val') ))
        self.add( setting.Float(
                'max', 1.,
                descr = _('Maximum value of color dataset'),
                usertext = _('Max val') ))
        self.add( setting.Choice(
                'scaling',
                ['linear', 'sqrt', 'log', 'squared'],
                'linear',
                descr = _('Scaling to transform numbers to color'),
                usertext=_('Scaling')))
