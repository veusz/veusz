#    Copyright (C) 2005 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This file is part of Veusz.
#
#    Veusz is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    Veusz is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Veusz. If not, see <https://www.gnu.org/licenses/>.
#
##############################################################################

"""Collections of predefined settings for common settings."""

from .. import qtall as qt

from . import setting
from .settings import Settings
from .reference import Reference

def _(text, disambiguation=None, context="Setting"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class Line(Settings):
    '''For holding properities of a line.'''

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.Color(
            'color',
            setting.Reference('/StyleSheet/Line/color'),
            descr=_('Color of line'),
            usertext=_('Color')) )
        self.add( setting.DistancePt(
            'width',
            setting.Reference('/StyleSheet/Line/width'),
            descr=_('Width of line'),
            usertext=_('Width')) )
        self.add( setting.LineStyle(
            'style', 'solid',
            descr=_('Line style'),
            usertext=_('Style')) )
        self.add( setting.Int(
            'transparency', 0,
            descr=_('Transparency percentage'),
            usertext=_('Transparency'),
            minval=0,
            maxval=100 ) )
        self.add( setting.Bool(
            'hide', False,
            descr=_('Hide the line'),
            usertext=_('Hide')) )

    def makeQPen(self, painter):
        '''Make a QPen from the settings (ignoring hide).
        '''

        color = self.get('color').color(painter)
        if self.transparency > 0:
            color.setAlphaF((100-self.transparency) / 100.)
        width = self.get('width').convert(painter)
        style, dashpattern = setting.LineStyle._linecnvt[self.style]
        pen = qt.QPen(color, width, style)
        # this needs to be set for PDF output, otherwise arrows don't
        # plot correctly
        pen.setMiterLimit(4)

        if dashpattern:
            pen.setDashPattern(dashpattern)

        return pen

    def makeQPenWHide(self, painter):
        """Make a pen, taking account of hide attribute."""
        if self.hide or self.transparency == 100:
            return qt.QPen(qt.Qt.PenStyle.NoPen)
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
            descr=_('Plot 90 degree steps instead of a line'),
            usertext=_('Steps')), 0 )
        self.add( setting.Choice(
            'interpType',
            [
                'linear', 'loose-Bezier', 'tight-Bezier',
            ],
            'linear',
            descr=_('Line style (linear/curved) to connect points'),
            usertext=_('Interpolation')), 1 )
        # translate bezierJoin to interpType
        self.add( setting.SettingBackwardCompat(
            'bezierJoin',
            'interpType',
            False,
            translatefn=lambda x: {
                True: 'loose-Bezier',
                False: 'linear'
            }[x],
            formatting=True,
        ) )
        self.get('color').newDefault( Reference('../color') )

class MarkerLine(Line):
    '''A line for marker border.'''

    def __init__(self, name, **args):
        Line.__init__(self, name, **args)

        self.add( setting.Bool(
            'scaleLine', True,
            descr=_('Scale line width with marker if scaling enabled'),
            usertext=_('Scale')), 4 )

class ErrorBarLine(Line):
    '''A line style for error bar plotting.'''

    def __init__(self, name, **args):
        Line.__init__(self, name, **args)

        self.add( setting.Float(
            'endsize', 1.0,
            minval=0.,
            descr=_('Scale ends of error bars by this factor'),
            usertext=_('End size')) )
        self.add( setting.Bool(
            'hideHorz', False,
            descr=_('Hide horizontal errors'),
            usertext=_('Hide horz.')) )
        self.add( setting.Bool(
            'hideVert', False,
            descr=_('Hide vertical errors'),
            usertext=_('Hide vert.')) )

class Brush(Settings):
    '''Settings of a fill.'''

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.Color(
            'color', 'foreground',
            descr=_('Fill colour'),
            usertext=_('Color')) )
        self.add( setting.FillStyle(
            'style', 'solid',
            descr=_('Fill style'),
            usertext=_('Style')) )
        self.add( setting.Int(
            'transparency', 0,
            descr=_('Transparency percentage'),
            usertext=_('Transparency'),
            minval=0,
            maxval=100 ) )
        self.add( setting.Bool(
            'hide', False,
            descr=_('Hide the fill'),
            usertext=_('Hide')) )

    def makeQBrush(self, painter):
        '''Make a QBrush from the settings.'''
        color = self.get('color').color(painter)
        if self.transparency > 0:
            color.setAlphaF((100-self.transparency) / 100.)
        return qt.QBrush(color, self.get('style').qtStyle())

    def makeQBrushWHide(self, painter):
        """Make a brush, taking account of hide attribute."""
        if self.hide or self.transparency == 100:
            return qt.QBrush()
        else:
            return self.makeQBrush(painter)

class BrushExtended(Settings):
    '''Extended brush style.'''

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.Color(
            'color', 'foreground',
            descr=_('Fill colour'),
            usertext=_('Color')) )
        self.add( setting.FillStyleExtended(
            'style', 'solid',
            descr=_('Fill style'),
            usertext=_('Style')) )

        self.add( setting.Bool(
            'hide', False,
            descr=_('Hide the fill'),
            usertext=_('Hide')) )

        self.add( setting.Int(
            'transparency', 0,
            descr=_('Transparency percentage'),
            usertext=_('Transparency'),
            minval=0,
            maxval=100 ) )
        self.add( setting.DistancePt(
            'linewidth', '0.5pt',
            descr=_('Width of hatch or pattern line'),
            usertext=_('Line width')) )
        self.add( setting.LineStyle(
            'linestyle', 'solid',
            descr=_('Hatch or pattern line style'),
            usertext=_('Line style')) )
        self.add( setting.DistancePt(
            'patternspacing', '5pt',
            descr=_('Hatch or pattern spacing'),
            usertext=_('Spacing')) )
        self.add( setting.Color(
            'backcolor', 'background',
            descr=_('Hatch or pattern background color'),
            usertext=_('Back color') ) )
        self.add( setting.Int(
            'backtransparency', 0,
            descr=_('Hatch or pattern background transparency percentage'),
            usertext=_('Back trans.'),
            minval=0,
            maxval=100 ) )
        self.add( setting.Bool(
            'backhide', True,
            descr=_('Hide hatch or pattern background'),
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

class PageBrush(BrushExtended):
    '''Fill used for back of page.'''

    def __init__(self, name, **args):
        BrushExtended.__init__(self, name, **args)

        self.get('color').newDefault('background')
        self.get('hide').newDefault(True)

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
            descr=_('Hide the filled region inside the error bars'),
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
            descr=_('Font name'),
            usertext=_('Font')) )
        self.add( setting.DistancePt(
            'size',
            setting.Reference('/StyleSheet/Font/size'),
            descr=_('Font size'),
            usertext=_('Size') ) )
        self.add( setting.Color(
            'color',
            setting.Reference('/StyleSheet/Font/color'),
            descr=_('Font color'),
            usertext=_('Color') ) )
        self.add( setting.FontStyle(
            'style',
            setting.Reference('/StyleSheet/Font/style'),
            'font',
            descr=_('Font style'),
            usertext=_('Style') ) )
        self.add( setting.Bool(
            'italic', False,
            descr=_('Italic font'),
            usertext=_('Italic') ) )
        self.add( setting.Bool(
            'bold', False,
            descr=_('Bold font'),
            usertext=_('Bold') ) )
        self.add( setting.Bool(
            'underline', False,
            descr=_('Underline font'),
            usertext=_('Underline') ) )
        self.add( setting.Bool(
            'hide', False,
            descr=_('Hide the text'),
            usertext=_('Hide')) )

    def copy(self):
        """Make copy of settings."""
        c = Settings.copy(self)
        c.defaultfamily = self.defaultfamily
        c.families = self.families
        return c

    def makeQFont(self, painthelper):
        '''Return a qt.QFont object corresponding to the settings.'''

        size = self.get('size').convertPts(painthelper)
        weight = qt.QFont.Weight.Normal
        if self.bold:
            weight = qt.QFont.Weight.Bold

        f = qt.QFont(self.font, int(size), weight, self.italic)
        if self.underline:
            f.setUnderline(True)
        style = self.style
        if style:
            f.setStyleName(style)
        else:
            f.setStyleHint(qt.QFont.StyleHint.Times)

        return f

    def makeQPen(self, painter):
        """ Return a qt.QPen object for the font pen """
        color = self.get('color').color(painter)
        return qt.QPen(color)

class PointLabel(Text):
    """For labelling points on plots."""

    def __init__(self, name, **args):
        Text.__init__(self, name, **args)

        self.add( setting.Float(
            'angle', 0.,
            descr=_('Angle of the labels in degrees'),
            usertext=_('Angle'),
            formatting=True), 0 )
        self.add( setting.AlignVert(
            'posnVert',
            'centre',
            descr=_('Vertical position of label'),
            usertext=_('Vert position'),
            formatting=True), 0 )
        self.add( setting.AlignHorz(
            'posnHorz',
            'right',
            descr=_('Horizontal position of label'),
            usertext=_('Horz position'),
            formatting=True), 0 )

class DataColor(Settings):
    """Settings for a coloring points using data values."""

    def __init__(self, name, dimensions=1):
        Settings.__init__(self, name, setnsmode='groupedsetting')
        self.add( setting.DatasetExtended(
            'points', '',
            descr=_('Use color value (0-1) in dataset'),
            dimensions=dimensions,
            usertext=_('Color')) )
        self.add( setting.Float(
            'min', 0.,
            descr=_('Minimum value of color dataset'),
            usertext=_('Min val') ))
        self.add( setting.Float(
            'max', 1.,
            descr=_('Maximum value of color dataset'),
            usertext=_('Max val') ))
        self.add( setting.Choice(
            'scaling',
            ['linear', 'sqrt', 'log', 'squared'],
            'linear',
            descr=_('Scaling to transform numbers to color'),
            usertext=_('Scaling')))

class Line3D(Settings):
    '''3d line properties.'''

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.Color(
            'color',
            setting.Reference('/StyleSheet/Line/color'),
            descr=_('Color of line'),
            usertext=_('Color')) )
        self.add( setting.Float(
            'width', 1.,
            minval=0., maxval=1000,
            descr=_('Width of line'),
            usertext=_('Width')) )
        self.add( setting.LineStyle(
            'style', 'solid',
            descr=_('Line style'),
            usertext=_('Style')) )
        self.add( setting.FloatSlider(
            'transparency', 0.,
            minval=0., maxval=100., tick=20., scale=1., step=5.,
            descr=_('Transparency percentage'),
            usertext=_('Transparency')) )
        self.add( setting.FloatSlider(
            'reflectivity', 0.,
            minval=0., maxval=100., tick=20., scale=1., step=5.,
            descr=_('Reflectivity percentage'),
            usertext=_('Reflectivity')) )
        self.add( setting.Bool(
            'hide', False,
            descr=_('Hide the line'),
            usertext=_('Hide')) )

    def makeLineProp(self, painter):
        """Construct line properties object for assigning to 3D object."""

        if self.hide:
            return None

        width = self.width
        if self.hide or width == 0.:
            style, dash = qt.Qt.PenStyle.NoPen, None
        else:
            style, dash = setting.LineStyle._linecnvt[self.style]

        col = self.get('color').color(painter)
        from ..helpers import threed
        prop = threed.LineProp(
            r=col.redF(), g=col.greenF(), b=col.blueF(),
            trans=self.transparency*0.01,
            refl=self.reflectivity*0.01,
            width=width, style=style)
        if dash is not None:
            prop.setDashPattern(threed.ValVector(dash))
        return prop

class Surface3D(Settings):
    '''3d surface properties.'''

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.Color(
            'color',
            'grey',
            descr=_('Color of surface'),
            usertext=_('Color')) )
        self.add( setting.FloatSlider(
            'transparency', 0.,
            minval=0., maxval=100., tick=20., scale=1., step=1.,
            descr=_('Transparency percentage'),
            usertext=_('Transparency')) )
        self.add( setting.FloatSlider(
            'reflectivity', 50.,
            minval=0., maxval=100., tick=20., scale=1., step=1.,
            descr=_('Reflectivity percentage'),
            usertext=_('Reflectivity')) )
        self.add( setting.Bool(
            'hide', False,
            descr=_('Hide surface'),
            usertext=_('Hide')) )

    def makeSurfaceProp(self, painter):
        """Properties to assign to surface."""
        if self.hide:
            return None

        col = self.get('color').color(painter)
        from ..helpers import threed
        return threed.SurfaceProp(
            r=col.redF(), g=col.greenF(), b=col.blueF(),
            trans=self.transparency*0.01, refl=self.reflectivity*0.01)

class Surface3DWColorMap(Surface3D):
    '''3d surface with color map setting.'''

    def __init__(self, name, **args):
        Surface3D.__init__(self, name, **args)

        self.add( setting.Colormap(
            'colorMap', 'grey',
            descr=_(
                'If color markers dataset is given, use this colormap '
                'instead of the fill color'),
            usertext=_('Color map'),
            formatting=True) )
        self.add( setting.Bool(
            'colorMapInvert', False,
            descr=_('Invert color map'),
            usertext=_('Invert map'),
            formatting=True) )

class Line3DWColorMap(Line3D):
    '''3d line with color map setting.'''

    def __init__(self, name, **args):
        Line3D.__init__(self, name, **args)

        self.add( setting.Colormap(
            'colorMap', 'grey',
            descr=_(
                'If color markers dataset is given, use this colormap '
                'instead of the fill color'),
            usertext=_('Color map'),
            formatting=True) )
        self.add( setting.Bool(
            'colorMapInvert', False,
            descr=_('Invert color map'),
            usertext=_('Invert map'),
            formatting=True) )

class LineGrid3D(Line3D):
    """Line for 3D with ability to hide horz/vert."""
    def __init__(self, name, **args):
        Line3D.__init__(self, name, **args)
        self.add( setting.Bool(
            'hidehorz', False,
            descr=_('Hide horizontal lines'),
            usertext=_('Hide horz.')) )
        self.add( setting.Bool(
            'hidevert', False,
            descr=_('Hide vertical lines'),
            usertext=_('Hide vert.')) )

class Lighting3D(Settings):
    '''Lighting options.'''

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.Bool(
            'enable', True,
            descr=_('Enable lighting'),
            usertext=_('Enable')) )
        self.add( setting.Color(
            'color', 'white',
            descr=_('Color of lighting'),
            usertext=_('Color')) )
        self.add( setting.FloatSlider(
            'intensity', 100.,
            minval=0., maxval=400., tick=100., scale=5., step=1.,
            descr=_('Light intensity percentage'),
            usertext=_('Intensity')) )
        self.add( setting.FloatSlider(
            'x', 0.,
            minval=-40, maxval=40, tick=10., scale=0.1, step=1.,
            descr=_('X coordinate of light (positive is right)'),
            usertext=_('X position')) )
        self.add( setting.FloatSlider(
            'y', 0.,
            minval=-40, maxval=40, tick=10., scale=0.1, step=1.,
            descr=_('Y coordinate of light (positive is below)'),
            usertext=_('Y position')) )
        self.add( setting.FloatSlider(
            'z', 0.,
            minval=-40, maxval=40, tick=10., scale=0.1, step=1.,
            descr=_('Z coordinate of light (positive is towards viewer)'),
            usertext=_('Z position')) )
