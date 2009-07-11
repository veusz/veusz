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

# $Id$

"""Collections of predefined settings for common settings."""

import veusz.qtall as qt4

import setting
from settings import Settings

class Line(Settings):
    '''For holding properities of a line.'''

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.Color('color',
                                setting.Reference('/StyleSheet/Line/color'),
                                descr = 'Color of line',
                                usertext='Color') )
        self.add( setting.Distance('width',
                                   setting.Reference('/StyleSheet/Line/width'),
                                   descr = 'Width of line',
                                   usertext='Width') )
        self.add( setting.LineStyle('style', 'solid',
                                    descr = 'Line style',
                                    usertext='Style') )
        self.add( setting.Int( 'transparency', 0,
                               descr = 'Transparency percentage',
                               usertext = 'Transparency',
                               minval = 0,
                               maxval = 100 ) )
        self.add( setting.Bool('hide', False,
                               descr = 'Hide the line',
                               usertext='Hide') )
        
    def makeQPen(self, painter):
        '''Make a QPen from the description.
        This currently ignores the hide attribute
        '''

        color = qt4.QColor(self.color)
        color.setAlphaF( (100-self.transparency) / 100.)
        width = self.get('width').convert(painter)
        style, dashpattern = setting.LineStyle._linecnvt[self.style]
        pen = qt4.QPen( color, width, style )

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

        self.add( setting.Choice('steps',
                                 ['off', 'left', 'centre', 'right'], 'off',
                                 descr='Plot horizontal steps '
                                 'instead of a line',
                                 usertext='Steps'), 0 )

class ErrorBarLine(Line):
    '''A line style for error bar plotting.'''

    def __init__(self, name, **args):
        Line.__init__(self, name, **args)

        self.add( setting.Bool('hideHorz', False,
                               descr = 'Hide horizontal errors',
                               usertext='Hide horz.') )
        self.add( setting.Bool('hideVert', False,
                               descr = 'Hide vertical errors',
                               usertext='Hide vert.') )

class Brush(Settings):
    '''Settings of a fill.'''

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.Color( 'color', 'black',
                                 descr = 'Fill colour',
                                 usertext='Color') )
        self.add( setting.FillStyle( 'style', 'solid',
                                     descr = 'Fill style',
                                     usertext='Style') )
        self.add( setting.Int( 'transparency', 0,
                               descr = 'Transparency percentage',
                               usertext = 'Transparency',
                               minval = 0,
                               maxval = 100 ) )
        self.add( setting.Bool( 'hide', False,
                                descr = 'Hide the fill',
                                usertext='Hide') )
        
    def makeQBrush(self):
        '''Make a qbrush from the settings.'''

        color = qt4.QColor(self.color)
        color.setAlphaF( (100-self.transparency) / 100.)
        return qt4.QBrush( color, self.get('style').qtStyle() )
    
    def makeQBrushWHide(self):
        """Make a brush, taking account of hide attribute."""
        if self.hide:
            return qt4.QBrush()
        else:
            return self.makeQBrush()

class KeyBrush(Brush):
    '''Fill used for back of key.'''

    def __init__(self, name, **args):
        Brush.__init__(self, name, **args)

        self.get('color').newDefault('white')

class GraphBrush(Brush):
    '''Fill used for back of graph.'''

    def __init__(self, name, **args):
        Brush.__init__(self, name, **args)

        self.get('color').newDefault('white')

class PlotterFill(Brush):
    '''Filling used for filling on plotters.'''

    def __init__(self, name, **args):
        Brush.__init__(self, name, **args)

        self.get('hide').newDefault(True)

class PointFill(Brush):
    '''Filling used for filling above/below line or inside error region for xy-point
    plotters.
    '''

    def __init__(self, name, **args):
        Brush.__init__(self, name, **args)

        hide = self.get('hide')
        hide.newDefault(True)
        hide.usertext = 'Hide edge fill'
        hide.descr = 'Hide the filled region to the edge of the plot'
        self.get('color').newDefault('grey')

        self.add( setting.Bool( 'hideerror', False,
                                descr = 'Hide the filled region inside the error bars',
                                usertext='Hide error fill') )

class ShapeFill(Brush):
    '''Filling used for filling shapes.'''

    def __init__(self, name, **args):
        Brush.__init__(self, name, **args)

        self.get('hide').newDefault(True)
        self.get('color').newDefault('white')

class Text(Settings):
    '''Text settings.'''

    # need to examine font table to see what's available
    # this is set on app startup
    defaultfamily = None
    families = None

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.FontFamily('font',
                                     setting.Reference('/StyleSheet/Font/font'),
                                     descr = 'Font name',
                                     usertext='Font') )
        self.add( setting.Distance('size',
                                   setting.Reference('/StyleSheet/Font/size'),
                                   descr = 'Font size', usertext='Size' ) )
        self.add( setting.Color( 'color',
                                 setting.Reference('/StyleSheet/Font/color'),
                                 descr = 'Font color', usertext='Color' ) )
        self.add( setting.Bool( 'italic', False,
                                descr = 'Italic font', usertext='Italic' ) )
        self.add( setting.Bool( 'bold', False,
                                descr = 'Bold font', usertext='Bold' ) )
        self.add( setting.Bool( 'underline', False,
                                descr = 'Underline font', usertext='Underline' ) )
        self.add( setting.Bool( 'hide', False,
                                descr = 'Hide the text', usertext='Hide') )

    def copy(self):
        """Make copy of settings."""
        c = Settings.copy(self)
        c.defaultfamily = self.defaultfamily
        c.families = self.families
        return c

    def makeQFont(self, painter):
        '''Return a qt4.QFont object corresponding to the settings.'''
        
        size = self.get('size').convertPts(painter)
        weight = qt4.QFont.Normal
        if self.bold:
            weight = qt4.QFont.Bold

        f = qt4.QFont(self.font, size,  weight, self.italic)
        if self.underline:
            f.setUnderline(True)
        f.setStyleHint( qt4.QFont.Times, qt4.QFont.PreferDevice )

        return f

    def makeQPen(self):
        """ Return a qt4.QPen object for the font pen """
        return qt4.QPen(qt4.QColor(self.color))
        

class ContourLabel(Text):
    """For tick labels on axes."""

    def __init__(self, name, **args):
        Text.__init__(self, name, **args)
        self.add( setting.Str( 'format', '%Vg',
                               descr = 'Format of the tick labels',
                               usertext='Format') )
        self.add( setting.Float('scale', 1.,
                                descr='A scale factor to apply to the values '
                                'of the tick labels',
                                usertext='Scale') )

        self.get('hide').newDefault(True)

class PointLabel(Text):
    """For labelling points on plots."""

    def __init__(self, name, **args):
        Text.__init__(self, name, **args)
        
        self.add( setting.Float('angle', 0.,
                                descr='Angle of the labels in degrees',
                                usertext='Angle',
                                formatting=True), 0 )
        self.add( setting.AlignVert('posnVert',
                                    'centre',
                                    descr='Vertical position of label',
                                    usertext='Vert position',
                                    formatting=True), 0 )
        self.add( setting.AlignHorz('posnHorz',
                                    'right',
                                    descr="Horizontal position of label",
                                    usertext='Horz position',
                                    formatting=True), 0 )
