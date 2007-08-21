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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id$

"""Collections of predefined settings for common settings."""

import veusz.qtall as qt4

import setting
from settings import Settings, StyleSheet

from veusz.utils import formatNumber
from veusz.application import Application

class StylesheetLine(Settings):
    """Hold the properties of the default line."""
    def __init__(self):
        Settings.__init__(self, 'Line', pixmap='plotline',
                          descr='Default line style for document',
                          usertext='Line')
        self.add( setting.Distance('width', '0.5pt',
                                   descr='Default line width',
                                   usertext='Width') )
        self.add( setting.Color('color', 'black',
                               descr='Default line color',
                               usertext='Color') )
# register these properties with the stylesheet
StyleSheet.register(StylesheetLine())

class Line(Settings):
    '''For holding properities of a line.'''

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.Color('color', setting.Reference('/StyleSheet/Line/color'),
                                descr = 'Color of line',
                                usertext='Color') )
        self.add( setting.Distance('width', setting.Reference('/StyleSheet/Line/width'),
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
        '''Make a QPen from the description'''

        color = qt4.QColor(self.color)
        color.setAlphaF( (100-self.transparency) / 100.)
        width = self.get('width').convert(painter)
        style, dashpattern = setting.LineStyle._linecnvt[self.style]
        pen = qt4.QPen( color, width, style )

        if dashpattern:
            pen.setDashPattern(dashpattern)

        return pen
        
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

class MajorTick(Line):
    '''Major tick settings.'''

    def __init__(self, name, **args):
        Line.__init__(self, name, **args)
        self.add( setting.Distance( 'length', '6pt',
                                    descr = 'Length of ticks',
                                    usertext='Length') )
        self.add( setting.Int( 'number', 5,
                               descr = 'Number of major ticks to aim for',
                               usertext='Number') )
        self.add( setting.FloatList('manualTicks',
                                    [],
                                    descr = 'List of tick values'
                                    ' overriding defaults',
                                    usertext='Manual ticks') )

    def getLength(self, painter):
        '''Return tick length in painter coordinates'''
        
        return self.get('length').convert(painter)
    
class MinorTick(Line):
    '''Minor tick settings.'''

    def __init__(self, name, **args):
        Line.__init__(self, name, **args)
        self.add( setting.Distance( 'length', '3pt',
                                    descr = 'Length of ticks',
                                    usertext='Length') )
        self.add( setting.Int( 'number', 20,
                               descr = 'Number of minor ticks to aim for',
                               usertext='Number') )

    def getLength(self, painter):
        '''Return tick length in painter coordinates'''
        
        return self.get('length').convert(painter)
    
class GridLine(Line):
    '''Grid line settings.'''

    def __init__(self, name, **args):
        Line.__init__(self, name, **args)

        self.get('color').newDefault('grey')
        self.get('hide').newDefault(True)
        self.get('style').newDefault('dotted')

def _registerFontStyleSheet():
    """Get fonts, and register default with StyleSheet."""
    families = [ unicode(name) for name in qt4.QFontDatabase().families() ]
    
    default = None
    for i in ['Times New Roman', 'Bitstream Vera Serif', 'Times', 'Utopia',
              'Serif']:
        if i in families:
            default = unicode(i)
            break
            
    if default is None:
        print >>sys.stderr, "Warning: did not find a sensible default font. Choosing first font."    
        default = unicode(_fontfamilies[0])

    class StylesheetText(Settings):
        """Hold properties of default text font."""

        def __init__(self, defaultfamily, families):
            """Initialise with default font family and list of families."""
            Settings.__init__(self, 'Font', pixmap='axislabel',
                              descr='Default font for document',
                              usertext='Font')
            self.defaultfamily = defaultfamily
            self.families = families

            self.add( setting.ChoiceOrMore('font', families, default,
                                           descr='Font name', usertext='Font'))
            self.add( setting.Distance('size', '14pt',
                                       descr='Default font size', usertext='Size'))
            self.add( setting.Color('color', 'black', descr='Default font color',
                                    usertext='Color'))

        def copy(self):
            """Make copy of settings."""
            c = Settings.copy(self)
            c.defaultfamily = self.defaultfamily
            c.families = self.families
            return c

    StyleSheet.register(StylesheetText(default, families))
    Text.defaultfamily = default
    Text.families = families

Application.startupfunctions.append(_registerFontStyleSheet)

class Text(Settings):
    '''Text settings.'''

    # need to examine font table to see what's available
    # this is set on app startup
    defaultfamily = None
    families = None

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        self.add( setting.ChoiceOrMore('font', Text.families,
                                       setting.Reference('/StyleSheet/Font/font'),
                                       descr = 'Font name',
                                       usertext='Font') )
        self.add( setting.Distance('size', setting.Reference('/StyleSheet/Font/size'),
                  descr = 'Font size', usertext='Size' ) )
        self.add( setting.Color( 'color', setting.Reference('/StyleSheet/Font/color'),
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
        
class AxisLabel(Text):
    """For axis labels."""

    def __init__(self, name, **args):
        Text.__init__(self, name, **args)
        self.add( setting.Bool( 'atEdge', False,
                                descr = 'Place axis label close to edge'
                                ' of graph',
                                usertext='At edge') )
        self.add( setting.Bool( 'rotate', False,
                                descr = 'Rotate the label by 90 degrees',
                                usertext='Rotate') )

class TickLabel(Text):
    """For tick labels on axes."""

    def __init__(self, name, **args):
        Text.__init__(self, name, **args)
        self.add( setting.Bool( 'rotate', False,
                                descr = 'Rotate the label by 90 degrees',
                                usertext='Rotate') )
        self.add( setting.Str( 'format', '%Vg',
                               descr = 'Format of the tick labels',
                               usertext='Format') )

        self.add( setting.Float('scale', 1.,
                                descr='A scale factor to apply to the values '
                                'of the tick labels',
                                usertext='Scale') )

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
        self.add( setting.Choice('posnVert',
                                 ['top', 'centre', 'bottom'], 'centre',
                                 descr='Vertical position of label',
                                 usertext='Vert position',
                                 formatting=True), 0 )
        self.add( setting.Choice('posnHorz',
                                 ['left', 'centre', 'right'], 'right',
                                 descr="Horizontal position of label",
                                 usertext='Horz position',
                                 formatting=True), 0 )
