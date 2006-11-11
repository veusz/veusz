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
from settings import Settings
from stylesheet import StyleSheet

from veusz.utils import formatNumber
from veusz.application import Application

StyleSheet.register('Line', setting.Distance('width', '0.5pt',
                                             descr='Default line width',
                                             usertext='Width'))
StyleSheet.register('Line', setting.Color('color', 'black',
                                          descr='Default line color',
                                          usertext='Color'))
StyleSheet.setPixmap('Line', 'plotline')

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
        self.add( setting.Bool('hide', False,
                               descr = 'Hide the line',
                               usertext='Hide') )
        
    def makeQPen(self, painter):
        '''Make a QPen from the description'''

        return qt4.QPen( qt4.QColor(self.color),
                         self.get('width').convert(painter),
                         self.get('style').qtStyle() )
    
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
        self.add( setting.Bool( 'hide', False,
                                descr = 'Hide the fill',
                                usertext='Hide') )
        
    def makeQBrush(self):
        '''Make a qbrush from the settings.'''

        return qt4.QBrush( qt4.QColor(self.color),
                           self.get('style').qtStyle() )
    
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

class _FontList(object):
    """A wrapped list class to interogate the list of fonts on usage.
    
    This is needed because we can't get the list of fonts until the QApplication has started.
    This class looks like a readonly list
    """
    
    def __init__(self):
        self.vals = None
    
    def __len__(self):
        if self.vals is None:
            self._getFonts()
        return len(self.vals)
    
    def __getitem__(self, key):
        if self.vals is None:
            self._getFonts()
        return self.vals[key]

    def __iter__(self):
        if self.vals is None:
            self._getFonts()
        return self.vals.__iter__()
    
    def _getFonts(self):
        """Construct list of fonts from Qt."""
        self.vals = [ unicode(name) for name in qt4.QFontDatabase().families() ]

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
    
    Text.defaultfamily = default
    Text.families = families
    StyleSheet.register('Font', setting.ChoiceOrMore('font', families, default,
                                                     descr='Font name', usertext='Font'))
    StyleSheet.register('Font', setting.Distance('size', '14pt',
                                                 descr='Default font size', usertext='Size'))
    StyleSheet.register('Font', setting.Color('color', 'black', descr='Default font color',
                                              usertext='Color'))
    StyleSheet.setPixmap('Font', 'axislabel')

Application.startupfunctions.append(_registerFontStyleSheet)

class Text(Settings):
    '''Text settings.'''

    # need to examine font table to see what's available
    defaultfamily = None
    families = None

    def __init__(self, name, **args):
        Settings.__init__(self, name, **args)

        if Text.defaultfamily == '':
            Text._getDefaultFamily()

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

    def _getFontFamilies():
        '''Make list of font families available.'''
        Text.families=[ unicode(name)
                        for name in qt4.QFontDatabase().families() ]
        
    _getFontFamilies = staticmethod(_getFontFamilies)

    def _getDefaultFamily():
        '''Choose a default font family. We check through a list until we
        get a sensible default.'''

        if not Text.families:
            Text._getFontFamilies()

        for i in ['Times New Roman', 'Bitstream Vera Serif', 'Times', 'Utopia',
                  'Serif']:
            if i in Text.families:
                Text.defaultfamily = i
                return

        Text.defaultfamily = Text.families[0]
        raise RuntimeError('Could not identify sensible default font for Veusz'
                           '. Please report this bug if you have any fonts '
                           'installed')

    _getDefaultFamily = staticmethod(_getDefaultFamily)
            
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

    def __init__(self, name, **args):
        Text.__init__(self, name, **args)
        self.add( setting.Bool( 'rotate', False,
                                descr = 'Rotate the label by 90 degrees',
                                usertext='Rotate') )
        self.add( setting.Str( 'format', 'g*',
                               descr = 'Format of the tick labels',
                               usertext='Format') )

        self.add( setting.Float('scale', 1.,
                                descr='A scale factor to apply to the values '
                                'of the tick labels',
                                usertext='Scale') )

