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

import qt

import settings
import setting
from utils import formatNumber

colors = [ 'black', 'red', 'green', 'blue',
           'cyan', 'magenta', 'yellow',
           'grey', 'darkred', 'darkgreen', 'darkblue',
           'darkcyan', 'darkmagenta', 'darkyellow' ]

linestyles = ['solid', 'dashed', 'dotted',
              'dash-dot', 'dash-dot-dot' ]

convertline = { 'solid': qt.Qt.SolidLine, 'dashed': qt.Qt.DashLine,
                'dotted': qt.Qt.DotLine, 'dash-dot': qt.Qt.DashDotLine,
                'dash-dot-dot': qt.Qt.DashDotDotLine }

fillstyles = [ 'solid', 'horizontal', 'vertical', 'cross',
               'forward diagonals', 'backward diagonals', 'diagonal cross',
               '94% dense', '88% dense', '63% dense', '50% dense',
               '37% dense', '12% dense', '6% dense' ]

convertfill = { 'solid': qt.Qt.SolidPattern, 'horizontal': qt.Qt.HorPattern,
                'vertical': qt.Qt.VerPattern, 'cross': qt.Qt.CrossPattern,
                'forward diagonals': qt.Qt.FDiagPattern,
                'backward diagonals': qt.Qt.BDiagPattern,
                'diagonal cross': qt.Qt.DiagCrossPattern,
                '94% dense': qt.Qt.Dense1Pattern,
                '88% dense': qt.Qt.Dense2Pattern,
                '63% dense': qt.Qt.Dense3Pattern,
                '50% dense': qt.Qt.Dense4Pattern,
                '37% dense': qt.Qt.Dense5Pattern,
                '12% dense': qt.Qt.Dense6Pattern,
                '6% dense': qt.Qt.Dense7Pattern }

class Line(settings.Settings):
    '''For holding properities of a line.'''

    def __init__(self, name, descr=''):
        settings.Settings.__init__(self, name, descr=descr)

        self.add( setting.ChoiceOrMore('color', colors, 'black',
                                       descr = 'Color of line') )
        self.add( setting.Distance('width', '0.5pt',
                                   descr = 'Width of line') )
        self.add( setting.Choice('style', linestyles, 'solid',
                                 descr = 'Line style') )
        self.add( setting.Bool('hide', False,
                               descr = 'Hide the line') )
        
    def makeQPen(self, painter):
        '''Make a QPen from the description'''

        return qt.QPen( qt.QColor(self.color),
                        self.get('width').convert(painter),
                        convertline[self.style] )

class XYPlotLine(Line):
    '''A plot line for plotting data, allowing histogram-steps
    to be plotted.'''
    
    def __init__(self, name, descr=''):
        Line.__init__(self, name, descr=descr)

        self.add( setting.Choice('steps',
                                 ['off', 'left', 'centre', 'right'], 'off',
                                 descr='Plot horizontal steps '
                                 'instead of a line'), 0 )

class Brush(settings.Settings):
    '''Settings of a fill.'''

    def __init__(self, name, descr=''):
        settings.Settings.__init__(self, name, descr=descr)

        self.add( setting.ChoiceOrMore( 'color', colors, 'black',
                                        descr = 'Fill colour' ) )
        self.add( setting.Choice( 'style', fillstyles, 'solid',
                                  descr = 'Fill style' ) )
        self.add( setting.Bool( 'hide', False,
                                descr = 'Hide the fill') )
        
    def makeQBrush(self):
        '''Make a qbrush from the settings.'''

        return qt.QBrush( qt.QColor(self.color),
                          convertfill[self.style] )
    
class KeyBrush(Brush):
    '''Fill used for back of key.'''

    def __init__(self, name, descr=''):
        Brush.__init__(self, name, descr)

        self.get('color').newDefault('white')

class GraphBrush(Brush):
    '''Fill used for back of graph.'''

    def __init__(self, name, descr=''):
        Brush.__init__(self, name, descr)

        self.get('color').newDefault('white')

class PlotterFill(Brush):
    '''Filling used for filling on plotters.'''

    def __init__(self, name, descr=''):
        Brush.__init__(self, name, descr)

        self.get('hide').newDefault(True)

class MajorTick(Line):
    '''Major tick settings.'''

    def __init__(self, name, descr=''):
        Line.__init__(self, name, descr)
        self.add( setting.Distance( 'length', '6pt',
                                    descr = 'Length of ticks' ) )
        self.add( setting.Int( 'number', 5,
                               descr = 'Number of major ticks to aim for' ) )

    def getLength(self, painter):
        '''Return tick length in painter coordinates'''
        
        return self.get('length').convert(painter)
    
class MinorTick(Line):
    '''Minor tick settings.'''

    def __init__(self, name, descr=''):
        Line.__init__(self, name, descr)
        self.add( setting.Distance( 'length', '3pt',
                                    descr = 'Length of ticks' ) )
        self.add( setting.Int( 'number', 20,
                               descr = 'Number of minor ticks to aim for' ) )

    def getLength(self, painter):
        '''Return tick length in painter coordinates'''
        
        return self.get('length').convert(painter)
    
class GridLine(Line):
    '''Grid line settings.'''

    def __init__(self, name, descr=''):
        Line.__init__(self, name, descr)

        self.get('color').newDefault('grey')
        self.get('hide').newDefault(True)
        self.get('style').newDefault('dotted')

class Text(settings.Settings):
    '''Text settings.'''

    # need to examine font table to see what's available
    defaultfamily=''
    families = []

    def __init__(self, name, descr = ''):
        settings.Settings.__init__(self, name, descr=descr)

        if Text.defaultfamily == '':
            Text._getDefaultFamily()

        self.add( setting.ChoiceOrMore('font', Text.families,
                                       Text.defaultfamily,
                                       descr = 'Font name' ) )
        self.add( setting.Distance('size', '14pt',
                  descr = 'Font size' ) )
        self.add( setting.ChoiceOrMore( 'color', colors, 'black',
                                        descr = 'Font color' ) )
        self.add( setting.Bool( 'italic', False,
                                descr = 'Italic font' ) )
        self.add( setting.Bool( 'bold', False,
                                descr = 'Bold font' ) )
        self.add( setting.Bool( 'underline', False,
                                descr = 'Underline font' ) )
        self.add( setting.Bool( 'hide', False,
                                descr = 'Hide the text') )

    def _getFontFamilies():
        '''Make list of font families available.'''
        Text.families=[ unicode(name)
                        for name in qt.QFontDatabase().families() ]
        
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
        '''Return a qt.QFont object corresponding to the settings.'''
        
        size = self.get('size').convertPts(painter)
        weight = qt.QFont.Normal
        if self.bold: weight = qt.QFont.Bold

        f = qt.QFont(self.font, size,  weight, self.italic)
        if self.underline: f.setUnderline(1)
        f.setStyleHint( qt.QFont.Times, qt.QFont.PreferDevice )

        return f

    def makeQPen(self):
        """ Return a qt.QPen object for the font pen """
        return qt.QPen(qt.QColor(self.color))
        
class AxisLabel(Text):

    def __init__(self, name, descr = ''):
        Text.__init__(self, name, descr=descr)
        self.add( setting.Bool( 'atEdge', False,
                                descr = 'Place axis label close to edge'
                                ' of graph') )
        self.add( setting.Bool( 'rotate', False,
                                descr = 'Rotate the label by 90 degrees' ) )

class TickLabel(Text):

    def __init__(self, name, descr = ''):
        Text.__init__(self, name, descr=descr)
        self.add( setting.Bool( 'rotate', False,
                                descr = 'Rotate the label by 90 degrees' ) )
        self.add( setting.Str( 'format', 'g*',
                               descr = 'Format of the tick labels' ) )

    def formatNumber(self, num):
        '''Format the number according to the format.'''

        return formatNumber(num, self.format)
