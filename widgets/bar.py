#    Copyright (C) 2009 Jeremy S. Sanders
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

"""For plotting bar graphs."""

from itertools import izip, repeat
import numpy as N

import veusz.qtall as qt4
import veusz.document as document
import veusz.setting as setting
import veusz.utils as utils

from plotters import GenericPlotter

class BarFill(setting.Settings):
    '''Filling of bars.'''
    def __init__(self, name, **args):
        setting.Settings.__init__(self, name, **args)
        self.add( setting.FillSet('fills', [('solid', 'grey', False)],
                                  descr = 'Fill styles for dataset bars',
                                  usertext='Fill styles') )

class BarLine(setting.Settings):
    '''Edges of bars.'''
    def __init__(self, name, **args):
        setting.Settings.__init__(self, name, **args)
        self.add( setting.LineSet('lines',
                                  [('solid', '0.5pt', 'black', False)],
                                  descr = 'Line styles for dataset bars', 
                                  usertext='Line styles') )

def extend1DArray(array, length, missing=0.):
    """Return array with length given (original if appropriate.
    Values are extended with value given."""

    if len(array) == length:
        return array
    retn = N.resize(array, length)
    retn[len(array):] = missing
    return retn

class BarPlotter(GenericPlotter):
    """Plot bar charts."""

    typename='bar'
    allowusercreation=True
    description='Plot bar charts'

    def __init__(self, parent, name=None):
        """Initialise bar chart."""
        GenericPlotter.__init__(self, parent, name=name)
        if type(self) == BarPlotter:
            self.readDefaults()

    @classmethod
    def addSettings(klass, s):
        """Construct list of settings."""
        GenericPlotter.addSettings(s)

        # get rid of default key setting
        s.remove('key')

        s.add( setting.Strings('keys', ('',),
                               descr='Key text for each dataset',
                               usertext='Key text'), 0)

        s.add( setting.DatasetOrStr('labels', '',
                                    descr='Dataset or string to label bars',
                                    usertext='Labels', datatype='text'), 5 )

        s.add( setting.Choice('mode', ('grouped', 'stacked'), 
                              'grouped', 
                              descr='Show datasets grouped '
                              'together or as a single bar', 
                              usertext='Mode'), 0)
        s.add( setting.Choice('direction', 
                              ('horizontal', 'vertical'), 'vertical', 
                              descr = 'Horizontal or vertical bar chart', 
                              usertext='Direction'), 0 )
        s.add( setting.Dataset('posn', '', 
                               descr = 'Dataset containing position of bars'
                               ' (optional)',
                               usertext='Positions'), 0 )
        s.add( setting.Datasets('lengths', ('y',),
                                descr = 'Datasets containing lengths of bars',
                                usertext='Lengths'), 0 )

        s.add( setting.Float('barfill', 0.75,
                             minval = 0., maxval = 1.,
                             descr = 'Filling fraction of bars'
                             ' (between 0 and 1)',
                             usertext='Bar fill',
                             formatting=True) )
        s.add( setting.Float('groupfill', 0.9,
                             minval = 0., maxval = 1.,
                             descr = 'Filling fraction of groups of bars'
                             ' (between 0 and 1)',
                             usertext='Group fill',
                             formatting=True) )

        s.add( setting.Choice('errorstyle', ('none', 'bar', 'barends'), 
                              'bar', 
                              descr='Error bar style to show', 
                              usertext='Error style',
                              formatting=True) )

        s.add(BarFill('BarFill', descr='Bar fill', usertext='Fill'),
              pixmap = 'settings_bgfill')
        s.add(BarLine('BarLine', descr='Bar line', usertext='Line'),
              pixmap = 'settings_border')

        s.add( setting.ErrorBarLine('ErrorBarLine',
                                    descr = 'Error bar line settings',
                                    usertext = 'Error bar line'),
               pixmap = 'settings_ploterrorline' )

    @property
    def userdescription(self):
        """User-friendly description."""

        s = self.settings
        return "lengths='%s', position='%s'" % (', '.join(s.lengths), 
                                                s.posn)

    def providesAxesDependency(self):
        """This widget provides range information about these axes."""
        s = self.settings
        return ( (s.xAxis, 'sx'), (s.yAxis, 'sy') )

    def getAxisLabels(self, direction):
        """Get labels for bar for appropriate axis."""
        s = self.settings
        if s.direction != direction:
            # if horizontal bars, want labels on vertical axis and vice versa
            doc = self.document

            labels = s.get('labels').getData(doc, checknull=True)
            positions = s.get('posn').getData(doc)
            if positions is None:
                lengths = s.get('lengths').getData(doc)
                if lengths is None:
                    return (None, None)
                p = N.arange( max([len(d.data) for d in lengths]) )+1.
            else:
                p = positions
            
            return (labels, p)

        else:
            return (None, None)

    def singleBarDataRange(self, datasets):
        """For single bars where multiple datasets are added,
        compute maximum range."""
        minv, maxv = 0., 0.
        for data in izip(*[ds.data for ds in datasets]):
            totpos = sum( [d for d in data if d > 0] )
            totneg = sum( [d for d in data if d < 0] )
            
            minv = min(minv, totneg)
            maxv = max(maxv, totpos)
        return minv,  maxv

    def updateAxisRange(self, axis, depname, axrange):
        """Update axis range from data."""
        s = self.settings
        if ((s.direction == 'horizontal' and depname == 'sx') or
            (s.direction == 'vertical' and depname == 'sy')):
                # update from lengths
                data = s.get('lengths').getData(self.document)
                if s.mode == 'grouped':
                    # update range from individual datasets
                    for d in data:
                        drange = d.getRange()
                        if drange is not None:
                            axrange[0] = min(axrange[0], drange[0])
                            axrange[1] = max(axrange[1], drange[1])
                else:
                    # update range from sum of datasets
                    minv, maxv = self.singleBarDataRange(data)
                    axrange[0] = min(axrange[0], minv)
                    axrange[1] = max(axrange[1], maxv)
        else:
            if s.posn:
                # use given positions
                data = s.get('posn').getData(self.document)
                if data:
                    drange = data.getRange()
                    if drange is not None:
                        axrange[0] = min(axrange[0], drange[0])
                        axrange[1] = max(axrange[1], drange[1])
            else:
                # count bars
                data = s.get('lengths').getData(self.document)
                if data:
                    maxlen = max([len(d) for d in data])
                    axrange[0] = min(1-0.5, axrange[0])
                    axrange[1] = max(maxlen+0.5,  axrange[1])

    def findBarPositions(self, lengths, positions, axes, posn):
        """Work out centres of bar / bar groups and maximum width."""

        ishorz = self.settings.direction == 'horizontal'

        if positions is None:
            p = N.arange( max([len(d.data) for d in lengths]) )+1.
        else:
            p = positions.data

        # work out positions of bars
        # get vertical axis if horz, and vice-versa
        axis = axes[ishorz]
        posns = axis.dataToPlotterCoords(posn, p)
        if len(posns) <= 1:
            if ishorz:
                maxwidth = posn[2]-posn[0]
            else:
                maxwidth = posn[3]-posn[1]
        else:
            maxwidth = N.nanmin(N.abs(posns[1:]-posns[:-1]))

        return posns,  maxwidth

    def calculateErrorBars(self, dataset, vals):
        """Get values for error bars."""
        minval = None
        maxval = None
        length = len(vals)
        if 'serr' in dataset:
            s = N.nan_to_num(dataset['serr'])
            minval = vals - s
            maxval = vals + s
        else:
            if 'nerr' in dataset:
                minval = vals + N.nan_to_num(dataset['nerr'])
            if 'perr' in dataset:
                maxval = vals + N.nan_to_num(dataset['perr'])
        return minval, maxval

    def drawErrorBars(self, painter, posns, barwidth,
                      yvals, dataset, axes, widgetposn):
        """Draw (optional) error bars on bars."""
        s = self.settings
        if s.errorstyle == 'none':
            return

        minval, maxval = self.calculateErrorBars(dataset, yvals)
        if minval is None and maxval is None:
            return

        # handle one sided errors
        if minval is None:
            minval = yvals
        if maxval is None:
            maxval = yvals

        # convert errors to coordinates
        ishorz = s.direction == 'horizontal'
        mincoord = axes[not ishorz].dataToPlotterCoords(widgetposn, minval)
        mincoord = N.clip(mincoord, -32767, 32767)
        maxcoord = axes[not ishorz].dataToPlotterCoords(widgetposn, maxval)
        maxcoord = N.clip(maxcoord, -32767, 32767)

        # draw error bars
        painter.setPen( self.settings.ErrorBarLine.makeQPenWHide(painter) )
        w = barwidth*0.25
        if ishorz:
            utils.plotLinesToPainter(painter, mincoord, posns,
                                     maxcoord, posns)
            if s.errorstyle == 'barends':
                utils.plotLinesToPainter(painter, mincoord, posns-w,
                                         mincoord, posns+w)
                utils.plotLinesToPainter(painter, maxcoord, posns-w,
                                         maxcoord, posns+w)
        else:
            utils.plotLinesToPainter(painter, posns, mincoord,
                                     posns, maxcoord)
            if s.errorstyle == 'barends':
                utils.plotLinesToPainter(painter, posns-w, mincoord,
                                         posns+w, mincoord)
                utils.plotLinesToPainter(painter, posns-w, maxcoord,
                                         posns+w, maxcoord)

    def barDrawGroup(self, painter, posns, maxwidth, dsvals,
                     axes, widgetposn, clip):
        """Draw groups of bars."""

        s = self.settings

        # calculate bar and group widths
        numgroups = len(dsvals)
        groupwidth = maxwidth
        usablewidth = groupwidth * s.groupfill
        bardelta = usablewidth / float(numgroups)
        barwidth = bardelta * s.barfill

        ishorz = s.direction == 'horizontal'

        # bar extends from these coordinates
        zeropt = axes[not ishorz].dataToPlotterCoords(widgetposn,
                                                      N.array([0.]))

        for dsnum, dataset in enumerate(dsvals):
            # set correct attributes for datasets
            painter.setBrush( s.BarFill.get('fills').makeBrush(dsnum) )
            painter.setPen( s.BarLine.get('lines').makePen(painter, dsnum) )
            
            # convert bar length to plotter coords
            lengthcoord = axes[not ishorz].dataToPlotterCoords(
                widgetposn, dataset['data'])
 
            # these are the coordinates perpendicular to the bar
            posns1 = posns + (-usablewidth*0.5 + bardelta*dsnum +
                              (bardelta-barwidth)*0.5)
            posns2 = posns1 + barwidth

            if ishorz:
                p = (zeropt + N.zeros(posns1.shape), posns1,
                     lengthcoord, posns2)
            else:
                p = (posns1, zeropt + N.zeros(posns2.shape),
                     posns2, lengthcoord)

            # iterate over coordinates to plot bars
            utils.plotBoxesToPainter(painter, p[0], p[1], p[2], p[3], clip)

            # draw error bars
            self.drawErrorBars(painter, posns2-barwidth*0.5, barwidth,
                               dataset['data'], dataset,
                               axes, widgetposn)

    def barDrawStacked(self, painter, posns, maxwidth, dsvals,
                       axes, widgetposn, clip):
        """Draw each dataset in a single bar."""

        s = self.settings

        # get positions of groups of bars
        barwidth = maxwidth * s.barfill

        ishorz = s.direction == 'horizontal'

        # keep track of last most negative or most positive values in bars
        poslen = len(posns)
        lastneg = N.zeros(poslen)
        lastpos = N.zeros(poslen)

        # keep track of bars for error bars
        barvals = []
        for dsnum, data in enumerate(dsvals):
            # set correct attributes for datasets
            painter.setBrush( s.BarFill.get('fills').makeBrush(dsnum) )
            painter.setPen( s.BarLine.get('lines').makePen(painter, dsnum) )
            
            # add on value to last value in correct direction
            data = data['data']
            last = N.where(data < 0., lastneg, lastpos)
            new = N.where(data < 0., lastneg+data, lastpos+data)

            # work out maximum extents for next time
            lastneg = N.min( N.vstack((lastneg, new)), axis=0 )
            lastpos = N.max( N.vstack((lastpos, new)), axis=0 )

            # convert values to plotter coordinates
            lastplt = axes[not ishorz].dataToPlotterCoords(
                widgetposn, last)
            newplt = axes[not ishorz].dataToPlotterCoords(
                widgetposn, new)

            # positions of bar perpendicular to bar direction
            posns1 = posns - barwidth*0.5
            posns2 = posns1 + barwidth 

            # we iterate over each of these coordinates
            if ishorz:
                p = (lastplt, posns1, newplt, posns2)
            else:
                p = (posns1, lastplt, posns2, newplt)

            # draw bars
            utils.plotBoxesToPainter(painter, p[0], p[1], p[2], p[3], clip)

            barvals.append(new)

        for barval, dsval in izip(barvals, dsvals):
            # draw error bars
            self.drawErrorBars(painter, posns, barwidth,
                               barval, dsval,
                               axes, widgetposn)

    def getNumberKeys(self):
        """Return maximum number of keys."""
        lengths = self.settings.get('lengths').getData(self.document)
        if not lengths:
            return 0
        return min( len([k for k in self.settings.keys if k]), len(lengths) )

    def getKeyText(self, number):
        """Get key entry."""
        return [k for k in self.settings.keys if k][number]

    def drawKeySymbol(self, number, painter, x, y, width, height):
        """Draw a fill rectangle for key entry."""

        s = self.settings
        painter.setBrush( s.BarFill.get('fills').makeBrush(number) )
        painter.setPen( s.BarLine.get('lines').makePen(painter, number) )
        painter.drawRect( qt4.QRectF(qt4.QPointF(x, y+height*0.1),
                                     qt4.QPointF(x+width, y+height*0.8)) )


    def draw(self, parentposn, painter, outerbounds=None):
        """Plot the data on a plotter."""

        widgetposn = GenericPlotter.draw(self, parentposn, painter,
                                         outerbounds=outerbounds)
        s = self.settings

        # exit if hidden
        if s.hide:
            return

        # get data
        doc = self.document
        positions = s.get('posn').getData(doc)
        lengths = s.get('lengths').getData(doc)
        if not lengths:
            return

        # get axes widgets
        axes = self.parent.getAxes( (s.xAxis, s.yAxis) )

        # return if there are no proper axes
        if ( None in axes or
             axes[0].settings.direction != 'horizontal' or
             axes[1].settings.direction != 'vertical' ):
            return

        # where the bars are to be placed horizontally
        barposns, maxwidth = self.findBarPositions(lengths, positions,
                                                   axes, widgetposn)
        
        # only use finite positions
        origposnlen = len(barposns)
        validposn = N.isfinite(barposns)
        barposns = barposns[validposn]

        # this is a bit rubbish - we take the datasets and
        # make sure they have the same lengths as posns and remove NaNs
        # Datasets are stored as dicts
        dsvals = []
        for dataset in lengths:
            vals = {}
            for key in ('data', 'serr', 'nerr', 'perr'):
                v = getattr(dataset, key)
                if v is not None:
                    vals[key] = extend1DArray(N.nan_to_num(v),
                                              origposnlen)[validposn]
            dsvals.append(vals)

        # clip data within bounds of plotter
        painter.beginPaintingWidget(self, widgetposn)
        painter.save()
        clip = self.clipAxesBounds(painter, axes, widgetposn)

        # actually do the drawing
        fn = {'stacked': self.barDrawStacked,
              'grouped': self.barDrawGroup}[s.mode]
        fn(painter, barposns, maxwidth, dsvals, axes, widgetposn, clip)

        painter.restore()
        painter.endPaintingWidget()

# allow the factory to instantiate a bar plotter
document.thefactory.register( BarPlotter )
