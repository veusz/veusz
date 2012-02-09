"""Paint hatched fills."""

import numpy as N
import veusz.qtall as qt4

try:
    from veusz.helpers.qtloops import plotLinesToPainter, polygonClip
except ImportError:
    from slowfuncs import plotLinesToPainter, polygonClip

def hatcher(painter, pen, painterpath, deltax, deltay):
    """Draw hatching on painter path given."""

    painter.save()
    painter.setPen(pen)

    painter.setClipPath(painterpath)

    # this is the bounding box of the path
    bb = painter.clipPath().boundingRect()

    # compute max number of steps
    numsteps = 0
    if deltax != 0:
        numsteps += bb.width() / abs(deltax) + 1
    if deltay != 0:
        numsteps += bb.height() / abs(deltay) + 1
    if numsteps == 0:
        raise ValueError, "deltax or deltay must be non-zero"

    # initial point inside the box
    # we want to make sure shifted regions lines in the same places
    if deltax == 0:
        startx = bb.left()
    else:
        startx = ( int(bb.left() / deltax) + 1 )*deltax
    if deltay == 0:
        starty = bb.bottom()
    else:
        starty = ( int(bb.bottom() / deltay) + 1 )*deltay

    # normalise line vector
    mag = N.sqrt( deltax**2 + deltay**2 )
    linedx, linedy = -deltay/mag, deltax/mag
    scale = max( bb.width(), bb.height() ) * 4

    # construct points along lines
    # this is scales to ensure the lines are bigger than the box
    idx = N.arange(-numsteps, numsteps)
    x = idx*deltax + startx
    y = idx*deltay + starty
    x1 = x - scale*linedx
    x2 = x + scale*linedx
    y1 = y - scale*linedy
    y2 = y + scale*linedy

    # plot lines, bounding to bb
    plotLinesToPainter(painter, x1, y1, x2, y2, bb)

    painter.restore()

_hatchmap = {
    'horizontal': ((0., 1.), ),
    'vertical': ((1., 0.), ),
    'cross': ( (0., 1.), (1., 0.), ),
    'forward diagonals': ( (0.7071, -0.7071), ),
    'backward diagonals': ( (0.7071, 0.7071), ),
    'diagonal cross': ( (0.7071, 0.7071), (0.7071, -0.7071), ),
}

def brushExtFillPath(painter, extbrush, path):
    """Use an BrushExtended settings object to fill a path on painter."""

    if extbrush.hide:
        return
    
    if extbrush.style in _hatchmap:
        # fill with hatching

        color = qt4.QColor(extbrush.color)
        color.setAlphaF( (100-extbrush.transparency) / 100.)
        width = extbrush.get('linewidth').convert(painter)
        style, dashpattern = extbrush.get('linestyle')._linecnvt[
            extbrush.linestyle]
        pen = qt4.QPen(color, width, style)

        if dashpattern:
            pen.setDashPattern(dashpattern)

        # do hatching with spacing
        spacing = extbrush.get('patternspacing').convert(painter)

        # iterate over each hatch
        for dx, dy in _hatchmap[extbrush.style]:
            hatcher(painter, pen, path, dx*spacing, dy*spacing)
    else:
        # standard fill: use Qt styles for painting
        color = qt4.QColor(extbrush.color)
        color.setAlphaF( (100-extbrush.transparency) / 100.)
        brush = qt4.QBrush( color, extbrush.get('style').qtStyle() )
        painter.fillPath(path, brush)

def brushExtFillPolygon(painter, extbrush, cliprect, polygon):
    """Fill a polygon with an extended brush."""
    clipped = qt4.QPolygonF()
    polygonClip(polygon, cliprect, clipped)
    path = qt4.QPainterPath()
    path.addPolygon(clipped)
    brushExtFillPath(painter, extbrush, path)
