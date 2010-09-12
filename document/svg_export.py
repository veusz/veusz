#    Copyright (C) 2010 Jeremy S. Sanders
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

"""A home-brewed SVG paint engine for doing svg with clipping
and exporting text as paths for WYSIWYG."""

import sys
import veusz.qtall as qt4

dpi = 90.
inch_mm = 25.4

def fltStr(v):
    """Change a float to a string, using a maximum number of decimal places
    but removing trailing zeros."""

    return ('%.2f' % round(v, 2)).rstrip('0').rstrip('.')

def createPath(path, scale):
    """Convert qt path to svg path.

    We use relative coordinates to make the file size smaller and help
    compression
    """
    p = []
    count = path.elementCount()
    i = 0
    ox, oy = 0, 0
    while i < count:
        e = path.elementAt(i)
        nx, ny = e.x*scale, e.y*scale
        if e.type == qt4.QPainterPath.MoveToElement:
            p.append( 'm%s,%s' % (fltStr(nx-ox), fltStr(ny-oy)) )
            ox, oy = nx, ny
        elif e.type == qt4.QPainterPath.LineToElement:
            p.append( 'l%s,%s' % (fltStr(nx-ox), fltStr(ny-oy)) )
            ox, oy = nx, ny
        elif e.type == qt4.QPainterPath.CurveToElement:
            e1 = path.elementAt(i+1)
            e2 = path.elementAt(i+2)
            p.append( 'c%s,%s,%s,%s,%s,%s' % (
                    fltStr(nx-ox), fltStr(ny-oy),
                    fltStr(e1.x*scale-ox), fltStr(e1.y*scale-oy),
                    fltStr(e2.x*scale-ox), fltStr(e2.y*scale-oy)) )
            ox, oy = e2.x*scale, e2.y*scale
            i += 2
        else:
            assert False

        i += 1
    return ''.join(p)

class SVGPaintEngine(qt4.QPaintEngine):
    """Paint engine class for writing to svg files."""

    def __init__(self, width_in, height_in):
        """Create the class, using width and height as size of canvas
        in inches."""

        qt4.QPaintEngine.__init__(self,
                                  qt4.QPaintEngine.Antialiasing |
                                  qt4.QPaintEngine.PainterPaths |
                                  qt4.QPaintEngine.PrimitiveTransform |
                                  qt4.QPaintEngine.PaintOutsidePaintEvent |
                                  qt4.QPaintEngine.PixmapTransform |
                                  qt4.QPaintEngine.AlphaBlend
                                  )
        
        self.width = width_in
        self.height = height_in

    def begin(self, paintdevice):
        """Start painting."""
        self.device = paintdevice
        self.fileobj = paintdevice.fileobj

        self.pen = qt4.QPen()
        self.brush = qt4.QBrush()
        self.clippath = None
        self.clipnum = 0
        self.existingclips = {}
        self.matrix = qt4.QMatrix()
        
        self.lastclip = None
        self.laststate = None
        
        self.defs = []
        
        self.fileobj.write('''<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" 
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="%spx" height="%spx" version="1.1"
     xmlns="http://www.w3.org/2000/svg"
     xmlns:xlink="http://www.w3.org/1999/xlink">
<desc>Veusz output document</desc>
''' % (fltStr(self.width*dpi), fltStr(self.height*dpi)))

        # as defaults use qt defaults
        self.fileobj.write('<g stroke-linejoin="bevel" '
                           'stroke-linecap="square" '
                           'stroke="#000000" '
                           'fill-rule="evenodd">\n')

        return True

    def end(self):
        # close any existing groups
        if self.laststate is not None:
            self.fileobj.write('</g>\n')
        if self.lastclip is not None:
            self.fileobj.write('</g>\n')

        # close defaults
        self.fileobj.write('</g>\n')
            
        # write any defined objects
        if self.defs:
            self.fileobj.write('<defs>\n')
            for d in self.defs:
                self.fileobj.write(d)
            self.fileobj.write('</defs>\n')

        # end svg file
        self.fileobj.write('</svg>\n')
        return True

    def _updateClipPath(self, clippath, clipoperation):
        """Update clip path given state change."""
        if clipoperation == qt4.Qt.NoClip:
            self.clippath = None
        elif clipoperation == qt4.Qt.ReplaceClip:
            self.clippath = clippath
        elif clipoperation == qt4.Qt.IntersectClip:
            self.clippath = self.clippath.intersected(clippath)
        elif clipoperation == qt4.Qt.UniteClip:
            self.clippath = self.clippath.unite(clippath)
        else:
            print clipoperation

    def updateState(self, state):
        """Examine what has changed in state and call apropriate function."""
        self.updatedstate = True
        ss = state.state()
        if ss & qt4.QPaintEngine.DirtyPen:
            self.pen = state.pen()
        if ss & qt4.QPaintEngine.DirtyBrush:
            self.brush = state.brush()
        if ss & qt4.QPaintEngine.DirtyClipPath:
            self._updateClipPath(state.clipPath(), state.clipOperation())
        if ss & qt4.QPaintEngine.DirtyClipRegion:
            path = qt4.QPainterPath()
            path.addRegion(state.clipRegion())
            self._updateClipPath(path, state.clipOperation())
        if ss & qt4.QPaintEngine.DirtyTransform:
            self.matrix = state.matrix()

    def getSVGState(self):
        """Get state as svg group."""
        # these are the values to write into the attribute
        vals = {}

        # PEN UPDATE
        p = self.pen
        # - color
        color = p.color().name()
        if color != '#000000':
            vals['stroke'] = p.color().name()
        # - opacity
        if p.color().alphaF() != 1.:
            vals['stroke-opacity'] = '%.3g' % p.color().alphaF()
        # - join style
        if p.joinStyle() != qt4.Qt.BevelJoin:
            vals['stroke-linejoin'] = {
                qt4.Qt.MiterJoin: 'miter',
                qt4.Qt.SvgMiterJoin: 'miter',
                qt4.Qt.RoundJoin: 'round',
                qt4.Qt.BevelJoin: 'bevel'
                }[p.joinStyle()]
        # - cap style
        if p.capStyle() != qt4.Qt.SquareCap:
            vals['stroke-linecap'] = {
                qt4.Qt.FlatCap: 'butt',
                qt4.Qt.SquareCap: 'square',
                qt4.Qt.RoundCap: 'round'
                }[p.capStyle()]
        # - width
        w = p.widthF()
        # width 0 is device width for qt
        if w == 0.:
            w = 1
        vals['stroke-width'] = fltStr(w)

        # - line style
        if p.style() == qt4.Qt.NoPen:
            vals['stroke'] = 'none'
        elif p.style() not in (qt4.Qt.SolidLine, qt4.Qt.NoPen):
            # convert from pen width fractions to pts
            nums = [str(w*x) for x in p.dashPattern()]
            vals['stroke-dasharray'] = ','.join(nums)

        # BRUSH STYLES
        b = self.brush
        if b.style() == qt4.Qt.NoBrush:
            vals['fill'] = 'none'
        else:
            vals['fill'] = b.color().name()
        if b.color().alphaF() != 1.0:
            vals['fill-opacity'] = '%.3g' % b.color().alphaF()

        # MATRIX
        if not self.matrix.isIdentity():
            m = self.matrix
            dx, dy = m.dx(), m.dy()
            if (m.m11(), m.m12(), m.m21(), m.m22()) == (1., 0., 0., 1):
                vals['transform'] = 'translate(%s, %s)' % (fltStr(dx),
                                                           fltStr(dy))
            else:
                vals['transform'] = 'matrix(%.4g %.4g %.4g %.4g %s %s)' % (
                    m.m11(), m.m12(), m.m21(), m.m22(), fltStr(dx), fltStr(dy))

        # build up group for state
        t = ['<g']
        for name, val in vals.iteritems():
            t.append('%s="%s"' % (name, val))
        state = ' '.join(t)+'>\n'
        return state

    def getClipState(self):
        """Get SVG clipping state. This is in the form of an svg group"""

        if self.clippath is None:
            return None

        path = createPath(self.clippath, 1.0)

        if path in self.existingclips:
            url = 'url(#c%i)' % self.existingclips[path]
        else:
            clippath = '<clipPath id="c%i"><path d="%s"/></clipPath>\n' % (
                self.clipnum, path)

            self.defs.append(clippath)
            url = 'url(#c%i)' % self.clipnum
            self.existingclips[path] = self.clipnum
            self.clipnum += 1

        return '<g clip-path="%s">\n' % url

    def doStateUpdate(self):
        """Handle changes of state, starting and stopping
        groups to modify clipping and attributes."""
        if not self.updatedstate:
            return

        clipgrp = self.getClipState()
        state = self.getSVGState()

        if clipgrp == self.lastclip and state == self.laststate:
            # do nothing if everything is unchanged
            pass
        elif clipgrp == self.lastclip:
            # if state has only changed
            if self.laststate is not None:
                self.fileobj.write('</g>\n')
            self.fileobj.write(state)
            self.laststate = state
        else:
            # clip and state have changed
            if self.laststate is not None:
                self.fileobj.write('</g>\n')
            if self.lastclip is not None:
                self.fileobj.write('</g>\n')
            self.fileobj.write(clipgrp)
            self.fileobj.write(state)
            self.laststate = state
            self.lastclip = clipgrp

    def drawPath(self, path):
        """Draw a path on the output."""
        self.doStateUpdate()
        p = createPath(path, 1.)

        self.fileobj.write('<path d="%s"' % p)
        if path.fillRule() == qt4.Qt.WindingFill:
            self.fileobj.write(' fill-rule="nonzero"')
        self.fileobj.write('/>\n')

    def drawTextItem(self, pt, textitem):
        """Convert text to a path and draw it.
        """
        self.doStateUpdate()
        path = qt4.QPainterPath()
        path.addText(pt, textitem.font(), textitem.text())
        p = createPath(path, 1.)
        self.fileobj.write('<path d="%s" fill="%s" stroke="none" '
                           'fill-opacity="%.3g"/>\n' % (
                p, self.pen.color().name(), self.pen.color().alphaF() ))

    def drawLines(self, lines):
        """Draw multiple lines."""
        self.doStateUpdate()
        paths = []
        for line in lines:
            path = 'M%s,%sl%s,%s' % (
                fltStr(line.x1()), fltStr(line.y1()),
                fltStr(line.x2()-line.x1()),
                fltStr(line.y2()-line.y1()))
            paths.append(path)
        self.fileobj.write('<path d="%s"/>\n' % (''.join(paths)))

    def drawPolygon(self, points, mode):
        """Draw polygon on output."""
        self.doStateUpdate()
        pts = []
        for p in points:
            pts.append( '%s,%s' % (fltStr(p.x()), fltStr(p.y())) )

        if mode == qt4.QPaintEngine.PolylineMode:
            self.fileobj.write('<polyline fill="none" points="%s"/>\n' %
                               ' '.join(pts))

        else:
            self.fileobj.write('<polygon points="%s"' %
                               ' '.join(pts))
            if mode == qt4.Qt.WindingFill:
                self.fileobj.write(' fill-rule="nonzero"')
            self.fileobj.write('/>\n')

    def drawEllipse(self, rect):
        """Draw an ellipse to the svg file."""
        self.doStateUpdate()
        self.fileobj.write('<ellipse cx="%s" cy="%s" rx="%s" ry="%s"/>\n' %
                           (fltStr(rect.center().x()), fltStr(rect.center().y()),
                            fltStr(rect.width()*0.5), fltStr(rect.height()*0.5)))

    def drawPoints(self, points):
        """Draw points."""
        self.doStateUpdate()
        for pt in points:
            self.fileobj.write( '<line x1="%s" y1="%s" x2="%s" y2="%s" '
                                'stroke-linecap="round"/>\n' %
                                fltStr(pt.x()), fltStr(pt.y()),
                                fltStr(pt.x()), fltStr(pt.y()) )

    def drawPixmap(self, r, pixmap, sr):
        """Draw pixmap to file.

        This is converted to a PNG and embedded in the output
        """

        self.doStateUpdate()
        self.fileobj.write( '<image x="%s" y="%s" width="%s" height="%s" ' %
                            (fltStr(r.x()), fltStr(r.y()),
                             fltStr(r.width()), fltStr(r.height())) )
        data = qt4.QByteArray()
        buf = qt4.QBuffer(data)
        buf.open(qt4.QBuffer.ReadWrite)
        pixmap.save(buf, "PNG", 0)
        buf.close()

        self.fileobj.write('xlink:href="data:image/png;base64,')
        self.fileobj.write(data.toBase64())
        self.fileobj.write('" preserveAspectRatio="none"/>\n')

class SVGPaintDevice(qt4.QPaintDevice):
     """Paint device for SVG paint engine."""

     def __init__(self, fileobj, width_in, height_in):
          qt4.QPaintDevice.__init__(self)
          self.engine = SVGPaintEngine(width_in, height_in)
          self.fileobj = fileobj

     def paintEngine(self):
          return self.engine

     def width(self):
          return self.engine.width*dpi

     def widthMM(self):
          return int(self.width() * inch_mm)

     def height(self):
          return self.engine.height*dpi

     def heightMM(self):
          return int(self.height() * inch_mm)

     def logicalDpiX(self):
          return dpi

     def logicalDpiY(self):
          return dpi

     def physicalDpiX(self):
          return dpi

     def physicalDpiY(self):
          return dpi

     def depth(self):
          return 24

     def numColors(self):
          return 2147483647

     def metric(self, m):
          if m & qt4.QPaintDevice.PdmWidth:
               return self.width()
          elif m & qt4.QPaintDevice.PdmHeight:
               return self.height()
          elif m & qt4.QPaintDevice.PdmWidthMM:
               return self.widthMM()
          elif m & qt4.QPaintDevice.PdmHeightMM:
               return self.heightMM()
          elif m & qt4.QPaintDevice.PdmNumColors:
               return self.numColors()
          elif m & qt4.QPaintDevice.PdmDepth:
               return self.depth()
          elif m & qt4.QPaintDevice.PdmDpiX:
               return self.logicalDpiX()
          elif m & qt4.QPaintDevice.PdmDpiY:
               return self.logicalDpiY()
          elif m & qt4.QPaintDevice.PdmPhysicalDpiX:
               return self.physicalDpiX()
          elif m & qt4.QPaintDevice.PdmPhysicalDpiY:
               return self.physcialDpiY()

