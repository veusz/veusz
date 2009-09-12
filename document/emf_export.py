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
##############################################################################

# $Id$

"""A paint engine to produce EMF exports.

Requires: PyQt-x11-gpl-4.6-snapshot-20090906.tar.gz
          sip-4.9-snapshot-20090906.tar.gz
          pyemf
"""

import sys
import struct

import pyemf
import veusz.qtall as qt4

inch_mm = 25.4
scale = 100

def isStockObject(obj):
     """Is this a stock windows object."""
     return (obj & 0x80000000) != 0

class EMFPaintEngine(qt4.QPaintEngine):
     "Custom EMF paint engine."""

     def __init__(self, width_in, height_in, dpi=75):
          qt4.QPaintEngine.__init__(self)
          self.width = width_in
          self.height = height_in
          self.dpi = dpi

     def begin(self, paintdevice):
          self.emf = pyemf.EMF(self.width, self.height, self.dpi*scale)
          self.pen = self.emf.GetStockObject(pyemf.BLACK_PEN)
          self.pencolor = (0, 0, 0)
          self.brush = self.emf.GetStockObject(pyemf.NULL_BRUSH)
          
          self.paintdevice = paintdevice
          return True

     def drawLines(self, lines):
          """Draw lines to emf output."""

          for line in lines:
               self.emf.Polyline(
                    [(line.x1()*scale, line.y1()*scale),
                     (line.x2()*scale, line.y2()*scale)] )

     def drawPolygon(self, points, mode):
          """Draw polygon on output."""
          # print "Polygon"
          pts = [(p.x()*scale, p.y()*scale) for p in points]

          if mode == qt4.QPaintEngine.PolylineMode:
               self.emf.Polyline(pts)
          else:
               self.emf.SetPolyFillMode( {qt4.QPaintEngine.WindingMode:
                                               pyemf.WINDING,
                                          qt4.QPaintEngine.OddEvenMode:
                                               pyemf.ALTERNATE,
                                          qt4.QPaintEngine.ConvexMode:
                                               pyemf.WINDING} )
               self.emf.Polygon(pts)

     def drawEllipse(self, rect):
          """Draw an ellipse."""
          # print "ellipse"
          args = (rect.left()*scale, rect.top()*scale,
                  rect.right()*scale, rect.bottom()*scale,
                  rect.left()*scale, rect.top()*scale,
                  rect.left()*scale, rect.top()*scale)
          self.emf.Pie(*args)
          self.emf.Arc(*args)

     def drawPoints(self, points):
          """Draw points."""
          # print "points"

          for pt in points:
               x, y = (pt.x()-0.5)*scale, (pt.y()-0.5)*scale
               self.emf.Pie( x, y,
                             (pt.x()+0.5)*scale, (pt.y()+0.5)*scale, 
                             x, y, x, y )

     def drawPixmap(self, r, pixmap, sr):
          """Draw pixmap to display."""

          # convert pixmap to BMP format
          bytes = qt4.QByteArray()
          buffer = qt4.QBuffer(bytes)
          buffer.open(qt4.QIODevice.WriteOnly)
          pixmap.save(buffer, "BMP")

          # chop off bmp header to get DIB
          bmp = str(buffer.data())
          dib = bmp[0xe:]
          hdrsize, = struct.unpack('<i', bmp[0xe:0x12])
          dataindex, = struct.unpack('<i', bmp[0xa:0xe])
          datasize, = struct.unpack('<i', bmp[0x22:0x26])

          epix = pyemf._EMR._STRETCHDIBITS()
          epix.rclBounds_left = r.left()*scale
          epix.rclBounds_top = r.top()*scale
          epix.rclBounds_right = r.right()*scale
          epix.rclBounds_bottom = r.bottom()*scale
          epix.xDest = r.left()*scale
          epix.yDest = r.top()*scale
          epix.cxDest = r.width()*scale
          epix.cyDest = r.height()*scale
          epix.xSrc = sr.left()
          epix.ySrc = sr.top()
          epix.cxSrc = sr.width()
          epix.cySrc = sr.height()
   
          epix.dwRop = 0xcc0020 # SRCCOPY
          epix.offBmiSrc = epix.serializeOffset()
          epix.cbBmiSrc = hdrsize
          epix.offBitsSrc = epix.serializeOffset() + dataindex - 0xe
          epix.cbBitsSrc = datasize
          epix.iUsageSrc = 0x0 # DIB_RGB_COLORS

          epix.unhandleddata = dib

          self.emf._append(epix)

          # print "pixmap", r, pixmap, sr, pixmap.width(), pixmap.height()

     def _createPath(self, path):
          """Convert qt path to emf path"""
          self.emf.BeginPath()
          count = path.elementCount()
          i = 0
          while i < count:
               e = path.elementAt(i)
               if e.type == qt4.QPainterPath.MoveToElement:
                    self.emf.MoveTo(e.x*scale, e.y*scale)
               elif e.type == qt4.QPainterPath.LineToElement:
                    self.emf.LineTo(e.x*scale, e.y*scale)
               elif e.type == qt4.QPainterPath.CurveToElement:
                    e1 = path.elementAt(i+1)
                    e2 = path.elementAt(i+2)
                    self.emf.PolyBezierTo((
                              (e.x*scale, e.y*scale),
                              (e1.x*scale, e1.y*scale),
                              (e2.x*scale, e2.y*scale)))
                    i += 2
               else:
                    assert False

               i += 1
          self.emf.CloseFigure()
          self.emf.EndPath()

     def drawPath(self, path):
          """Draw a path on the output."""
          # print "path"

          self._createPath(path)
          self.emf.StrokeAndFillPath()

     def drawTextItem(self, pt, textitem):
          """Convert text to a path and draw it.
          """
          # print "text", pt, textitem.text()
          path = qt4.QPainterPath()
          path.addText(pt, textitem.font(), textitem.text())

          fill = self.emf.CreateSolidBrush(self.pencolor)
          self.emf.SelectObject(fill)
          self._createPath(path)
          self.emf.FillPath()
          self.emf.SelectObject(self.brush)
          self.emf.DeleteObject(fill)

     def end(self):
          return True

     def saveFile(self, filename):
          self.emf.save(filename)

     def _updatePen(self, pen):
          """Update the pen to the currently selected one."""

          style = {qt4.Qt.NoPen: pyemf.PS_NULL,
                   qt4.Qt.SolidLine: pyemf.PS_SOLID,
                   qt4.Qt.DashLine: pyemf.PS_DASH,
                   qt4.Qt.DotLine: pyemf.PS_DOT,
                   qt4.Qt.DashDotLine: pyemf.PS_DASHDOT,
                   qt4.Qt.DashDotDotLine: pyemf.PS_DASHDOTDOT,
                   qt4.Qt.CustomDashLine: pyemf.PS_SOLID}[pen.style()]

          style |= {qt4.Qt.FlatCap: pyemf.PS_ENDCAP_FLAT,
                    qt4.Qt.SquareCap: pyemf.PS_ENDCAP_SQUARE,
                    qt4.Qt.RoundCap: pyemf.PS_ENDCAP_ROUND}[pen.capStyle()]

          style |= {qt4.Qt.MiterJoin: pyemf.PS_JOIN_MITER,
                    qt4.Qt.BevelJoin: pyemf.PS_JOIN_BEVEL,
                    qt4.Qt.RoundJoin: pyemf.PS_JOIN_ROUND,
                    qt4.Qt.SvgMiterJoin: pyemf.PS_JOIN_MITER}[pen.joinStyle()]

          width = pen.widthF()*scale
          qc = pen.color()
          color = (qc.red(), qc.green(), qc.blue())
          self.pencolor = color
          # print "pen", color
          newpen = self.emf.CreatePen(style, width, color)
          self.emf.SelectObject(newpen)

          if not isStockObject(self.pen):
               self.emf.DeleteObject(self.pen)
          self.pen = newpen

     def _updateBrush(self, brush):
          """Update to selected brush."""

          style = brush.style()
          qc = brush.color()
          color = (qc.red(), qc.green(), qc.blue())
          # print "brush", color
          if style == qt4.Qt.SolidPattern:
               newbrush = self.emf.CreateSolidBrush(color)
          elif style == qt4.Qt.NoBrush:
               newbrush = self.emf.GetStockObject(pyemf.NULL_BRUSH)
          else:
               try:
                    hatch = {qt4.Qt.HorPattern: pyemf.HS_HORIZONTAL,
                             qt4.Qt.VerPattern: pyemf.HS_VERTICAL,
                             qt4.Qt.CrossPattern: pyemf.HS_CROSS,
                             qt4.Qt.BDiagPattern: pyemf.HS_BDIAGONAL,
                             qt4.Qt.FDiagPattern: pyemf.HS_FDIAGONAL,
                             qt4.Qt.DiagCrossPattern:
                                  pyemf.HS_DIAGCROSS}[brush.style()]
               except KeyError:
                    newbrush = self.emf.CreateSolidBrush(color)
               else:
                    newbrush = CreateHatchBrush(hatch, color)
          self.emf.SelectObject(newbrush)

          if not isStockObject(self.brush):
               self.emf.DeleteObject(self.brush)
          self.brush = newbrush

     def _updateClipPath(self, path, operation):
          """Update clipping path."""
          # print "clip"
          if operation != qt4.Qt.NoClip:
               self._createPath(path)
               clipmode = {
                    qt4.Qt.ReplaceClip: pyemf.RGN_COPY,
                    qt4.Qt.IntersectClip: pyemf.RGN_AND,
                    qt4.Qt.UniteClip: pyemf.RGN_OR}[operation]
          else:
               # is this the only wave to get rid of clipping?
               self.emf.BeginPath()
               self.emf.MoveTo(0,0)
               w = self.width*self.dpi*scale
               h = self.height*self.dpi*scale
               self.emf.LineTo(w, 0)
               self.emf.LineTo(w, h)
               self.emf.LineTo(0, h)
               self.emf.CloseFigure()
               self.emf.EndPath()
               clipmode = pyemf.RGN_COPY

          self.emf.SelectClipPath(mode=clipmode)

     def _updateTransform(self, m):
          """Update transformation."""
          # print "transform", (m.m11(), m.m12(),
          #                     m.m21(), m.m22(),
          #                     m.dx(), m.dy())
          self.emf.SetWorldTransform(m.m11(), m.m12(),
                                     m.m21(), m.m22(),
                                     m.dx()*scale, m.dy()*scale)

     def updateState(self, state):
          """Examine what has changed in state and call apropriate function."""
          ss = state.state()
          if ss & qt4.QPaintEngine.DirtyPen:
               self._updatePen(state.pen())
          if ss & qt4.QPaintEngine.DirtyBrush:
               self._updateBrush(state.brush())
          if ss & qt4.QPaintEngine.DirtyClipPath:
               self._updateClipPath(state.clipPath(), state.clipOperation())
          if ss & qt4.QPaintEngine.DirtyClipRegion:
               path = qt4.QPainterPath()
               path.addRegion(state.clipRegion())
               self._updateClipPath(path, state.clipOperation())
          if ss & qt4.QPaintEngine.DirtyTransform:
               self._updateTransform(state.matrix())

     def type(self):
          return qt4.QPaintEngine.PostScript

     def hasFeature(self, f):
          return f in (
               qt4.QPaintEngine.Antialiasing,
               qt4.QPaintEngine.PainterPaths,
               qt4.QPaintEngine.PrimitiveTransform,
               qt4.QPaintEngine.PaintOutsidePaintEvent,
               )

class EMFPaintDevice(qt4.QPaintDevice):
     """Paint device for EMF paint engine."""

     def __init__(self, width_in, height_in, dpi=75):
          qt4.QPaintDevice.__init__(self)
          self.engine = EMFPaintEngine(width_in, height_in, dpi=dpi)

     def paintEngine(self):
          return self.engine

     def width(self):
          return self.engine.width*self.engine.dpi

     def widthMM(self):
          return int(self.width() * inch_mm)

     def height(self):
          return self.engine.height*self.engine.dpi

     def heightMM(self):
          return int(self.height() * inch_mm)

     def logicalDpiX(self):
          return self.engine.dpi

     def logicalDpiY(self):
          return self.engine.dpi

     def physicalDpiX(self):
          return self.engine.dpi

     def physicalDpiY(self):
          return self.engine.dpi

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

def main():
     app = qt4.QApplication(sys.argv)

     device = EMFPaintDevice(5,5)
     p = qt4.QPainter(device)
     p.drawLines( [qt4.QLineF(0, 0, 100, 100), qt4.QLineF(100,100,200,200)] )
     p.drawLine( qt4.QLineF(200,200,300,300))

     p.setBrush(qt4.QBrush(qt4.Qt.red))
     p.setPen(qt4.QPen(qt4.QBrush(qt4.Qt.green), 3.))
     p.save()
     p.setClipRect( qt4.QRectF( qt4.QPointF(0,0), qt4.QPointF(70,70)))
     p.drawPolygon(qt4.QPolygonF([qt4.QPointF(0,0), qt4.QPointF(100,0),
                                  qt4.QPointF(0,100), qt4.QPointF(100,100)]))
     p.setBrush(qt4.QBrush(qt4.Qt.blue))
     p.drawEllipse(qt4.QRectF(200,200,50,50))
     p.restore()

     p.setPen(qt4.QPen(qt4.Qt.blue))
     p.setFont(qt4.QFont("Arial", 60))
     p.drawText(100, 100, "Hi there")

     x = qt4.QPixmap(10,10)
     x.fill(qt4.Qt.red)
     p.drawPixmap(50,50,100,100,x)
     p.setPen(qt4.QPen(qt4.QBrush(qt4.Qt.black), 1.))
     p.drawRect(50,50,100,100)

     p.end()

     device.paintEngine().saveFile("out.emf")

if __name__ == '__main__':
     main()
