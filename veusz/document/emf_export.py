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

"""A paint engine to produce EMF exports.

Requires: PyQt-x11-gpl-4.6-snapshot-20090906.tar.gz
          sip-4.9-snapshot-20090906.tar.gz
          pyemf
"""

import struct

import pyemf3
import pyemf3.emr

from .. import qtall as qt

inch_mm = 25.4
scale = 100

def isStockObject(obj):
    """Is this a stock windows object."""
    return (obj & 0x80000000) != 0

class EMFPaintEngine(qt.QPaintEngine):
    """Custom EMF paint engine."""

    def __init__(self, width_in, height_in, dpi=75):
        qt.QPaintEngine.__init__(
            self,
            qt.QPaintEngine.PaintEngineFeature.Antialiasing |
            qt.QPaintEngine.PaintEngineFeature.PainterPaths |
            qt.QPaintEngine.PaintEngineFeature.PrimitiveTransform |
            qt.QPaintEngine.PaintEngineFeature.PaintOutsidePaintEvent |
            qt.QPaintEngine.PaintEngineFeature.PatternBrush
        )
        self.width = width_in
        self.height = height_in
        self.dpi = dpi

    def begin(self, paintdevice):
        self.emf = pyemf3.EMF(self.width, self.height, int(self.dpi*scale))
        self.pen = self.emf.GetStockObject(pyemf3.BLACK_PEN)
        self.pencolor = (0, 0, 0)
        self.brush = self.emf.GetStockObject(pyemf3.NULL_BRUSH)

        self.paintdevice = paintdevice
        return True

    def drawLines(self, lines):
        """Draw lines to emf output."""

        for line in lines:
            self.emf.Polyline(
                 [ (int(line.x1()*scale), int(line.y1()*scale)),
                   (int(line.x2()*scale), int(line.y2()*scale)) ] )

    def drawPolygon(self, points, mode):
        """Draw polygon on output."""
        # print "Polygon"
        pts = [(int(p.x()*scale), int(p.y()*scale)) for p in points]

        if mode == qt.QPaintEngine.PolygonDrawMode.PolylineMode:
            self.emf.Polyline(pts)
        else:
            self.emf.SetPolyFillMode({
                qt.QPaintEngine.PolygonDrawMode.WindingMode: pyemf3.WINDING,
                qt.QPaintEngine.PolygonDrawMode.OddEvenMode: pyemf3.ALTERNATE,
                qt.QPaintEngine.PolygonDrawMode.ConvexMode: pyemf3.WINDING
            })
            self.emf.Polygon(pts)

    def drawEllipse(self, rect):
        """Draw an ellipse."""
        # print "ellipse"
        args = (
            int(rect.left()*scale),  int(rect.top()*scale),
            int(rect.right()*scale), int(rect.bottom()*scale),
            int(rect.left()*scale),  int(rect.top()*scale),
            int(rect.left()*scale),  int(rect.top()*scale),
        )
        self.emf.Pie(*args)
        self.emf.Arc(*args)

    def drawPoints(self, points):
        """Draw points."""
        # print "points"

        for pt in points:
            x, y = (pt.x()-0.5)*scale, (pt.y()-0.5)*scale
            self.emf.Pie(
                int(x), int(y),
                int((pt.x()+0.5)*scale), int((pt.y()+0.5)*scale),
                int(x), int(y), int(x), int(y) )

    def drawPixmap(self, r, pixmap, sr):
        """Draw pixmap to display."""

        # convert pixmap to BMP format
        bytearr = qt.QByteArray()
        buf = qt.QBuffer(bytearr)
        buf.open(qt.QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buf, "BMP")

        bmp = bytes(buf.data())
        self.emf.BitmapOut(
            int(r.left()*scale), int(r.top()*scale),
            int(r.width()*scale), int(r.bottom()*scale),
            int(sr.left()), int(sr.top()),
            int(sr.width()), int(sr.height()),
            bmp,
        )

    def _createPath(self, path):
        """Convert qt path to emf path"""
        self.emf.BeginPath()
        count = path.elementCount()
        i = 0
        #print "Start path"
        while i < count:
            e = path.elementAt(i)
            if e.type == qt.QPainterPath.ElementType.MoveToElement:
                self.emf.MoveTo( int(e.x*scale), int(e.y*scale) )
                #print "M", e.x*scale, e.y*scale
            elif e.type == qt.QPainterPath.ElementType.LineToElement:
                self.emf.LineTo( int(e.x*scale), int(e.y*scale) )
                #print "L", e.x*scale, e.y*scale
            elif e.type == qt.QPainterPath.ElementType.CurveToElement:
                e1 = path.elementAt(i+1)
                e2 = path.elementAt(i+2)
                params = (
                    ( int(e.x*scale), int(e.y*scale) ),
                    ( int(e1.x*scale), int(e1.y*scale) ),
                    ( int(e2.x*scale), int(e2.y*scale) ),
                )
                self.emf.PolyBezierTo(params)
                #print "C", params

                i += 2
            else:
                assert False

            i += 1

        ef = path.elementAt(0)
        el = path.elementAt(count-1)
        if ef.x == el.x and ef.y == el.y:
            self.emf.CloseFigure()
            #print "closing"
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
        path = qt.QPainterPath()
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

        # line style
        style = {
            qt.Qt.PenStyle.NoPen: pyemf3.PS_NULL,
            qt.Qt.PenStyle.SolidLine: pyemf3.PS_SOLID,
            qt.Qt.PenStyle.DashLine: pyemf3.PS_DASH,
            qt.Qt.PenStyle.DotLine: pyemf3.PS_DOT,
            qt.Qt.PenStyle.DashDotLine: pyemf3.PS_DASHDOT,
            qt.Qt.PenStyle.DashDotDotLine: pyemf3.PS_DASHDOTDOT,
            qt.Qt.PenStyle.CustomDashLine: pyemf3.PS_USERSTYLE,
        }[pen.style()]

        if style != pyemf3.PS_NULL:
            # set cap style
            style |= {
                qt.Qt.PenCapStyle.FlatCap: pyemf3.PS_ENDCAP_FLAT,
                qt.Qt.PenCapStyle.SquareCap: pyemf3.PS_ENDCAP_SQUARE,
                qt.Qt.HighDpiScaleFactorRoundingPolicy.RoundCap: pyemf3.PS_ENDCAP_ROUND,
            }[pen.capStyle()]

            # set join style
            style |= {
                qt.Qt.PenJoinStyle.MiterJoin: pyemf3.PS_JOIN_MITER,
                qt.Qt.PenJoinStyle.BevelJoin: pyemf3.PS_JOIN_BEVEL,
                qt.Qt.HighDpiScaleFactorRoundingPolicy.RoundJoin: pyemf3.PS_JOIN_ROUND,
                qt.Qt.PenJoinStyle.SvgMiterJoin: pyemf3.PS_JOIN_MITER,
            }[pen.joinStyle()]

            # use proper widths of lines
            style |= pyemf3.PS_GEOMETRIC

        width = int(pen.widthF()*scale)
        qc = pen.color()
        color = (qc.red(), qc.green(), qc.blue())
        self.pencolor = color

        if pen.style() == qt.Qt.PenStyle.CustomDashLine:
            # custom dash pattern
            dash = [int(pen.widthF()*scale*f) for f in pen.dashPattern()]
        else:
            dash = None

        newpen = self.emf.CreatePen(style, width, color, styleentries=dash)
        self.emf.SelectObject(newpen)

        # delete old pen if it is not a stock object
        if not isStockObject(self.pen):
            self.emf.DeleteObject(self.pen)
        self.pen = newpen

    def _updateBrush(self, brush):
        """Update to selected brush."""

        style = brush.style()
        qc = brush.color()
        color = (qc.red(), qc.green(), qc.blue())
        # print "brush", color
        if style == qt.Qt.BrushStyle.SolidPattern:
            newbrush = self.emf.CreateSolidBrush(color)
        elif style == qt.Qt.BrushStyle.NoBrush:
            newbrush = self.emf.GetStockObject(pyemf3.NULL_BRUSH)
        else:
            try:
                hatch = {
                    qt.Qt.BrushStyle.HorPattern: pyemf3.HS_HORIZONTAL,
                    qt.Qt.BrushStyle.VerPattern: pyemf3.HS_VERTICAL,
                    qt.Qt.BrushStyle.CrossPattern: pyemf3.HS_CROSS,
                    qt.Qt.BrushStyle.BDiagPattern: pyemf3.HS_BDIAGONAL,
                    qt.Qt.BrushStyle.FDiagPattern: pyemf3.HS_FDIAGONAL,
                    qt.Qt.BrushStyle.DiagCrossPattern: pyemf3.HS_DIAGCROSS
                }[brush.style()]
            except KeyError:
                newbrush = self.emf.CreateSolidBrush(color)
            else:
                newbrush = self.emf.CreateHatchBrush(hatch, color)
        self.emf.SelectObject(newbrush)

        if not isStockObject(self.brush):
            self.emf.DeleteObject(self.brush)
        self.brush = newbrush

    def _updateClipPath(self, path, operation):
        """Update clipping path."""
        # print "clip"
        if operation != qt.Qt.ClipOperation.NoClip:
            self._createPath(path)
            clipmode = {
                 qt.Qt.ClipOperation.ReplaceClip: pyemf3.RGN_COPY,
                 qt.Qt.ClipOperation.IntersectClip: pyemf3.RGN_AND,
            }[operation]
        else:
            # is this the only wave to get rid of clipping?
            self.emf.BeginPath()
            self.emf.MoveTo(0,0)
            w = int(self.width*self.dpi*scale)
            h = int(self.height*self.dpi*scale)
            self.emf.LineTo(w, 0)
            self.emf.LineTo(w, h)
            self.emf.LineTo(0, h)
            self.emf.CloseFigure()
            self.emf.EndPath()
            clipmode = pyemf3.RGN_COPY

        self.emf.SelectClipPath(mode=clipmode)

    def _updateTransform(self, m):
        """Update transformation."""
        self.emf.SetWorldTransform(
            m.m11(), m.m12(),
            m.m21(), m.m22(),
            m.dx()*scale, m.dy()*scale)

    def updateState(self, state):
        """Examine what has changed in state and call apropriate function."""
        ss = state.state()
        if ss & qt.QPaintEngine.DirtyFlag.DirtyPen:
            self._updatePen(state.pen())
        if ss & qt.QPaintEngine.DirtyFlag.DirtyBrush:
            self._updateBrush(state.brush())
        if ss & qt.QPaintEngine.DirtyFlag.DirtyTransform:
            self._updateTransform(state.transform())
        if ss & qt.QPaintEngine.DirtyFlag.DirtyClipPath:
            self._updateClipPath(state.clipPath(), state.clipOperation())
        if ss & qt.QPaintEngine.DirtyFlag.DirtyClipRegion:
            path = qt.QPainterPath()
            path.addRegion(state.clipRegion())
            self._updateClipPath(path, state.clipOperation())

    def type(self):
        return qt.QPaintEngine.Type.PostScript

class EMFPaintDevice(qt.QPaintDevice):
    """Paint device for EMF paint engine."""

    def __init__(self, width_in, height_in, dpi=75):
        qt.QPaintDevice.__init__(self)
        self.engine = EMFPaintEngine(width_in, height_in, dpi=dpi)

    def paintEngine(self):
        return self.engine

    def metric(self, m):
        """Return the metrics of the painter."""
        if m == qt.QPaintDevice.PaintDeviceMetric.PdmWidth:
            return int(self.engine.width * self.engine.dpi)
        elif m == qt.QPaintDevice.PaintDeviceMetric.PdmHeight:
            return int(self.engine.height * self.engine.dpi)
        elif m == qt.QPaintDevice.PaintDeviceMetric.PdmWidthMM:
            return int(self.engine.width * inch_mm)
        elif m == qt.QPaintDevice.PaintDeviceMetric.PdmHeightMM:
            return int(self.engine.height * inch_mm)
        elif m == qt.QPaintDevice.PaintDeviceMetric.PdmNumColors:
            return 2147483647
        elif m == qt.QPaintDevice.PaintDeviceMetric.PdmDepth:
            return 24
        elif m == qt.QPaintDevice.PaintDeviceMetric.PdmDpiX:
            return int(self.engine.dpi)
        elif m == qt.QPaintDevice.PaintDeviceMetric.PdmDpiY:
            return int(self.engine.dpi)
        elif m == qt.QPaintDevice.PaintDeviceMetric.PdmPhysicalDpiX:
            return int(self.engine.dpi)
        elif m == qt.QPaintDevice.PaintDeviceMetric.PdmPhysicalDpiY:
            return int(self.engine.dpi)
        elif m == qt.QPaintDevice.PaintDeviceMetric.PdmDevicePixelRatio:
            return 1

        # Qt >= 5.6
        elif m == getattr(qt.QPaintDevice, 'PdmDevicePixelRatioScaled', -1):
            return 1

        else:
            # fall back
            return qt.QPaintDevice.metric(self, m)
