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

"""A home-brewed SVG paint engine for doing svg with clipping
and exporting text as paths for WYSIWYG."""

from __future__ import division, print_function
import re

from ..compat import crange, cbytes
from .. import qtall as qt4

# dpi runs at many times usual, and results are scaled down
# helps fix point issues in font sizes
dpi = 900.
scale = 0.1

inch_mm = 25.4
inch_pt = 72.0

def printpath(path):
    """Debugging print path."""
    print("Contents of", path)
    for i in crange(path.elementCount()):
        el = path.elementAt(i)
        print(" ", el.type, el.x, el.y)

def fltStr(v, prec=2):
    """Change a float to a string, using a maximum number of decimal places
    but removing trailing zeros."""

    # ensures consistent rounding behaviour on different platforms
    v = round(v, prec+2)

    val = ('% 20.10f' % v)[:10+prec]

    # drop any trailing zeros
    val = val.rstrip('0').lstrip(' ').rstrip('.')
    # get rid of -0s (platform differences here)
    if val == '-0':
        val = '0'
    return val

def escapeXML(text):
    """Escape special characters in XML."""
    # we have swap & with an unused character, so we can replace it later
    text = text.replace('&', u'\ue001')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&apos;')
    text = text.replace(u'\ue001', '&amp;')
    return text

def createPath(path):
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

class SVGElement(object):
    """SVG element in output.
    This represents the XML tree in memory
    """

    def __init__(self, parent, eltype, attrb, text=None):
        """Intialise element.
        parent: parent element or None
        eltype: type (e.g. 'polyline')
        attrb: attribute string appended to output
        text: text to output between this and closing element.
        """
        self.eltype = eltype
        self.attrb = attrb
        self.children = []
        self.parent = parent
        self.text = text

        if parent:
            parent.children.append(self)

    def write(self, fileobj):
        """Write element and its children to the output file."""
        fileobj.write('<%s' % self.eltype)
        if self.attrb:
            fileobj.write(' ' + self.attrb)

        if self.text:
            fileobj.write('>%s</%s>\n' % (self.text, self.eltype))
        elif self.children:
            fileobj.write('>\n')
            for c in self.children:
                c.write(fileobj)
            fileobj.write('</%s>\n' % self.eltype)
        else:
            # simple close tag if not children or text
            fileobj.write('/>\n')

class SVGPaintEngine(qt4.QPaintEngine):
    """Paint engine class for writing to svg files."""

    def __init__(self, width_in, height_in, writetextastext=False):
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

        self.imageformat = 'png'
        self.writetextastext = writetextastext

    def begin(self, paintdevice):
        """Start painting."""
        self.device = paintdevice

        self.pen = qt4.QPen()
        self.brush = qt4.QBrush()
        self.clippath = None
        self.clipnum = 0
        self.existingclips = {}
        self.transform = qt4.QTransform()

        # svg root element for qt defaults
        self.rootelement = SVGElement(
            None, 'svg',
            ('width="%spx" height="%spx" version="1.1"\n'
             '    xmlns="http://www.w3.org/2000/svg"\n'
             '    xmlns:xlink="http://www.w3.org/1999/xlink"') %
            (fltStr(self.width*dpi*scale), fltStr(self.height*dpi*scale)))
        SVGElement(self.rootelement, 'desc', '', 'Veusz output document')

        # definitions, for clips, etc.
        self.defs = SVGElement(self.rootelement, 'defs', '')

        # this is where all the drawing goes
        self.celement = SVGElement(
            self.rootelement, 'g',
            'stroke-linejoin="bevel" stroke-linecap="square" '
            'stroke="#000000" fill-rule="evenodd"')

        # previous transform, stroke and clip states
        self.oldstate = [None, None, None]

        # cache paths to avoid duplication
        self.pathcache = {}
        self.pathcacheidx = 0

        return True

    def pruneEmptyGroups(self):
        """Take the element tree and remove any empty group entries."""

        def recursive(root):
            children = list(root.children)
            # remove any empty children first
            for c in children:
                recursive(c)
            if root.eltype == 'g' and len(root.children) == 0:
                # safe to remove
                index = root.parent.children.index(root)
                del root.parent.children[index]

            # merge equal groups
            last = None
            i = 0
            while i < len(root.children):
                this = root.children[i]
                if ( last is not None and
                     last.eltype == this.eltype and last.attrb == this.attrb
                     and last.text == this.text ):
                    last.children += this.children
                    del root.children[i]
                else:
                    last = this
                    i += 1

        recursive(self.rootelement)

    def end(self):
        self.pruneEmptyGroups()

        fileobj = self.device.fileobj
        fileobj.write('<?xml version="1.0" standalone="no"?>\n'
                      '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n'
                      '  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n')

        # write all the elements
        self.rootelement.write(fileobj)

        return True

    def _updateClipPath(self, clippath, clipoperation):
        """Update clip path given state change."""

        clippath = self.transform.map(clippath)

        if clipoperation == qt4.Qt.NoClip:
            self.clippath = None
        elif clipoperation == qt4.Qt.ReplaceClip:
            self.clippath = clippath
        elif clipoperation == qt4.Qt.IntersectClip:
            self.clippath = self.clippath.intersected(clippath)
        elif clipoperation == qt4.Qt.UniteClip:
            self.clippath = self.clippath.united(clippath)
        else:
            assert False

    def updateState(self, state):
        """Examine what has changed in state and call apropriate function."""
        ss = state.state()

        # state is a list of transform, stroke/fill and clip states
        statevec = list(self.oldstate)
        if ss & qt4.QPaintEngine.DirtyTransform:
            self.transform = state.transform()
            statevec[0] = self.transformState()
        if ss & qt4.QPaintEngine.DirtyPen:
            self.pen = state.pen()
            statevec[1] = self.strokeFillState()
        if ss & qt4.QPaintEngine.DirtyBrush:
            self.brush = state.brush()
            statevec[1] = self.strokeFillState()
        if ss & qt4.QPaintEngine.DirtyClipPath:
            self._updateClipPath(state.clipPath(), state.clipOperation())
            statevec[2] = self.clipState()
        if ss & qt4.QPaintEngine.DirtyClipRegion:
            path = qt4.QPainterPath()
            path.addRegion(state.clipRegion())
            self._updateClipPath(path, state.clipOperation())
            statevec[2] = self.clipState()

        # work out which state differs first
        pop = 0
        for i in crange(2, -1, -1):
            if statevec[i] != self.oldstate[i]:
                pop = i+1
                break

        # go back up the tree the required number of times
        for i in crange(pop):
            if self.oldstate[i]:
                self.celement = self.celement.parent

        # create new elements for changed states
        for i in crange(pop-1, -1, -1):
            if statevec[i]:
                self.celement = SVGElement(
                    self.celement, 'g', ' '.join(statevec[i]))

        self.oldstate = statevec

    def clipState(self):
        """Get SVG clipping state. This is in the form of an svg group"""

        if self.clippath is None:
            return ()

        path = createPath(self.clippath)

        if path in self.existingclips:
            url = 'url(#c%i)' % self.existingclips[path]
        else:
            clippath = SVGElement(self.defs, 'clipPath',
                                  'id="c%i"' % self.clipnum)
            SVGElement(clippath, 'path', 'd="%s"' % path)
            url = 'url(#c%i)' % self.clipnum
            self.existingclips[path] = self.clipnum
            self.clipnum += 1

        return ('clip-path="%s"' % url,)

    def strokeFillState(self):
        """Return stroke-fill state."""

        vals = {}
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
            w = 1./scale
        vals['stroke-width'] = fltStr(w*scale)

        # - line style
        if p.style() == qt4.Qt.NoPen:
            vals['stroke'] = 'none'
        elif p.style() not in (qt4.Qt.SolidLine, qt4.Qt.NoPen):
            # convert from pen width fractions to pts
            nums = [fltStr(scale*w*x) for x in p.dashPattern()]
            vals['stroke-dasharray'] = ','.join(nums)

        # BRUSH STYLES
        b = self.brush
        if b.style() == qt4.Qt.NoBrush:
            vals['fill'] = 'none'
        else:
            vals['fill'] = b.color().name()
        if b.color().alphaF() != 1.0:
            vals['fill-opacity'] = '%.3g' % b.color().alphaF()

        items = ['%s="%s"' % x for x in sorted(vals.items())]
        return tuple(items)

    def transformState(self):
        if not self.transform.isIdentity():
            m = self.transform
            dx, dy = m.dx(), m.dy()
            if (m.m11(), m.m12(), m.m21(), m.m22()) == (1., 0., 0., 1):
                out = ('transform="translate(%s,%s)"' % (
                        fltStr(dx*scale), fltStr(dy*scale)) ,)
            else:
                out = ('transform="matrix(%s %s %s %s %s %s)"' % (
                        fltStr(m.m11(), 4), fltStr(m.m12(), 4),
                        fltStr(m.m21(), 4), fltStr(m.m22(), 4),
                        fltStr(dx*scale), fltStr(dy*scale) ),)
        else:
            out = ()
        return out

    def drawPath(self, path):
        """Draw a path on the output."""
        p = createPath(path)

        attrb = 'd="%s"' % p
        if path.fillRule() == qt4.Qt.WindingFill:
            attrb += ' fill-rule="nonzero"'

        if attrb in self.pathcache:
            element, num = self.pathcache[attrb]
            if num is None:
                # this is the first time an element has been referenced again
                # assign it an id for use below
                num = self.pathcacheidx
                self.pathcacheidx += 1
                self.pathcache[attrb] = element, num
                # add an id attribute
                element.attrb += ' id="p%i"' % num

            # if the parent is a translation, swallow this into the use element
            m = re.match('transform="translate\(([-0-9.]+),([-0-9.]+)\)"',
                         self.celement.attrb)
            if m:
                SVGElement(self.celement.parent, 'use',
                           'xlink:href="#p%i" x="%s" y="%s"' % (
                        num, m.group(1), m.group(2)))
            else:
                SVGElement(self.celement, 'use', 'xlink:href="#p%i"' % num)
        else:
            pathel = SVGElement(self.celement, 'path', attrb)
            self.pathcache[attrb] = [pathel, None]

    def drawTextItem(self, pt, textitem):
        """Convert text to a path and draw it.
        """

        if self.writetextastext:
            # size
            f = textitem.font()
            if f.pixelSize() > 0:
                size = f.pixelSize()*scale
            else:
                size = f.pointSizeF()*scale*dpi/inch_pt

            font = textitem.font()
            grpattrb = [
                'stroke="none"',
                'fill="%s"' % self.pen.color().name(),
                'fill-opacity="%.3g"' % self.pen.color().alphaF(),
                'font-family="%s"' % escapeXML(font.family()),
                'font-size="%s"' % size,
                ]
            if font.italic():
                grpattrb.append('font-style="italic"')
            if font.bold():
                grpattrb.append('font-weight="bold"')

            grp = SVGElement(
                self.celement, 'g',
                ' '.join(grpattrb) )

            text = escapeXML( textitem.text() )

            textattrb = [
                'x="%s"' % fltStr(pt.x()*scale),
                'y="%s"' % fltStr(pt.y()*scale),
                'textLength="%s"' % fltStr(textitem.width()*scale),
                ]

            # spaces get lost without this
            if text.find('  ') >= 0 or text[:1] == ' ' or text[-1:] == ' ':
                textattrb.append('xml:space="preserve"')

            # write as an SVG text element
            SVGElement(
                grp, 'text',
                ' '.join(textattrb),
                text=text )

        else:
            # convert to a path
            path = qt4.QPainterPath()
            path.addText(pt, textitem.font(), textitem.text())
            p = createPath(path)
            SVGElement(
                self.celement, 'path',
                'd="%s" fill="%s" stroke="none" fill-opacity="%.3g"' % (
                    p, self.pen.color().name(), self.pen.color().alphaF()) )

    def drawLines(self, lines):
        """Draw multiple lines."""
        paths = []
        for line in lines:
            path = 'M%s,%sl%s,%s' % (
                fltStr(line.x1()*scale), fltStr(line.y1()*scale),
                fltStr((line.x2()-line.x1())*scale),
                fltStr((line.y2()-line.y1())*scale))
            paths.append(path)
        SVGElement(self.celement, 'path', 'd="%s"' % ''.join(paths))

    def drawPolygon(self, points, mode):
        """Draw polygon on output."""
        pts = []
        for p in points:
            pts.append( '%s,%s' % (fltStr(p.x()*scale), fltStr(p.y()*scale)) )

        if mode == qt4.QPaintEngine.PolylineMode:
            SVGElement(self.celement, 'polyline',
                       'fill="none" points="%s"' % ' '.join(pts))

        else:
            attrb = 'points="%s"' % ' '.join(pts)
            if mode == qt4.Qt.WindingFill:
                attrb += ' fill-rule="nonzero"'
            SVGElement(self.celement, 'polygon', attrb)

    def drawEllipse(self, rect):
        """Draw an ellipse to the svg file."""
        SVGElement(self.celement, 'ellipse',
                   'cx="%s" cy="%s" rx="%s" ry="%s"' %
                   (fltStr(rect.center().x()*scale),
                    fltStr(rect.center().y()*scale),
                    fltStr(rect.width()*0.5*scale),
                    fltStr(rect.height()*0.5*scale)))

    def drawPoints(self, points):
        """Draw points."""
        for pt in points:
            x, y = fltStr(pt.x()*scale), fltStr(pt.y()*scale)
            SVGElement(self.celement, 'line',
                       ('x1="%s" y1="%s" x2="%s" y2="%s" '
                        'stroke-linecap="round"') % (x, y, x, y))

    def drawImage(self, r, img, sr, flags):
        """Draw image.
        As the pixmap method uses the same code, just call this."""
        self.drawPixmap(r, img, sr)

    def drawPixmap(self, r, pixmap, sr):
        """Draw pixmap svg item.

        This is converted to a bitmap and embedded in the output
        """

        # convert pixmap to textual data
        data = qt4.QByteArray()
        buf = qt4.QBuffer(data)
        buf.open(qt4.QBuffer.ReadWrite)
        pixmap.save(buf, self.imageformat.upper(), 0)
        buf.close()

        attrb = [ 'x="%s" y="%s" ' % (fltStr(r.x()*scale), fltStr(r.y()*scale)),
                  'width="%s" ' % fltStr(r.width()*scale),
                  'height="%s" ' % fltStr(r.height()*scale),
                  'xlink:href="data:image/%s;base64,' % self.imageformat,
                  cbytes(data.toBase64()).decode('ascii'),
                  '" preserveAspectRatio="none"' ]
        SVGElement(self.celement, 'image', ''.join(attrb))

    def type(self):
        """A random number for the engine."""
        return qt4.QPaintEngine.User + 11

class SVGPaintDevice(qt4.QPaintDevice):
    """Paint device for SVG paint engine."""

    def __init__(self, fileobj, width_in, height_in,
                 writetextastext=False):
        qt4.QPaintDevice.__init__(self)
        self.engine = SVGPaintEngine(width_in, height_in,
                                     writetextastext=writetextastext)
        self.fileobj = fileobj

    def paintEngine(self):
        return self.engine

    def metric(self, m):
        """Return the metrics of the painter."""

        if m == qt4.QPaintDevice.PdmWidth:
            return int(self.engine.width * dpi)
        elif m == qt4.QPaintDevice.PdmHeight:
            return int(self.engine.height * dpi)
        elif m == qt4.QPaintDevice.PdmWidthMM:
            return int(self.engine.width * inch_mm)
        elif m == qt4.QPaintDevice.PdmHeightMM:
            return int(self.engine.height * inch_mm)
        elif m == qt4.QPaintDevice.PdmNumColors:
            return 2147483647
        elif m == qt4.QPaintDevice.PdmDepth:
            return 24
        elif m == qt4.QPaintDevice.PdmDpiX:
            return int(dpi)
        elif m == qt4.QPaintDevice.PdmDpiY:
            return int(dpi)
        elif m == qt4.QPaintDevice.PdmPhysicalDpiX:
            return int(dpi)
        elif m == qt4.QPaintDevice.PdmPhysicalDpiY:
            return int(dpi)
        elif m == qt4.QPaintDevice.PdmDevicePixelRatio:
            return 1

        # Qt >= 5.6
        elif m == getattr(qt4.QPaintDevice, 'PdmDevicePixelRatioScaled', -1):
            return 1

        else:
            raise RuntimeError('Invalid metric argument '+repr(m))
