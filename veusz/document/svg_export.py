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
from .. import qtall as qt

# physical sizes
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
        if e.type == qt.QPainterPath.MoveToElement:
            p.append( 'm%s,%s' % (fltStr(nx-ox), fltStr(ny-oy)) )
            ox, oy = nx, ny
        elif e.type == qt.QPainterPath.LineToElement:
            p.append( 'l%s,%s' % (fltStr(nx-ox), fltStr(ny-oy)) )
            ox, oy = nx, ny
        elif e.type == qt.QPainterPath.CurveToElement:
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

class SVGPaintEngine(qt.QPaintEngine):
    """Paint engine class for writing to svg files."""

    def __init__(self, writetextastext=False):
        qt.QPaintEngine.__init__(
            self,
            qt.QPaintEngine.Antialiasing |
            qt.QPaintEngine.PainterPaths |
            qt.QPaintEngine.PrimitiveTransform |
            qt.QPaintEngine.PaintOutsidePaintEvent |
            qt.QPaintEngine.PixmapTransform |
            qt.QPaintEngine.AlphaBlend
        )

        self.imageformat = 'png'
        self.writetextastext = writetextastext

    def begin(self, paintdevice):
        """Start painting."""
        self.device = paintdevice
        self.scale = paintdevice.scale

        self.pen = qt.QPen()
        self.brush = qt.QBrush()
        self.clippath = None
        self.clipnum = 0
        self.existingclips = {}
        self.transform = qt.QTransform()

        # svg root element for qt defaults
        self.rootelement = SVGElement(
            None, 'svg',
            ('width="%spx" height="%spx" version="1.1"\n'
             '    xmlns="http://www.w3.org/2000/svg"\n'
             '    xmlns:xlink="http://www.w3.org/1999/xlink"') % (
                 fltStr(self.device.width*self.device.sdpi*self.scale),
                 fltStr(self.device.height*self.device.sdpi*self.scale))
        )
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

        if clipoperation == qt.Qt.NoClip:
            self.clippath = None
        elif clipoperation == qt.Qt.ReplaceClip:
            self.clippath = clippath
        elif clipoperation == qt.Qt.IntersectClip:
            self.clippath = self.clippath.intersected(clippath)
        elif clipoperation == qt.Qt.UniteClip:
            self.clippath = self.clippath.united(clippath)
        else:
            assert False

    def updateState(self, state):
        """Examine what has changed in state and call apropriate function."""
        ss = state.state()

        # state is a list of transform, stroke/fill and clip states
        statevec = list(self.oldstate)
        if ss & qt.QPaintEngine.DirtyTransform:
            self.transform = state.transform()
            statevec[0] = self.transformState()
        if ss & qt.QPaintEngine.DirtyPen:
            self.pen = state.pen()
            statevec[1] = self.strokeFillState()
        if ss & qt.QPaintEngine.DirtyBrush:
            self.brush = state.brush()
            statevec[1] = self.strokeFillState()
        if ss & qt.QPaintEngine.DirtyClipPath:
            self._updateClipPath(state.clipPath(), state.clipOperation())
            statevec[2] = self.clipState()
        if ss & qt.QPaintEngine.DirtyClipRegion:
            path = qt.QPainterPath()
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

        path = createPath(self.clippath, self.scale)

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
        if p.joinStyle() != qt.Qt.BevelJoin:
            vals['stroke-linejoin'] = {
                qt.Qt.MiterJoin: 'miter',
                qt.Qt.SvgMiterJoin: 'miter',
                qt.Qt.RoundJoin: 'round',
                qt.Qt.BevelJoin: 'bevel'
                }[p.joinStyle()]
        # - cap style
        if p.capStyle() != qt.Qt.SquareCap:
            vals['stroke-linecap'] = {
                qt.Qt.FlatCap: 'butt',
                qt.Qt.SquareCap: 'square',
                qt.Qt.RoundCap: 'round'
                }[p.capStyle()]
        # - width
        w = p.widthF()
        # width 0 is device width for qt
        if w == 0.:
            w = 1./self.scale
        vals['stroke-width'] = fltStr(w*self.scale)

        # - line style
        if p.style() == qt.Qt.NoPen:
            vals['stroke'] = 'none'
        elif p.style() not in (qt.Qt.SolidLine, qt.Qt.NoPen):
            # convert from pen width fractions to pts
            nums = [fltStr(self.scale*w*x) for x in p.dashPattern()]
            vals['stroke-dasharray'] = ','.join(nums)

        # BRUSH STYLES
        b = self.brush
        if b.style() == qt.Qt.NoBrush:
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
                        fltStr(dx*self.scale), fltStr(dy*self.scale)) ,)
            else:
                out = ('transform="matrix(%s %s %s %s %s %s)"' % (
                        fltStr(m.m11(), 4), fltStr(m.m12(), 4),
                        fltStr(m.m21(), 4), fltStr(m.m22(), 4),
                        fltStr(dx*self.scale), fltStr(dy*self.scale) ),)
        else:
            out = ()
        return out

    def drawPath(self, path):
        """Draw a path on the output."""
        p = createPath(path, self.scale)

        attrb = 'd="%s"' % p
        if path.fillRule() == qt.Qt.WindingFill:
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
            m = re.match(r'transform="translate\(([-0-9.]+),([-0-9.]+)\)"',
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
                size = f.pixelSize()*self.scale
            else:
                size = f.pointSizeF()*self.scale*self.device.sdpi/inch_pt

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
                'x="%s"' % fltStr(pt.x()*self.scale),
                'y="%s"' % fltStr(pt.y()*self.scale),
                'textLength="%s"' % fltStr(textitem.width()*self.scale),
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
            path = qt.QPainterPath()
            path.addText(pt, textitem.font(), textitem.text())
            p = createPath(path, self.scale)
            SVGElement(
                self.celement, 'path',
                'd="%s" fill="%s" stroke="none" fill-opacity="%.3g"' % (
                    p, self.pen.color().name(), self.pen.color().alphaF()) )

    def drawLines(self, lines):
        """Draw multiple lines."""
        paths = []
        for line in lines:
            path = 'M%s,%sl%s,%s' % (
                fltStr(line.x1()*self.scale),
                fltStr(line.y1()*self.scale),
                fltStr((line.x2()-line.x1())*self.scale),
                fltStr((line.y2()-line.y1())*self.scale))
            paths.append(path)
        SVGElement(self.celement, 'path', 'd="%s"' % ''.join(paths))

    def drawPolygon(self, points, mode):
        """Draw polygon on output."""
        pts = []
        for p in points:
            pts.append( '%s,%s' % (fltStr(p.x()*self.scale), fltStr(p.y()*self.scale)) )

        if mode == qt.QPaintEngine.PolylineMode:
            SVGElement(self.celement, 'polyline',
                       'fill="none" points="%s"' % ' '.join(pts))

        else:
            attrb = 'points="%s"' % ' '.join(pts)
            if mode == qt.Qt.WindingFill:
                attrb += ' fill-rule="nonzero"'
            SVGElement(self.celement, 'polygon', attrb)

    def drawEllipse(self, rect):
        """Draw an ellipse to the svg file."""
        SVGElement(self.celement, 'ellipse',
                   'cx="%s" cy="%s" rx="%s" ry="%s"' %
                   (fltStr(rect.center().x()*self.scale),
                    fltStr(rect.center().y()*self.scale),
                    fltStr(rect.width()*0.5*self.scale),
                    fltStr(rect.height()*0.5*self.scale)))

    def drawPoints(self, points):
        """Draw points."""
        for pt in points:
            x, y = fltStr(pt.x()*self.scale), fltStr(pt.y()*self.scale)
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
        data = qt.QByteArray()
        buf = qt.QBuffer(data)
        buf.open(qt.QBuffer.ReadWrite)
        pixmap.save(buf, self.imageformat.upper(), 0)
        buf.close()

        attrb = [ 'x="%s" y="%s" ' % (fltStr(r.x()*self.scale), fltStr(r.y()*self.scale)),
                  'width="%s" ' % fltStr(r.width()*self.scale),
                  'height="%s" ' % fltStr(r.height()*self.scale),
                  'xlink:href="data:image/%s;base64,' % self.imageformat,
                  cbytes(data.toBase64()).decode('ascii'),
                  '" preserveAspectRatio="none"' ]
        SVGElement(self.celement, 'image', ''.join(attrb))

    def type(self):
        """A random number for the engine."""
        return qt.QPaintEngine.User + 11

class SVGPaintDevice(qt.QPaintDevice):
    """Paint device for SVG paint engine.

    dpi is the real output DPI (unscaled)
    scale is a scaling value to apply to outputted values
    """

    def __init__(self, fileobj, width_in, height_in,
                 writetextastext=False, dpi=90, scale=0.1):
        qt.QPaintDevice.__init__(self)
        self.fileobj = fileobj
        self.width = width_in
        self.height = height_in
        self.scale = scale
        self.sdpi = dpi/scale
        self.engine = SVGPaintEngine(writetextastext=writetextastext)

    def paintEngine(self):
        return self.engine

    def metric(self, m):
        """Return the metrics of the painter."""

        if m == qt.QPaintDevice.PdmWidth:
            return int(self.width*self.sdpi)
        elif m == qt.QPaintDevice.PdmHeight:
            return int(self.height*self.sdpi)
        elif m == qt.QPaintDevice.PdmWidthMM:
            return int(self.engine.width*inch_mm)
        elif m == qt.QPaintDevice.PdmHeightMM:
            return int(self.engine.height*inch_mm)
        elif m == qt.QPaintDevice.PdmNumColors:
            return 2147483647
        elif m == qt.QPaintDevice.PdmDepth:
            return 24
        elif m == qt.QPaintDevice.PdmDpiX:
            return int(self.sdpi)
        elif m == qt.QPaintDevice.PdmDpiY:
            return int(self.sdpi)
        elif m == qt.QPaintDevice.PdmPhysicalDpiX:
            return int(self.sdpi)
        elif m == qt.QPaintDevice.PdmPhysicalDpiY:
            return int(self.sdpi)
        elif m == qt.QPaintDevice.PdmDevicePixelRatio:
            return 1

        # Qt >= 5.6
        elif m == getattr(qt.QPaintDevice, 'PdmDevicePixelRatioScaled', -1):
            return 1

        else:
            # fall back
            return qt.QPaintDevice.metric(self, m)
