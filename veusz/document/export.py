#    Copyright (C) 2011 Jeremy S. Sanders
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

"""Routines to export the document."""

from __future__ import division
import os.path
import random
import math
import codecs
import re

from ..compat import crange
from .. import qtall as qt4
from .. import utils

try:
    from . import emf_export
    hasemf = True
except ImportError:
    hasemf = False

from . import svg_export
from . import selftest_export
from . import painthelper

# 1m in inch
m_inch = 39.370079

def _(text, disambiguation=None, context="Export"):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

def scalePDFMediaBox(text, pagewidth, reqdsizes):
    """Take the PDF file text and adjust the page size.
    pagewidth: full page width
    reqdsizes: list of tuples of width, height
    """

    outtext = b''
    outidx = 0
    for size, match in zip(
            reqdsizes,
            re.finditer(
                br'^/MediaBox \[([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+)\]$',
                text, re.MULTILINE)):

        box = [float(x) for x in match.groups()]
        widthfactor = (box[2]-box[0])/pagewidth
        newbox = ('/MediaBox [%i %i %i %i]' % (
            int(box[0]),
            int(math.floor(box[3]-widthfactor*size[1])),
            int(math.ceil(box[0]+widthfactor*size[0])),
            int(math.ceil(box[3]))
        )).encode('ascii')

        outtext += text[outidx:match.start()] + newbox
        outidx = match.end()

    outtext += text[outidx:]
    return outtext

def fixupPDFIndices(text):
    """Fixup index table in PDF.

    Basically, we need to find the start of each obj in the file
    These indices are then placed in an xref table at the end
    The index to the xref table is placed after a startxref
    """

    # find occurences of obj in string
    indices = {}
    for m in re.finditer(b'([0-9]+) 0 obj', text):
        index = int(m.group(1))
        indices[index] = m.start(0)

    # build up xref block (note trailing spaces)
    xref = [b'xref', ('0 %i' % (len(indices)+1)).encode('ascii'),
            b'0000000000 65535 f ']
    for i in crange(len(indices)):
        xref.append( ('%010i %05i n ' % (indices[i+1], 0)).encode('ascii') )
    xref.append(b'trailer\n')
    xref = b'\n'.join(xref)

    # replace current xref with this one
    xref_match = re.search(b'^xref\n.*trailer\n', text, re.DOTALL | re.MULTILINE)
    xref_index = xref_match.start(0)
    text = text[:xref_index] + xref + text[xref_match.end(0):]

    # put the correct index to the xref after startxref
    startxref_re = re.compile(b'^startxref\n[0-9]+\n', re.DOTALL | re.MULTILINE)
    text = startxref_re.sub( ('startxref\n%i\n' % xref_index).encode('ascii'),
                             text)

    return text

def fixupPSBoundingBox(infname, outfname, pagewidth, size):
    """Make bounding box for EPS/PS match size given."""
    with open(infname, 'rU') as fin:
        with open(outfname, 'w') as fout:
            for line in fin:
                if line[:14] == '%%BoundingBox:':
                    # replace bounding box line by calculated one
                    parts = line.split()
                    widthfactor = float(parts[3]) / pagewidth
                    origheight = float(parts[4])
                    line = "%s %i %i %i %i\n" % (
                        parts[0], 0,
                        int(math.floor(origheight-widthfactor*size[1])),
                        int(math.ceil(widthfactor*size[0])),
                        int(math.ceil(origheight)) )
                fout.write(line)

class Export(object):
    """Class to do the document exporting.
    
    This is split from document to make that class cleaner.
    """

    formats = [
        (["bmp"], _("Windows bitmap")),
        (["eps"], _("Encapsulated Postscript")),
        (["ps"], _("Postscript")),
        (["jpg"], _("Jpeg bitmap")),
        (["pdf"], _("Portable Document Format")),
        #(["pic"], _("QT Pic format")),
        (["png"], _("Portable Network Graphics")),
        (["svg"], _("Scalable Vector Graphics")),
        (["tiff"], _("Tagged Image File Format bitmap")),
        (["xpm"], _("X Pixmap")),
        ]

    if hasemf:
        formats.append( (["emf"], _("Windows Enhanced Metafile")) )
        formats.sort()

    def __init__(self, doc, filename, pagenumber, color=True, bitmapdpi=100,
                 antialias=True, quality=85, backcolor='#ffffff00',
                 pdfdpi=150, svgtextastext=False):
        """Initialise export class. Parameters are:
        doc: document to write
        filename: output filename
        pagenumber: pagenumber to export or list of pages for some formats
        color: use color or try to use monochrome
        bitmapdpi: assume this dpi value when writing images
        antialias: antialias text and lines when writing bitmaps
        quality: compression factor for bitmaps
        backcolor: background color default for bitmaps (default transparent).
        pdfdpi: dpi for pdf and eps files
        svgtextastext: write text in SVG as text, rather than curves
        """

        self.doc = doc
        self.filename = filename
        self.pagenumber = pagenumber
        self.color = color
        self.bitmapdpi = bitmapdpi
        self.antialias = antialias
        self.quality = quality
        self.backcolor = backcolor
        self.pdfdpi = pdfdpi
        self.svgtextastext = svgtextastext

    def export(self):
        """Export the figure to the filename."""

        ext = os.path.splitext(self.filename)[1].lower()

        if ext in ('.eps', '.ps', '.pdf'):
            self.exportPDFOrPS(ext)

        elif ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.xpm'):
            self.exportBitmap(ext)

        elif ext == '.svg':
            self.exportSVG()

        elif ext == '.selftest':
            self.exportSelfTest()

        elif ext == '.pic':
            self.exportPIC()

        elif ext == '.emf' and hasemf:
            self.exportEMF()

        else:
            raise RuntimeError("File type '%s' not supported" % ext)

    def renderPage(self, page, size, dpi, painter):
        """Render page using paint helper to painter.
        This first renders to the helper, then to the painter
        """
        helper = painthelper.PaintHelper(size, dpi=dpi, directpaint=painter)
        painter.setClipRect( qt4.QRectF(
                qt4.QPointF(0,0), qt4.QPointF(*size)) )
        painter.save()
        self.doc.paintTo(helper, page)
        painter.restore()
        painter.end()

    def getSinglePage(self):
        """Check single number of pages or throw exception,
        else return page number."""

        try:
            if len(self.pagenumber) != 1:
                raise RuntimeError(
                    'Can only export a single page in this format')
            return self.pagenumber[0]
        except TypeError:
            return self.pagenumber

    def exportBitmap(self, ext):
        """Export to a bitmap format."""

        format = ext[1:] # setFormat() doesn't want the leading '.'
        if format == 'jpeg':
            format = 'jpg'

        page = self.getSinglePage()

        # get size for bitmap's dpi
        dpi = self.bitmapdpi
        size = self.doc.pageSize(page, dpi=(dpi,dpi))

        # create real output image
        backqcolor = utils.extendedColorToQColor(self.backcolor)
        if format == 'png':
            # transparent output
            image = qt4.QImage(size[0], size[1],
                               qt4.QImage.Format_ARGB32_Premultiplied)
        else:
            # non transparent output
            image = qt4.QImage(size[0], size[1],
                               qt4.QImage.Format_RGB32)
            backqcolor.setAlpha(255)

        image.setDotsPerMeterX(dpi*m_inch)
        image.setDotsPerMeterY(dpi*m_inch)
        if backqcolor.alpha() == 0:
            image.fill(qt4.qRgba(0,0,0,0))
        else:
            image.fill(backqcolor.rgb())

        # paint to the image
        painter = painthelper.DirectPainter(image)
        painter.setRenderHint(qt4.QPainter.Antialiasing, self.antialias)
        painter.setRenderHint(qt4.QPainter.TextAntialiasing, self.antialias)
        self.renderPage(page, size, (dpi,dpi), painter)

        # write image to disk
        writer = qt4.QImageWriter()
        writer.setFormat(qt4.QByteArray(format))
        writer.setFileName(self.filename)

        # enable LZW compression for TIFFs
        writer.setCompression(1)

        try:
            # try to enable optimal JPEG compression using new
            # options added in Qt 5.5
            writer.setOptimizedWrite(True)
            writer.setProgressiveScanWrite(True)
        except AttributeError:
            pass

        if format == 'png':
            # min quality for png as it makes no difference to output
            # and makes file size smaller
            writer.setQuality(0)
        else:
            writer.setQuality(self.quality)

        writer.write(image)

    def exportPDFOrPS(self, ext):
        """Export to EPS or PDF format."""

        # setup printer with requested parameters
        printer = qt4.QPrinter()
        printer.setResolution(self.pdfdpi)
        printer.setFullPage(True)
        printer.setColorMode(
            qt4.QPrinter.Color if self.color else qt4.QPrinter.GrayScale)
        printer.setOutputFormat(
            qt4.QPrinter.PdfFormat if ext=='.pdf' else
            qt4.QPrinter.PostScriptFormat)
        printer.setOutputFileName(self.filename)
        printer.setCreator('Veusz %s' % utils.version())

        # convert page to list if necessary
        try:
            pages = list(self.pagenumber)
        except TypeError:
            pages = [self.pagenumber]

        if len(pages) != 1 and ext == '.eps':
            raise RuntimeError(
                'Only single pages allowed for .eps. Use .ps instead.')

        # render ranges and return size of each page
        sizes = self.doc.printTo(printer, pages)

        # We have to modify the page sizes or bounding boxes to match
        # the document. This is copied to a temporary file.
        tmpfile = "%s.tmp.%i" % (self.filename, random.randint(0,1000000))

        if ext == '.eps' or ext == '.ps':
            # only 1 size allowed for PS, so use maximum
            maxsize = sizes[0]
            for size in sizes[1:]:
                maxsize = max(size[0], maxsize[0]), max(size[1], maxsize[1])

            fixupPSBoundingBox(self.filename, tmpfile, printer.width(), maxsize)

        elif ext == '.pdf':
            # change pdf bounding box and correct pdf index
            with open(self.filename, 'rb') as fin:
                text = fin.read()
            text = scalePDFMediaBox(text, printer.width(), sizes)
            text = fixupPDFIndices(text)
            with open(tmpfile, 'wb') as fout:
                fout.write(text)
        else:
            raise RuntimeError('Invalid file type')

        # replace original by temporary
        os.remove(self.filename)
        os.rename(tmpfile, self.filename)

    def exportSVG(self):
        """Export document as SVG"""

        page = self.getSinglePage()

        dpi = svg_export.dpi * 1.
        size = self.doc.pageSize(
            page, dpi=(dpi,dpi), integer=False)
        with codecs.open(self.filename, 'w', 'utf-8') as f:
            paintdev = svg_export.SVGPaintDevice(
                f, size[0]/dpi, size[1]/dpi, writetextastext=self.svgtextastext)
            painter = painthelper.DirectPainter(paintdev)
            self.renderPage(page, size, (dpi,dpi), painter)

    def exportSelfTest(self):
        """Export document for testing"""

        page = self.getSinglePage()

        dpi = svg_export.dpi * 1.
        size = width, height = self.doc.pageSize(
            page, dpi=(dpi,dpi), integer=False)

        f = open(self.filename, 'w')
        paintdev = selftest_export.SelfTestPaintDevice(f, width/dpi, height/dpi)
        painter = painthelper.DirectPainter(paintdev)
        self.renderPage(page, size, (dpi,dpi), painter)
        f.close()

    def exportPIC(self):
        """Export document as Qt PIC"""

        page = self.getSinglePage()

        pic = qt4.QPicture()
        painter = painthelper.DirectPainter(pic)

        dpi = (pic.logicalDpiX(), pic.logicalDpiY())
        size = self.doc.pageSize(page, dpi=dpi)
        self.renderPage(page, size, dpi, painter)
        pic.save(self.filename)

    def exportEMF(self):
        """Export document as EMF."""

        page = self.getSinglePage()

        dpi = 90.
        size = self.doc.pageSize(page, dpi=(dpi,dpi), integer=False)
        paintdev = emf_export.EMFPaintDevice(size[0]/dpi, size[1]/dpi, dpi=dpi)
        painter = painthelper.DirectPainter(paintdev)
        self.renderPage(page, size, (dpi,dpi), painter)
        paintdev.paintEngine().saveFile(self.filename)

def printDialog(parentwindow, document, filename=None):
    """Open a print dialog and print document."""

    if document.getNumberPages() == 0:
        qt4.QMessageBox.warning(parentwindow, _("Error - Veusz"),
                                _("No pages to print"))
        return

    prnt = qt4.QPrinter(qt4.QPrinter.HighResolution)
    prnt.setColorMode(qt4.QPrinter.Color)
    prnt.setCreator(_('Veusz %s') % utils.version())
    if filename:
        prnt.setDocName(filename)

    dialog = qt4.QPrintDialog(prnt, parentwindow)
    dialog.setMinMax(1, document.getNumberPages())
    if dialog.exec_():
        # get page range
        if dialog.printRange() == qt4.QAbstractPrintDialog.PageRange:
            # page range
            minval, maxval = dialog.fromPage(), dialog.toPage()
        else:
            # all pages
            minval, maxval = 1, document.getNumberPages()

        # pages are relative to zero
        minval -= 1
        maxval -= 1

        # reverse or forward order
        if prnt.pageOrder() == qt4.QPrinter.FirstPageFirst:
            pages = list(crange(minval, maxval+1))
        else:
            pages = list(crange(maxval, minval-1, -1))

        # if more copies are requested
        pages *= prnt.numCopies()

        # do the printing
        document.printTo(prnt, pages)
