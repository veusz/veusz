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
import sys
import subprocess

from ..compat import crange
from .. import qtall as qt4
from .. import setting
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

def printPages(doc, printer, pages, scaling=1., antialias=False, setsizes=False):
    """Print onto printing device.
    Returns list of page sizes
    setsizes: Set page size on printer to page sizes
    """

    if not pages:
        return

    dpi = (printer.logicalDpiX(), printer.logicalDpiY())

    def getUpdateSize(page):
        size = doc.pageSize(page, dpi=dpi, integer=False)
        if setsizes:
            # update paper size on printer
            sizeinchx, sizeinchy = size[0]/dpi[0], size[1]/dpi[1]
            pagesize = qt4.QPageSize(
                qt4.QSizeF(sizeinchx, sizeinchy), qt4.QPageSize.Inch)
            layout = qt4.QPageLayout(
                pagesize, qt4.QPageLayout.Portrait, qt4.QMarginsF())
            printer.setPageLayout(layout)
        return size

    size = getUpdateSize(pages[0])
    painter = painthelper.DirectPainter(printer)
    if antialias:
        painter.setRenderHint(qt4.QPainter.Antialiasing, True)
        painter.setRenderHint(qt4.QPainter.TextAntialiasing, True)

    # This all assumes that only pages can go into the root widget
    for count, page in enumerate(pages):
        painter.save()
        painter.setClipRect(qt4.QRectF(
            qt4.QPointF(0,0), qt4.QPointF(*size)))
        helper = painthelper.PaintHelper(
            doc, size, dpi=dpi, directpaint=painter)
        doc.paintTo(helper, page)
        painter.restore()

        # start new pages between each page
        if count < len(pages)-1:
            # set page size before newPage!
            size = getUpdateSize(pages[count+1])
            printer.newPage()

    painter.end()

class Export(object):
    """Class to do the document exporting.

    This is split from document to make that class cleaner.
    """

    # whether ghostscript has been searched for
    gs_searched = False
    # its path if it exists
    gs_exe = None
    # map extensions to ghostscript devices
    gs_dev = None

    @classmethod
    def searchGhostscript(klass):
        """Find location of Ghostscript executable."""
        if not klass.gs_searched:
            # test for existence of ghostscript

            gs_exe = None

            gs = setting.settingdb["external_ghostscript"]
            if gs:
                if os.path.isfile(gs) and os.access(gs, os.X_OK):
                    gs_exe = gs
            else:
                if sys.platform == "win32":
                    # look for ghostscript as 64 and 32 bit versions
                    gs_exe = utils.findOnPath("gswin64c.exe")
                    if not gs_exe:
                        gs_exe = utils.findOnPath("gswin32c.exe")
                else:
                    # unix tends to call it just gs
                    gs_exe = utils.findOnPath("gs")

            klass.gs_dev = dev = {}
            if gs_exe:
                try:
                    # check output devices contain
                    #  ps2write/eps2write or pswrite/epswrite
                    popen = subprocess.Popen(
                        [gs_exe, '-h'],
                        stdout=subprocess.PIPE,
                        universal_newlines=True)
                    text = popen.stdout.read()

                    if re.search(r'\beps2write\b', text):
                        dev['.eps'] = 'eps2write'
                    elif re.search(r'\bepswrite\b', text):
                        dev['.eps'] = 'epswrite'
                    if re.search(r'\bps2write\b', text):
                        dev['.ps'] = 'ps2write'
                    elif re.search(r'\bpswrite\b', text):
                        dev['.ps'] = 'pswrite'
                except Exception as e:
                    pass
                else:
                    klass.gs_exe = gs_exe

            klass.gs_searched = True

    @classmethod
    def getFormats(klass):
        """Return list of formats in form of tuples of extension and description."""
        formats = [
            (["bmp"], _("Windows bitmap")),
            (["jpg"], _("Jpeg bitmap")),
            (["pdf"], _("Portable Document Format")),
            (["png"], _("Portable Network Graphics")),
            (["svg"], _("Scalable Vector Graphics")),
            (["tiff"], _("Tagged Image File Format bitmap")),
            (["xpm"], _("X Pixmap")),
        ]

        if hasemf:
            formats.append( (["emf"], _("Windows Enhanced Metafile")) )

        klass.searchGhostscript()
        if '.eps' in klass.gs_dev:
            formats.append((["eps"], _("Encapsulated Postscript")))
        if '.ps' in klass.gs_dev:
            formats.append((["ps"], _("Postscript")))

        formats.sort()
        return formats

    def __init__(self, doc, filename, pagenumbers, color=True, bitmapdpi=100,
                 antialias=True, quality=85, backcolor='#ffffff00',
                 pdfdpi=150, svgtextastext=False):
        """Initialise export class. Parameters are:
        doc: document to write
        filename: output filename
        pagenumbers: list of pages to export
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
        self.pagenumbers = pagenumbers
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

        if ext == '.pdf':
            self.exportPDF(self.filename)

        elif ext in ('.eps', '.ps'):
            self.exportPS(self.filename, ext)

        elif ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.xpm'):
            self.exportBitmap(self.filename, ext)

        elif ext == '.svg':
            self.exportSVG(self.filename)

        elif ext == '.selftest':
            self.exportSelfTest(self.filename)

        elif ext == '.pic':
            self.exportPIC(self.filename)

        elif ext == '.emf' and hasemf:
            self.exportEMF(self.filename)

        else:
            raise RuntimeError("File type '%s' not supported" % ext)

    def renderPage(self, page, size, dpi, painter):
        """Render page using paint helper to painter.
        This first renders to the helper, then to the painter
        """
        helper = painthelper.PaintHelper(self.doc, size, dpi=dpi, directpaint=painter)
        painter.setClipRect( qt4.QRectF(
                qt4.QPointF(0,0), qt4.QPointF(*size)) )
        painter.save()
        self.doc.paintTo(helper, page)
        painter.restore()
        painter.end()

    def getSinglePage(self):
        """Check single number of pages or throw exception,
        else return page number."""

        if len(self.pagenumbers) != 1:
            raise RuntimeError(
                'Can only export a single page in this format')
        return self.pagenumbers[0]

    def exportBitmap(self, filename, ext):
        """Export to a bitmap format."""

        fmt = ext.lstrip('.') # setFormat() doesn't want the leading '.'
        if fmt == 'jpeg':
            fmt = 'jpg'

        page = self.getSinglePage()

        # get size for bitmap's dpi
        dpi = self.bitmapdpi
        size = self.doc.pageSize(page, dpi=(dpi,dpi))

        # create real output image
        backqcolor = self.doc.evaluate.colors.get(self.backcolor)
        if fmt == 'png':
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
        writer.setFormat(fmt.encode('ascii'))
        writer.setFileName(filename)

        # enable LZW compression for TIFFs
        writer.setCompression(1)

        try:
            # try to enable optimal JPEG compression using new
            # options added in Qt 5.5
            writer.setOptimizedWrite(True)
            writer.setProgressiveScanWrite(True)
        except AttributeError:
            pass

        if fmt == 'png':
            # min quality for png as it makes no difference to output
            # and makes file size smaller
            writer.setQuality(0)
        else:
            writer.setQuality(self.quality)

        writer.write(image)

    def exportPDF(self, filename):
        """Export to PDF format."""

        # setup printer with requested parameters
        printer = qt4.QPrinter()
        printer.setResolution(self.pdfdpi)
        printer.setFullPage(True)
        printer.setColorMode(
            qt4.QPrinter.Color if self.color else qt4.QPrinter.GrayScale)
        printer.setOutputFormat(qt4.QPrinter.PdfFormat)
        printer.setOutputFileName(filename)
        printer.setCreator('Veusz %s' % utils.version())

        printPages(self.doc, printer, self.pagenumbers, setsizes=True)

    def exportPS(self, filename, ext):
        """Export to PS/EPS via conversion with Ghostscript.

        ext == '.eps' or '.ps'
        """

        if len(self.pagenumbers) != 1 and ext == '.eps':
            raise RuntimeError(
                'Only single pages allowed for .eps. Use .ps instead.')

        self.searchGhostscript()
        if not self.gs_exe:
            raise RuntimeError("Cannot write Postscript with Ghostscript")

        # write to pdf file first
        tmpfilepdf = "%s.tmp.%i.pdf" % (
            filename, random.randint(0,1000000))
        tmpfileps = "%s.tmp.%i%s" % (
            filename, random.randint(0,1000000), ext)

        self.exportPDF(tmpfilepdf)

        # run ghostscript to covert from pdf to postscript
        cmd = [
            self.gs_exe,
            '-q', '-dNOCACHE', '-dNOPAUSE', '-dBATCH', '-dSAFER',
            '-sDEVICE=%s' % self.gs_dev[ext],
            '-sOutputFile=%s' % tmpfileps,
            tmpfilepdf
            ]
        try:
            subprocess.check_call(cmd)
        except Exception as e:
            raise RuntimeError("Could not run ghostscript: %s" % str(e))

        if not os.path.isfile(tmpfileps):
            raise RuntimeError("Ghostscript failed to create %s" % tmpfileps)

        os.remove(tmpfilepdf)
        try:
            os.remove(filename)
        except OSError:
            pass
        os.rename(tmpfileps, filename)

    def exportSVG(self, filename):
        """Export document as SVG"""

        page = self.getSinglePage()

        dpi = svg_export.dpi * 1.
        size = self.doc.pageSize(
            page, dpi=(dpi,dpi), integer=False)
        with codecs.open(filename, 'w', 'utf-8') as f:
            paintdev = svg_export.SVGPaintDevice(
                f, size[0]/dpi, size[1]/dpi, writetextastext=self.svgtextastext)
            painter = painthelper.DirectPainter(paintdev)
            self.renderPage(page, size, (dpi,dpi), painter)

    def exportSelfTest(self, filename):
        """Export document for testing"""

        page = self.getSinglePage()

        dpi = svg_export.dpi * 1.
        size = width, height = self.doc.pageSize(
            page, dpi=(dpi,dpi), integer=False)

        with open(filename, 'w') as fout:
            paintdev = selftest_export.SelfTestPaintDevice(
                fout, width/dpi, height/dpi)
            painter = painthelper.DirectPainter(paintdev)
            self.renderPage(page, size, (dpi,dpi), painter)

    def exportPIC(self, filename):
        """Export document as Qt PIC"""

        page = self.getSinglePage()

        pic = qt4.QPicture()
        painter = painthelper.DirectPainter(pic)

        dpi = (pic.logicalDpiX(), pic.logicalDpiY())
        size = self.doc.pageSize(page, dpi=dpi)
        self.renderPage(page, size, dpi, painter)
        pic.save(filename)

    def exportEMF(self, filename):
        """Export document as EMF."""

        page = self.getSinglePage()

        dpi = 90.
        size = self.doc.pageSize(page, dpi=(dpi,dpi), integer=False)
        paintdev = emf_export.EMFPaintDevice(size[0]/dpi, size[1]/dpi, dpi=dpi)
        painter = painthelper.DirectPainter(paintdev)
        self.renderPage(page, size, (dpi,dpi), painter)
        paintdev.paintEngine().saveFile(filename)

def printDialog(parentwindow, document, filename=None):
    """Open a print dialog and print document."""

    if document.getNumberPages() == 0:
        qt4.QMessageBox.warning(
            parentwindow, _("Error - Veusz"), _("No pages to print"))
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
        pages *= prnt.copyCount()

        # do the printing
        printPages(document, prnt, pages)
