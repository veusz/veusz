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
from .. import qtall as qt
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

# scale factor for svg dpi
svg_dpi_scale = 0.1

def _(text, disambiguation=None, context="Export"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

def getSinglePage(pagenumbers):
    """Check single number of pages or throw exception, else return page number."""

    if len(pagenumbers) != 1:
        raise RuntimeError(
            'Can only export a single page in this format')
    return pagenumbers[0]

class ExportRunnable(qt.QRunnable):
    """For running export in thread."""

    def __init__(self, aexport, filename, phelpers):
        qt.QRunnable.__init__(self)
        self.aexport = aexport
        self.filename = filename
        self.phelpers = phelpers

    def run(self):
        """Do export.

        Exceptions are returned to AsyncExport class."""
        try:
            self.doExport()
        except (RuntimeError, EnvironmentError) as e:
            self.aexport.exception = e

    def renderPage(self, dev, phelper):
        """Render page, clipping."""
        painter = qt.QPainter(dev)
        painter.setClipRect(qt.QRectF(
            qt.QPointF(0,0), qt.QPointF(*phelper.pagesize)))
        painter.save()
        phelper.renderToPainter(painter)
        painter.restore()
        painter.end()

class ExportBitmapRunnable(ExportRunnable):
    """Runnable task to export a bitmap."""

    def doExport(self):
        """Do the export."""
        ext = os.path.splitext(self.filename)[1].lower()
        fmt = ext.lstrip('.') # setFormat() doesn't want the leading '.'
        if fmt == 'jpeg':
            fmt = 'jpg'

        # create real output image
        size = self.phelpers[0].pagesize
        backqcolor = self.aexport.backqcolor
        if fmt == 'png':
            # transparent output
            image = qt.QImage(
                size[0], size[1],
                qt.QImage.Format_ARGB32_Premultiplied)
        else:
            # non transparent output
            image = qt.QImage(
                size[0], size[1],
                qt.QImage.Format_RGB32)
            backqcolor.setAlpha(255)

        image.setDotsPerMeterX(self.phelpers[0].dpi[0]*m_inch)
        image.setDotsPerMeterY(self.phelpers[0].dpi[1]*m_inch)
        if backqcolor.alpha() == 0:
            image.fill(qt.qRgba(0,0,0,0))
        else:
            image.fill(backqcolor.rgb())

        # paint to the image
        painter = qt.QPainter(image)
        painter.setRenderHint(qt.QPainter.Antialiasing, self.aexport.antialias)
        painter.setRenderHint(qt.QPainter.TextAntialiasing, self.aexport.antialias)
        self.phelpers[0].renderToPainter(painter)
        painter.end()

        # write image to disk
        writer = qt.QImageWriter()
        writer.setFormat(fmt.encode('ascii'))
        writer.setFileName(self.filename)

        if fmt == 'png':
            # max compression for PNGs (this number comes from the
            # source code)
            writer.setCompression(100)
            writer.setQuality(0)
        elif fmt == 'tiff':
            # enable LZW compression for TIFFs
            writer.setCompression(1)
        elif fmt == 'jpg':
            # enable optimal JPEG compression using new Qt 5.5 options
            writer.setOptimizedWrite(True)
            writer.setProgressiveScanWrite(True)

        if fmt != 'png':
            writer.setQuality(self.aexport.quality)

        writer.write(image)

class ExportPDFRunnable(ExportRunnable):
    """Runnable task to export a PDF file."""

    def doExport(self):
        """Do the export."""
        printer = qt.QPrinter()
        printer.setResolution(self.aexport.pdfdpi)
        printer.setFullPage(True)
        printer.setColorMode(
            qt.QPrinter.Color if self.aexport.color else qt.QPrinter.GrayScale)
        printer.setOutputFormat(qt.QPrinter.PdfFormat)
        printer.setOutputFileName(self.filename)
        printer.setCreator('Veusz %s' % utils.version())

        def updateSize(ph):
            """Update page size in QPrinter"""
            sizeinchx, sizeinchy = ph.pagesize[0]/ph.dpi[0], ph.pagesize[1]/ph.dpi[1]
            pagesize = qt.QPageSize(
                qt.QSizeF(sizeinchx, sizeinchy), qt.QPageSize.Inch)
            layout = qt.QPageLayout(
                pagesize, qt.QPageLayout.Portrait, qt.QMarginsF())
            printer.setPageLayout(layout)

        updateSize(self.phelpers[0])

        painter = qt.QPainter(printer)
        for i, phelper in enumerate(self.phelpers):
            if i>0:
                updateSize(phelper)
                printer.newPage()
            phelper.renderToPainter(painter)
        painter.end()

class ExportPostscriptRunnable(ExportRunnable):
    """Task to export .ps/.eps files."""

    # whether ghostscript has been searched for
    gs_searched = False
    # its path if it exists
    gs_exe = None
    # map extensions to ghostscript devices
    gs_dev = None

    @classmethod
    def searchGhostscript(klass):
        """Find location of Ghostscript executable."""
        if klass.gs_searched:
            return

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

    def doExport(self):
        """Do the export."""

        self.searchGhostscript()
        if not self.gs_exe:
            raise RuntimeError("Cannot write Postscript without Ghostscript available")

        # write to pdf file first
        ext = os.path.splitext(self.filename)[1].lower()
        tmpfilepdf = "%s.tmp.%i.pdf" % (
            self.filename, random.randint(0,1000000))
        tmpfileps = "%s.tmp.%i%s" % (
            self.filename, random.randint(0,1000000), ext)

        pdfrunnable = ExportPDFRunnable(self.aexport, tmpfilepdf, self.phelpers)
        pdfrunnable.run()

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
            os.remove(self.filename)
        except OSError:
            pass
        os.rename(tmpfileps, self.filename)

class ExportSVGRunnable(ExportRunnable):
    """Runnable task to export an SVG file."""

    def doExport(self):
        sdpi = self.phelpers[0].dpi
        size = self.phelpers[0].pagesize
        with codecs.open(self.filename, 'w', 'utf-8') as f:
            paintdev = svg_export.SVGPaintDevice(
                f,
                size[0]/sdpi[0], size[1]/sdpi[1],
                writetextastext=self.aexport.svgtextastext,
                dpi=sdpi[1]*svg_dpi_scale,
                scale=svg_dpi_scale)
            self.renderPage(paintdev, self.phelpers[0])

class ExportSelfTestRunnable(ExportRunnable):
    """Runnable task to export a self-test output."""

    def doExport(self):
        sdpi = self.phelpers[0].dpi
        size = self.phelpers[0].pagesize
        with codecs.open(self.filename, 'w', 'utf-8') as f:
            paintdev = selftest_export.SelfTestPaintDevice(
                f,
                size[0]/sdpi[0], size[1]/sdpi[1],
                dpi=sdpi[1])
            self.renderPage(paintdev, self.phelpers[0])

class ExportPICRunnable(ExportRunnable):
    """Runnable task to export Qt PIC output."""

    def doExport(self):
        paintdev = qt.QPicture()
        self.renderPage(paintdev, self.phelpers[0])
        paintdev.save(self.filename)

class ExportEMFRunnable(ExportRunnable):
    """Runnable task to export EMF output."""

    def doExport(self):
        dpi = self.phelpers[0].dpi
        size = self.phelpers[0].pagesize
        paintdev = emf_export.EMFPaintDevice(
            size[0]/dpi[0], size[1]/dpi[1], dpi=dpi[1])
        self.renderPage(paintdev, self.phelpers[0])
        paintdev.paintEngine().saveFile(self.filename)

class AsyncExport(qt.QObject):
    """Asynchronous export.

    Add export tasks with add() and wait with finish().
    """

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

        ExportPostscriptRunnable.searchGhostscript()
        if '.eps' in ExportPostscriptRunnable.gs_dev:
            formats.append((["eps"], _("Encapsulated Postscript")))
        if '.ps' in ExportPostscriptRunnable.gs_dev:
            formats.append((["ps"], _("Postscript")))

        formats.sort()
        return formats

    def __init__(self, doc, color=True, bitmapdpi=100,
                 antialias=True, quality=85, backcolor='#ffffff00',
                 pdfdpi=150, svgdpi=96, svgtextastext=False):
        """Initialise export class. Parameters are:
        doc: document to write
        color: use color or try to use monochrome
        bitmapdpi: assume this dpi value when writing images
        antialias: antialias text and lines when writing bitmaps
        quality: compression factor for bitmaps
        backcolor: background color default for bitmaps (default transparent).
        pdfdpi: dpi for pdf and eps files
        svgdpi: dpi for svg files
        svgtextastext: write text in SVG as text, rather than curves
        """

        qt.QObject.__init__(self)
        self.doc = doc
        self.color = color
        self.bitmapdpi = bitmapdpi
        self.antialias = antialias
        self.quality = quality
        self.backcolor = backcolor
        self.pdfdpi = pdfdpi
        self.svgdpi = svgdpi
        self.svgtextastext = svgtextastext

        self.backqcolor = self.doc.evaluate.colors.get(self.backcolor)

        # any exceptions in runnables should set this to be reported
        # in main thread for UI
        self.exception = None

        # pool that export threads use to execute
        self.pool = qt.QThreadPool(self)
        self.pool.setMaxThreadCount(
            max(setting.settingdb['plot_numthreads'], 1)
        )

    def finish(self):
        self.pool.waitForDone()

        if self.exception is not None:
            exception = self.exception
            self.exception = None
            raise exception

    def haveDone(self):
        """Have all the threads finished?

        Note: call finish afterwards for cleanup
        """
        return self.pool.waitForDone(0)

    def getDPI(self, ext):
        """Get DPI to use for filename extension."""

        if ext in {'.pdf', '.eps', '.ps'}:
            return (self.pdfdpi, self.pdfdpi)

        elif ext in {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.xpm'}:
            return (self.bitmapdpi, self.bitmapdpi)

        elif ext == '.svg':
            dpi = self.svgdpi / svg_dpi_scale
            return (dpi, dpi)

        elif ext == '.selftest':
            return (90, 90)

        elif ext == '.pic':
            pic = qt.QPicture()
            return (pic.logicalDpiX(), pic.logicalDpiY())

        elif ext == '.emf':
            return (90, 90)

        else:
            raise RuntimeError('Unknown export file type')

    def add(self, filename, pages):
        """Add export to list to be processed.

        filename: output filename
        pages: list of pages (0-indexed) to be added.

        sync: if True, then do not execute in different thread."""

        ext = os.path.splitext(filename)[1].lower()
        dpi = self.getDPI(ext)

        # render each page to a PaintHelper
        phelpers = []
        for page in pages:
            size = self.doc.pageSize(page, dpi=dpi, integer=False)
            phelper = painthelper.PaintHelper(self.doc, size, dpi=dpi)
            self.doc.paintTo(phelper, page)
            phelpers.append(phelper)

        # single page only formats
        if len(phelpers) != 1 and ext not in ('.ps', '.pdf'):
            raise RuntimeError('Only single page allowed for format')

        # make a runnable task for the right file type
        runnable = {
            '.png': ExportBitmapRunnable,
            '.jpg': ExportBitmapRunnable,
            '.jpeg': ExportBitmapRunnable,
            '.bmp': ExportBitmapRunnable,
            '.tiff': ExportBitmapRunnable,
            '.xpm': ExportBitmapRunnable,

            '.pdf': ExportPDFRunnable,
            '.ps': ExportPostscriptRunnable,
            '.eps': ExportPostscriptRunnable,

            '.svg': ExportSVGRunnable,
            '.selftest': ExportSelfTestRunnable,
            '.pic': ExportPICRunnable,
            '.emf': ExportEMFRunnable,
            }[ext](self, filename, phelpers)

        self.pool.start(runnable)

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
            pagesize = qt.QPageSize(
                qt.QSizeF(sizeinchx, sizeinchy), qt.QPageSize.Inch)
            layout = qt.QPageLayout(
                pagesize, qt.QPageLayout.Portrait, qt.QMarginsF())
            printer.setPageLayout(layout)
        return size

    size = getUpdateSize(pages[0])

    painter = painthelper.DirectPainter(printer)
    if antialias:
        painter.setRenderHint(qt.QPainter.Antialiasing, True)
        painter.setRenderHint(qt.QPainter.TextAntialiasing, True)

    # This all assumes that only pages can go into the root widget
    visible = set(doc.getVisiblePages())
    filtpages = [page for page in pages if page in visible]

    for count, page in enumerate(filtpages):
        psize = doc.pageSize(page, dpi=dpi, integer=False, scaling=scaling)
        phelper = painthelper.PaintHelper(doc, psize, dpi=dpi, scaling=scaling)
        doc.paintTo(phelper, page)
        phelper.renderToPainter(painter)

        # start new pages between each page
        if count < len(filtpages)-1:
            # set page size before newPage!
            size = getUpdateSize(pages[count+1])
            printer.newPage()

    painter.end()

def printDialog(parentwindow, document, filename=None):
    """Open a print dialog and print document."""

    if not document.getVisiblePages():
        qt.QMessageBox.warning(
            parentwindow, _("Error - Veusz"), _("No pages to print"))
        return

    prnt = qt.QPrinter(qt.QPrinter.HighResolution)
    prnt.setColorMode(qt.QPrinter.Color)
    prnt.setCreator(_('Veusz %s') % utils.version())
    if filename:
        prnt.setDocName(filename)

    dialog = qt.QPrintDialog(prnt, parentwindow)
    dialog.setMinMax(1, document.getNumberPages())
    if dialog.exec_():
        # get page range
        if dialog.printRange() == qt.QAbstractPrintDialog.PageRange:
            # page range
            minval, maxval = dialog.fromPage(), dialog.toPage()
        else:
            # all pages
            minval, maxval = 1, document.getNumberPages()

        # pages are relative to zero
        minval -= 1
        maxval -= 1

        # reverse or forward order
        if prnt.pageOrder() == qt.QPrinter.FirstPageFirst:
            pages = list(crange(minval, maxval+1))
        else:
            pages = list(crange(maxval, minval-1, -1))

        # if more copies are requested
        pages *= prnt.copyCount()

        # do the printing
        printPages(document, prnt, pages)
