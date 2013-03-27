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

import os.path
import random
import math

import veusz.qtall as qt4
import veusz.utils as utils

try:
    import emf_export
    hasemf = True
except ImportError:
    hasemf = False

import svg_export
import selftest_export
import painthelper

# 1m in inch
m_inch = 39.370079

def _(text, disambiguation=None, context="Export"):
    """Translate text."""
    return unicode(
        qt4.QCoreApplication.translate(context, text, disambiguation))

class Export(object):
    """Class to do the document exporting.
    
    This is split from document to make that class cleaner.
    """

    formats = [
        (["bmp"], _("Windows bitmap")),
        (["eps"], _("Encapsulated Postscript")),
        (["jpg", "jpeg"], _("Jpeg bitmap")),
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
        pagenumber: pagenumber to export
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

        if ext in ('.eps', '.pdf'):
            self.exportPS(ext)

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
            raise RuntimeError, "File type '%s' not supported" % ext

    def renderPage(self, size, dpi, painter):
        """Render page using paint helper to painter.
        This first renders to the helper, then to the painter
        """
        helper = painthelper.PaintHelper(size, dpi=dpi, directpaint=painter)
        painter.setClipRect( qt4.QRectF(
                qt4.QPointF(0,0), qt4.QPointF(*size)) )
        painter.save()
        self.doc.paintTo(helper, self.pagenumber)
        painter.restore()
        painter.end()

    def exportBitmap(self, format):
        """Export to a bitmap format."""

        # get size for bitmap's dpi
        dpi = self.bitmapdpi
        size = self.doc.pageSize(self.pagenumber, dpi=(dpi,dpi))

        # create real output image
        backqcolor = utils.extendedColorToQColor(self.backcolor)
        if format == '.png':
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
        self.renderPage(size, (dpi,dpi), painter)

        # write image to disk
        writer = qt4.QImageWriter()
        # format below takes extension without dot
        writer.setFormat(qt4.QByteArray(format[1:]))
        writer.setFileName(self.filename)

        if format == 'png':
            # min quality for png as it makes no difference to output
            # and makes file size smaller
            writer.setQuality(0)
        else:
            writer.setQuality(self.quality)

        writer.write(image)

    def exportPS(self, ext):
        """Export to EPS or PDF format."""

        printer = qt4.QPrinter()
        printer.setFullPage(True)

        # set printer parameters
        printer.setColorMode( (qt4.QPrinter.GrayScale, qt4.QPrinter.Color)[
                self.color] )

        if ext == '.pdf':
            fmt = qt4.QPrinter.PdfFormat
        else:
            fmt = qt4.QPrinter.PostScriptFormat
        printer.setOutputFormat(fmt)
        printer.setOutputFileName(self.filename)
        printer.setCreator('Veusz %s' % utils.version())
        printer.setResolution(self.pdfdpi)

        # setup for printing
        printer.newPage()
        painter = painthelper.DirectPainter(printer)

        # write to printer with correct dpi
        dpi = (printer.logicalDpiX(), printer.logicalDpiY())
        width, height = size = self.doc.pageSize(self.pagenumber, dpi=dpi)
        self.renderPage(size, dpi, painter)

        # fixup eps/pdf file - yuck HACK! - hope qt gets fixed
        # this makes the bounding box correct
        # copy output to a temporary file
        tmpfile = "%s.tmp.%i" % (self.filename, random.randint(0,1000000))
        fout = open(tmpfile, 'wb')
        fin = open(self.filename, 'rb')

        if ext == '.eps':
            # adjust bounding box
            for line in fin:
                if line[:14] == '%%BoundingBox:':
                    # replace bounding box line by calculated one
                    parts = line.split()
                    widthfactor = float(parts[3]) / printer.width()
                    origheight = float(parts[4])
                    line = "%s %i %i %i %i\n" % (
                        parts[0], 0,
                        int(math.floor(origheight-widthfactor*height)),
                        int(math.ceil(widthfactor*width)),
                        int(math.ceil(origheight)) )
                fout.write(line)

        elif ext == '.pdf':
            # change pdf bounding box and correct pdf index
            text = fin.read()
            text = utils.scalePDFMediaBox(text, printer.width(),
                                          width, height)
            text = utils.fixupPDFIndices(text)
            fout.write(text)

        fout.close()
        fin.close()
        os.remove(self.filename)
        os.rename(tmpfile, self.filename)

    def exportSVG(self):
        """Export document as SVG"""


        if qt4.PYQT_VERSION >= 0x40600:
            # custom paint devices don't work in old PyQt versions

            dpi = svg_export.dpi * 1.
            size = self.doc.pageSize(
                self.pagenumber, dpi=(dpi,dpi), integer=False)
            f = open(self.filename, 'w')
            paintdev = svg_export.SVGPaintDevice(
                f, size[0]/dpi, size[1]/dpi, writetextastext=self.svgtextastext)
            painter = painthelper.DirectPainter(paintdev)
            self.renderPage(size, (dpi,dpi), painter)
            f.close()
        else:
            # use built-in svg generation, which doesn't work very well
            # (no clipping, font size problems)
            import PyQt4.QtSvg

            dpi = 90.
            size = self.doc.pageSize(
                self.pagenumber, dpi=(dpi,dpi), integer=False)

            # actually paint the image
            gen = PyQt4.QtSvg.QSvgGenerator()
            gen.setFileName(self.filename)
            gen.setResolution(dpi)
            gen.setSize( qt4.QSize(int(size[0]), int(size[1])) )
            painter = painthelper.DirectPainter(gen)
            self.renderPage(size, (dpi,dpi), painter)

    def exportSelfTest(self):
        """Export document for testing"""

        dpi = svg_export.dpi * 1.
        size = width, height = self.doc.pageSize(
            self.pagenumber, dpi=(dpi,dpi), integer=False)

        f = open(self.filename, 'w')
        paintdev = selftest_export.SelfTestPaintDevice(f, width/dpi, height/dpi)
        painter = painthelper.DirectPainter(paintdev)
        self.renderPage(size, (dpi,dpi), painter)
        f.close()

    def exportPIC(self):
        """Export document as Qt PIC"""

        pic = qt4.QPicture()
        painter = painthelper.DirectPainter(pic)

        dpi = (pic.logicalDpiX(), pic.logicalDpiY())
        size = self.doc.pageSize(self.pagenumber, dpi=dpi)
        self.renderPage(size, dpi, painter)
        pic.save(self.filename)

    def exportEMF(self):
        """Export document as EMF."""

        dpi = 90.
        size = self.doc.pageSize(self.pagenumber, dpi=(dpi,dpi), integer=False)

        paintdev = emf_export.EMFPaintDevice(size[0]/dpi, size[1]/dpi, dpi=dpi)
        painter = painthelper.DirectPainter(paintdev)
        self.renderPage(size, (dpi,dpi), painter)
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
            pages = range(minval, maxval+1)
        else:
            pages = range(maxval, minval-1, -1)

        # if more copies are requested
        pages *= prnt.numCopies()

        # do the printing
        document.printTo( prnt, pages )
