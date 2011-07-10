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

class Export(object):
    """Class to do the document exporting.
    
    This is split from document to make that class cleaner.
    """

    formats = [
        (["bmp"], "Windows bitmap"),
        (["eps"], "Encapsulated Postscript"),
        (["jpg"], "Jpeg bitmap"),
        (["pdf"], "Portable Document Format"),
        #(["pic"], "QT Pic format"),
        (["png"], "Portable Network Graphics"),
        (["svg"], "Scalable Vector Graphics"),
        (["tiff"], "Tagged Image File Format bitmap"),
        (["xpm"], "X Pixmap"),
        ]

    if hasemf:
        formats.append( (["emf"], "Windows Enhanced Metafile") )
        formats.sort()

    def __init__(self, doc, filename, pagenumber, color=True, bitmapdpi=100,
                 antialias=True, quality=85, backcolor='#ffffff00'):
        """Initialise export class. Parameters are:
        doc: document to write
        filename: output filename
        pagenumber: pagenumber to export
        color: use color or try to use monochrome
        bitmapdpi: assume this dpi value when writing images
        antialias: antialias text and lines when writing bitmaps
        quality: compression factor for bitmaps
        backcolor: background color default for bitmaps (default transparent)."""

        self.doc = doc
        self.filename = filename
        self.pagenumber = pagenumber
        self.color = color
        self.bitmapdpi = bitmapdpi
        self.antialias = antialias
        self.quality = quality
        self.backcolor = backcolor
        
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

    def renderPage(self, phelper, painter):
        """Render page using paint helper to painter.
        This first renders to the helper, then to the painter
        """
        self.doc.paintTo(phelper, self.pagenumber)
        phelper.renderToPainter(self.doc.basewidget, painter)

    def exportBitmap(self, format):
        """Export to a bitmap format."""

        # work out what scaling is required to get from
        # Qt system dpi to required dpi
        # also work out scaled image size
        helper = painthelper.PaintHelper( (1,1) )
        scaling = self.bitmapdpi * 1. / helper.dpi
        width, height = self.doc.basewidget.getSize(helper)
        newsize = (int(width*scaling), int(height*scaling))

        # create real output image
        pixmap = qt4.QPixmap(*newsize)
        backqcolor = utils.extendedColorToQColor(self.backcolor)
        if format != '.png':
            # not transparent
            backqcolor.setAlpha(255)
        pixmap.fill(backqcolor)

        # paint to the image
        painter = qt4.QPainter(pixmap)
        painter.setRenderHint(qt4.QPainter.Antialiasing, self.antialias)
        painter.setRenderHint(qt4.QPainter.TextAntialiasing, self.antialias)
        helper = painthelper.PaintHelper(newsize, scaling=scaling)
        self.renderPage(helper, painter)
        painter.end()

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

        writer.write( pixmap.toImage() )

    def exportPS(self, ext):
        """Export to EPS or PDF format."""

        helper = painthelper.PaintHelper( (1,1) )
        size = self.doc.basewidget.getSize(helper)
        helper = painthelper.PaintHelper( size )

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

        # draw the page
        printer.newPage()
        painter = qt4.QPainter(printer)
        self.renderPage(helper, painter)
        painter.end()

        # we now need to adjust the output size
        # the helper's width and height are scaled by helper dpi vs painter dpi
        scaling = printer.logicalDpiY() / helper.dpi
        width, height = helper.pagesize[0]*scaling, helper.pagesize[1]*scaling

        # fixup eps/pdf file - yuck HACK! - hope qt gets fixed
        # this makes the bounding box correct
        if ext == '.eps' or ext == '.pdf':
            # copy eps to a temporary file
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
                text = utils.scalePDFMediaBox(text,
                                              printer.width(),
                                              width, height)
                text = utils.fixupPDFIndices(text)
                fout.write(text)

            fout.close()
            fin.close()
            os.remove(self.filename)
            os.rename(tmpfile, self.filename)

    def exportSVG(self):
        """Export document as SVG"""

        dpi = 90.
        helper = painthelper.PaintHelper( (1,1) )
        size = self.doc.basewidget.getSize(helper)
        scaling = dpi / helper.dpi
        width, height = size[0]*scaling, size[1]*scaling

        helper = painthelper.PaintHelper( size )

        if qt4.PYQT_VERSION >= 0x40600:
            # custom paint devices don't work in old PyQt versions

            f = open(self.filename, 'w')
            paintdev = svg_export.SVGPaintDevice(f, width/dpi, height/dpi)
            painter = qt4.QPainter(paintdev)
            self.renderPage(helper, painter)
            painter.end()
            f.close()

        else:
            # use built-in svg generation, which doesn't work very well
            # (no clipping, font size problems)
            import PyQt4.QtSvg

            # actually paint the image
            gen = PyQt4.QtSvg.QSvgGenerator()
            gen.setFileName(self.filename)
            gen.setResolution(dpi)
            gen.setSize( qt4.QSize(int(width), int(height)) )
            painter = qt4.QPainter(gen)
            self.renderPage(helper, painter)
            painter.end()

    def exportSelfTest(self, filename, page):
        """Export document for testing"""

        dpi = 90.
        width, height = self._getDocSize(dpi)

        f = open(filename, 'w')
        paintdev = selftest_export.SelfTestPaintDevice(f, width/dpi, height/dpi)
        painter = Painter(paintdev)
        self.basewidget.draw(painter, page)
        painter.end()
        f.close()

    def exportPIC(self, filename, page):
        """Export document as Qt PIC"""

        pic = qt4.QPicture()
        painter = Painter(pic)
        self.basewidget.draw( painter, page )
        painter.end()
        pic.save(filename)

    def exportEMF(self, filename, page):
        """Export document as EMF."""

        dpi = 90.
        width, height = self._getDocSize(dpi)

        paintdev = emf_export.EMFPaintDevice(width/dpi, height/dpi,
                                             dpi=dpi)
        painter = Painter(paintdev)
        self.basewidget.draw( painter, page )
        painter.end()
        paintdev.paintEngine().saveFile(filename)

