#    Copyright (C) 2014 Jeremy S. Sanders
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

from __future__ import division, print_function
import os.path

from .. import qtall as qt4
from .. import setting
from .. import utils
from .. import document
from ..compat import citems
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context='ExportDialog'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

# used in filename to mark page number
PAGENUM = '%PAGE%'

# formats which can have multiple pages
multipageformats = set(('ps', 'pdf'))

# map formats to names of radio buttons
formatradio = (
    ('pdf', 'radioFormatPDF'),
    ('eps', 'radioFormatEPS'),
    ('ps',  'radioFormatPS' ),
    ('svg', 'radioFormatSVG'),
    ('emf', 'radioFormatEMF'),
    ('png', 'radioFormatPNG'),
    ('bmp', 'radioFormatBMP'),
    ('jpg', 'radioFormatJPG'),
    ('tiff', 'radioFormatTIFF'),
    ('xpm', 'radioFormatXPM'),
)

class ExportDialog(VeuszDialog):
    """Export dialog."""

    def __init__(self, mainwindow, doc, docfilename):
        """Setup dialog."""
        VeuszDialog.__init__(self, mainwindow, 'export.ui')

        self.document = doc
        doc.signalModified.connect(self.updatePageMinMax)
        self.updatePageMinMax()

        # change 'Save' button to 'Export'
        self.buttonBox.button(qt4.QDialogButtonBox.Save).setText(_('Export'))

        # these are mappings between filetypes and radio buttons
        self.fmtradios = dict([(f, getattr(self, r)) for f, r in formatradio])
        self.radiofmts = dict([(getattr(self, r), f) for f, r in formatradio])

        # get allowed types (some formats are disabled if no helper)
        docfmts = set()
        for types, descr in document.Export.formats:
            docfmts.update(types)
        # disable type if not allowed
        for fmt, radio in citems(self.fmtradios):
            if fmt not in docfmts:
                radio.setEnabled(False)

        # connect format radio buttons
        def fmtclicked(f):
            return lambda: self.formatClicked(f)
        for r, f in citems(self.radiofmts):
            r.clicked.connect(fmtclicked(f))

        # connect page radio buttons
        self.radioPageSingle.clicked.connect(lambda: self.pageClicked('single'))
        self.radioPageAll.clicked.connect(lambda: self.pageClicked('all'))
        self.radioPageRange.clicked.connect(lambda: self.pageClicked('range'))

        self.checkMultiPage.clicked.connect(self.updateFilename)

        # set default filename
        ext = setting.settingdb.get('export_format', 'pdf')
        if not docfilename:
            docfilename = 'export'
        filename = os.path.join(
            setting.settingdb['dirname_export'],
            os.path.splitext(os.path.basename(docfilename))[0] + '.' + ext)
        self.editFileName.setText(filename)

        self.formatselected = ext
        self.pageselected = setting.settingdb.get('export_page', 'single')

        self.checkMultiPage.setChecked(
            setting.settingdb.get('export_multipage', True))

        # set correct format
        self.fmtradios[ext].click()

        # set page mode
        {
            'single': self.radioPageSingle,
            'all': self.radioPageAll,
            'range': self.radioPageRange,
        }[self.pageselected].click()

    def formatClicked(self, fmt):
        """If the format is changed."""
        setting.settingdb['export_format'] = fmt
        self.formatselected = fmt
        self.checkMultiPage.setEnabled(fmt in multipageformats)
        self.updateFilename()

    def pageClicked(self, page):
        """If page type is set."""
        setting.settingdb['export_page'] = page
        self.pageselected = page
        self.updateFilename()

        self.pageRangeMin.setEnabled(page=='range')
        self.pageRangeMax.setEnabled(page=='range')

    def isMultiFile(self):
        """Is output going to be multiple pages?"""
        multipage = self.pageselected != 'single'
        if (self.formatselected in multipageformats and
            self.checkMultiPage.isChecked()):
            multipage = False
        return multipage

    def updateFilename(self):
        """Change filename according to selected radio buttons."""
        filename = self.editFileName.text()
        setting.settingdb['export_multipage'] = self.checkMultiPage.isChecked()
        if filename:
            dotpos = filename.rfind('.')
            left = filename if dotpos==-1 else filename[:dotpos]
            multifile = self.isMultiFile()

            if multifile and PAGENUM not in os.path.basename(left):
                left += PAGENUM
            elif not multifile and PAGENUM in os.path.basename(left):
                left = left.replace(PAGENUM, '')

            newfilename = left + '.' + self.formatselected
            self.editFileName.setText(newfilename)

    def updatePageMinMax(self):
        """Update widgets allowing user to set page range."""
        npages = self.document.getNumberPages()
        if self.pageRangeMax.value() == 0:
            self.pageRangeMax.setValue(npages)
        self.pageRangeMin.setMinimum(1)
        self.pageRangeMin.setMaximum(npages)
        self.pageRangeMax.setMinimum(1)
        self.pageRangeMax.setMaximum(npages)

    def showMessage(self, text):
        """Show a message in a label, clearing after a time."""
        self.labelStatus.setText(text)
        qt4.QTimer.singleShot(3000, lambda: self.labelStatus.clear())

    def accept(self):
        """Do the export"""

        filename = self.editFileName.text()
        if self.isMultiFile() and PAGENUM not in os.path.basename(filename):
            self.showMessage(_('Error: %s not in filename') % PAGENUM)
            return
        if self.pageselected == 'range' and (
                self.pageRangeMin.value() > self.pageRangeMax.value()):
            self.showMessage(_('Error: page range invalid'))
            return

        if self.pageselected == 'single':
            pages = [self.mainwindow.plot.getPageNumber()]
        elif self.pageselected == 'all':
            pages = list(range(self.document.getNumberPages()))
        elif self.pageselected == 'range':
            pages = list(range(
                self.pageRangeMin.value()-1, self.pageRangeMax.value()))

        setdb = setting.settingdb
        export = document.Export(
            self.document,
            '',  # filename
            0,   # page number
            bitmapdpi=setdb['export_DPI'],
            pdfdpi=setdb['export_DPI_PDF'],
            antialias=setdb['export_antialias'],
            color=setdb['export_color'],
            quality=setdb['export_quality'],
            backcolor=setdb['export_background'],
            svgtextastext=setdb['export_SVG_text_as_text'],
        )

        if self.isMultiFile():
            # write pages to multiple files
            for page in pages:
                export.pagenumber = page
                export.filename = os.path.join(
                    os.path.dirname(filename),
                    os.path.basename(filename).replace(PAGENUM, str(page+1)))
                export.export()
        else:
            # write page/pages to single file
            export.pagenumber = pages
            export.filename = filename
            export.export()

        self.showMessage(_('Exported %i page(s)') % len(pages))
