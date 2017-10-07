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
import os
import os.path

from .. import qtall as qt4
from .. import setting
from .. import utils
from .. import document
from ..compat import citems, cstrerror, cstr, cgetcwd
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context='ExportDialog'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

# used in filename to mark page number
PAGENUM = '%PAGENUM%'
PAGENAME = '%PAGENAME%'

# formats which can have multiple pages
multipageformats = set(('ps', 'pdf'))

bitmapformats = set(('png', 'bmp', 'jpg', 'tiff', 'xpm'))

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
        for types, descr in document.Export.getFormats():
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

        # other controls
        self.checkMultiPage.clicked.connect(self.updateFilename)
        self.buttonBrowse.clicked.connect(self.browseClicked)

        setdb = setting.settingdb

        eloc = setdb['dirname_export_location']

        # where to export file
        if eloc == 'doc':
            self.dirname = os.path.dirname(os.path.abspath(docfilename))
        elif eloc == 'cwd':
            self.dirname = cgetcwd()
        else: # 'prev'
            self.dirname = setdb.get('dirname_export', qt4.QDir.homePath())

        # set default filename
        ext = setdb.get('export_format', 'pdf')
        if not docfilename:
            docfilename = 'export'
        filename = os.path.join(
            self.dirname,
            os.path.splitext(os.path.basename(docfilename))[0] + '.' + ext)
        self.editFileName.setText(filename)

        self.formatselected = ext
        self.pageselected = setdb.get('export_page', 'single')

        self.checkOverwrite.setChecked(setdb.get('export_overwrite', False))
        self.checkMultiPage.setChecked(setdb.get('export_multipage', True))
        self.exportSVGTextAsText.setChecked(setdb['export_SVG_text_as_text'])
        self.exportAntialias.setChecked(setdb['export_antialias'])
        self.exportQuality.setValue(setdb['export_quality'])

        # validate and set DPIs
        dpis = ('75', '90', '100', '150', '200', '300')
        self.exportDPI.addItems(dpis)
        self.exportDPIPDF.addItems(dpis)
        self.exportDPI.setValidator(qt4.QIntValidator(10, 10000, self))
        self.exportDPI.setEditText(str(setdb['export_DPI']))
        self.exportDPIPDF.setValidator(qt4.QIntValidator(10, 10000, self))
        self.exportDPIPDF.setEditText(str(setdb['export_DPI_PDF']))

        # button to change bitmap background
        self.exportBackgroundButton.clicked.connect(
            self.slotExportBackgroundClicked)
        self.updateExportBackground(setdb['export_background'])

        # set correct format
        self.fmtradios[ext].click()

        # set page mode
        {
            'single': self.radioPageSingle,
            'all': self.radioPageAll,
            'range': self.radioPageRange,
        }[self.pageselected].click()

        # label showing success/failure
        self.labelStatus.clear()

        # fix height as widgets are hidden
        width = self.size().width()
        self.adjustSize()
        self.resize(width, self.size().height())

    def formatClicked(self, fmt):
        """If the format is changed."""
        setting.settingdb['export_format'] = fmt
        self.formatselected = fmt
        self.checkMultiPage.setEnabled(fmt in multipageformats)
        self.updateFilename()

        for c in (self.exportAntialias, self.exportDPI, self.labelDPI,
                  self.exportBackgroundButton, self.labelBackgroundButton):
            c.setVisible(fmt in bitmapformats)

        for c in (self.exportDPIPDF, self.labelDPIPDF,
                  self.exportColor, self.labelColor):
            c.setVisible(fmt in ('pdf', 'ps', 'eps'))

        for c in (self.exportQuality, self.labelQuality):
            c.setVisible(fmt == 'jpg')

        for c in (self.exportSVGTextAsText, self.labelSVGTextAsText):
            c.setVisible(fmt == 'svg')

    def pageClicked(self, page):
        """If page type is set."""
        setting.settingdb['export_page'] = page
        self.pageselected = page
        self.updateFilename()

        self.pageRangeMin.setEnabled(page=='range')
        self.pageRangeMax.setEnabled(page=='range')

    def browseClicked(self):
        """Browse for file."""

        setdb = setting.settingdb

        # File types we can export to in the form ([extensions], Name)
        fd = qt4.QFileDialog(self, _('Export page'))

        filename = self.editFileName.text()
        dirname = os.path.dirname(self.editFileName.text())
        fd.setDirectory(dirname if dirname else self.dirname)

        fd.setFileMode(qt4.QFileDialog.AnyFile)
        fd.setAcceptMode(qt4.QFileDialog.AcceptSave)
        fd.setOptions(qt4.QFileDialog.DontConfirmOverwrite)

        # Create a mapping between a format string and extensions
        filtertoext = {}
        # convert extensions to filter
        exttofilter = {}
        filters = []
        # a list of extensions which are allowed
        validextns = []
        formats = document.Export.getFormats()
        for extns, name in formats:
            extensions = " ".join(["*." + item for item in extns])
            # join eveything together to make a filter string
            filterstr = '%s (%s)' % (name, extensions)
            filtertoext[filterstr] = extns
            for e in extns:
                exttofilter[e] = filterstr
            filters.append(filterstr)
            validextns += extns
        fd.setNameFilters(filters)

        fd.selectNameFilter(exttofilter[setdb['export_format']])

        filename = self.editFileName.text()
        dirname = os.path.dirname(os.path.abspath(filename))
        if os.path.isdir(dirname):
            fd.selectFile(filename)

        if fd.exec_() == qt4.QDialog.Accepted:
            # convert filter to extension
            filterused = str(fd.selectedNameFilter())
            chosenext = filtertoext[filterused][0]

            filename = fd.selectedFiles()[0]
            fileext = os.path.splitext(filename)[1][1:]
            if fileext not in validextns or fileext != chosenext:
                filename += "." + chosenext
            self.editFileName.setText(filename)
            self.fmtradios[chosenext].click()

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

            if multifile and (PAGENUM not in left and PAGENAME not in left):
                left += PAGENUM
            elif not multifile:
                left = left.replace(PAGENUM, '')
                left = left.replace(PAGENAME, '')

            newfilename = left + '.' + self.formatselected
            self.editFileName.setText(newfilename)

    def updatePageMinMax(self):
        """Update widgets allowing user to set page range."""
        npages = self.document.getNumberPages()
        if npages == 0:
            return
        if self.pageRangeMax.value() == 0:
            self.pageRangeMax.setValue(npages)
        self.pageRangeMin.setMinimum(1)
        self.pageRangeMin.setMaximum(npages)
        self.pageRangeMax.setMinimum(1)
        self.pageRangeMax.setMaximum(npages)

    @qt4.pyqtSlot()
    def clearLabel(self):
        """Clear label.
        Defined as a slot to work around PyQt C++ object deleted bug. """
        self.labelStatus.clear()

    def showMessage(self, text):
        """Show a message in a label, clearing after a time."""
        self.labelStatus.setText(text)
        qt4.QTimer.singleShot(3000, self.clearLabel)

    def updateExportBackground(self, colorname):
        """Update color on export background."""
        pixmap = qt4.QPixmap(16, 16)
        col = self.document.evaluate.colors.get(colorname)
        pixmap.fill(col)

        # update button (storing color in button itself - what fun!)
        self.exportBackgroundButton.setIcon(qt4.QIcon(pixmap))
        self.exportBackgroundButton.iconcolor = colorname

    def slotExportBackgroundClicked(self):
        """Button clicked to change background."""
        qcolor = self.document.evaluate.colors.get(
            self.exportBackgroundButton.iconcolor)
        color = qt4.QColorDialog.getColor(
            qcolor,
            self,
            "Choose color",
            qt4.QColorDialog.ShowAlphaChannel )
        if color.isValid():
            self.updateExportBackground(utils.extendedColorFromQColor(color))

    def accept(self):
        """Do the export"""

        if self.document.getNumberPages() == 0:
            self.showMessage(_('Error: no pages in document'))
            return

        filename = self.editFileName.text()
        if (self.isMultiFile() and PAGENUM not in filename and
            PAGENAME not in filename):
            self.showMessage(_('Error: %s or %s not in filename') % (
                PAGENUM, PAGENAME))
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

        # update settings from controls
        setdb['export_overwrite'] = self.checkOverwrite.isChecked()
        setdb['export_antialias'] = self.exportAntialias.isChecked()
        setdb['export_quality'] = self.exportQuality.value()
        setdb['export_color'] = self.exportColor.currentIndex() == 0
        setdb['export_background'] = self.exportBackgroundButton.iconcolor
        setdb['export_SVG_text_as_text'] = self.exportSVGTextAsText.isChecked()

        # update dpi if possible
        # FIXME: requires some sort of visual notification of validator
        for cntrl, setn in ((self.exportDPI, 'export_DPI'),
                            (self.exportDPIPDF, 'export_DPI_PDF')):
            try:
                text = cntrl.currentText()
                valid = cntrl.validator().validate(text, 0)[0]
                if valid == qt4.QValidator.Acceptable:
                    setdb[setn] = int(text)
            except ValueError:
                pass

        export = document.Export(
            self.document,
            '',    # filename
            [0],   # page numbers
            bitmapdpi=setdb['export_DPI'],
            pdfdpi=setdb['export_DPI_PDF'],
            antialias=setdb['export_antialias'],
            color=setdb['export_color'],
            quality=setdb['export_quality'],
            backcolor=setdb['export_background'],
            svgtextastext=setdb['export_SVG_text_as_text'],
        )

        def _overwriteQuestion(filename):
            """Ask user whether file can be overwritten."""
            retn = qt4.QMessageBox.question(
                self,
                _("Overwrite file?"),
                _("The file %s already exists") % os.path.basename(filename),
                qt4.QMessageBox.Save | qt4.QMessageBox.Cancel,
                qt4.QMessageBox.Cancel)
            return retn == qt4.QMessageBox.Save

        # count exported pages (in list so can be modified in function)
        pagecount = [0]
        def _checkAndExport():
            """Check whether file exists and export if ok."""
            if os.path.exists(export.filename):
                if not setdb['export_overwrite']:
                    if not _overwriteQuestion(export.filename):
                        return

            # show busy cursor
            qt4.QApplication.setOverrideCursor(qt4.QCursor(qt4.Qt.WaitCursor))
            # delete file if already exists
            try:
                os.unlink(export.filename)
            except EnvironmentError:
                pass

            try:
                # actually do the export
                export.export()
                pagecount[0] += len(export.pagenumbers)
            except (RuntimeError, EnvironmentError) as e:
                # errors from the export
                if isinstance(e, EnvironmentError):
                    msg = cstrerror(e)
                else:
                    msg = cstr(e)
                qt4.QApplication.restoreOverrideCursor()
                qt4.QMessageBox.critical(
                    self, _("Error - Veusz"),
                    _("Error exporting to file '%s'\n\n%s") %
                    (export.filename, msg))
            else:
                qt4.QApplication.restoreOverrideCursor()

        if self.isMultiFile():
            # write pages to multiple files
            for page in pages:
                pagename = self.document.getPage(page).name
                export.pagenumbers = [page]
                export.filename = os.path.join(
                    os.path.dirname(filename),
                    os.path.basename(filename).replace(PAGENUM, str(page+1)).
                    replace(PAGENAME, pagename)
                )
                _checkAndExport()
        else:
            # write page/pages to single file
            export.pagenumbers = pages
            export.filename = filename
            _checkAndExport()

        dirname = os.path.dirname(filename)
        if dirname:
            setting.settingdb['dirname_export'] = dirname

        if pagecount[0] > 0:
            self.showMessage(_('Exported %i page(s)') % pagecount[0])
