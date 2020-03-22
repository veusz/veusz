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

from .. import qtall as qt
from .. import setting
from .. import utils
from .. import document
from ..compat import citems, cstrerror, cstr, cgetcwd
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context='ExportDialog'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

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
        doc.signalModified.connect(self.updatePagePages)
        self.updatePagePages()

        # change 'Save' button to 'Export'
        self.buttonBox.button(qt.QDialogButtonBox.Save).setText(_('Export'))

        # these are mappings between filetypes and radio buttons
        self.fmtradios = dict([(f, getattr(self, r)) for f, r in formatradio])
        self.radiofmts = dict([(getattr(self, r), f) for f, r in formatradio])

        # get allowed types (some formats are disabled if no helper)
        docfmts = set()
        for types, descr in document.AsyncExport.getFormats():
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
        self.radioPagePages.clicked.connect(lambda: self.pageClicked('pages'))

        # other controls
        self.checkMultiPage.clicked.connect(self.updateSingleMulti)
        self.buttonBrowse.clicked.connect(self.browseClicked)

        setdb = setting.settingdb

        eloc = setdb['dirname_export_location']

        # where to export file
        if eloc == 'doc':
            self.dirname = os.path.dirname(os.path.abspath(docfilename))
        elif eloc == 'cwd':
            self.dirname = cgetcwd()
        else: # 'prev'
            self.dirname = setdb.get('dirname_export', qt.QDir.homePath())

        # set default filename
        ext = setdb.get('export_format', 'pdf')
        if not docfilename:
            docfilename = 'export'
        self.docname = os.path.splitext(os.path.basename(docfilename))[0]

        self.formatselected = ext
        self.pageselected = setdb.get('export_page', 'single')

        self.checkMultiPage.setChecked(setdb.get('export_multipage', True))
        self.updateSingleMulti()

        self.checkOverwrite.setChecked(setdb.get('export_overwrite', False))
        self.exportSVGTextAsText.setChecked(setdb['export_SVG_text_as_text'])
        self.exportAntialias.setChecked(setdb['export_antialias'])
        self.exportQuality.setValue(setdb['export_quality'])

        # validate and set DPIs
        dpis = ('72', '75', '90', '96', '100', '150', '200', '300')
        for cntrl in self.exportDPI, self.exportDPISVG, self.exportDPIPDF:
            cntrl.addItems(dpis)
            cntrl.setValidator(qt.QIntValidator(10, 10000, self))
        self.exportDPI.setEditText(str(setdb['export_DPI']))
        self.exportDPISVG.setEditText(str(setdb['export_DPI_SVG']))
        self.exportDPIPDF.setEditText(str(setdb['export_DPI_PDF']))

        # button to change bitmap background
        self.exportBackgroundButton.clicked.connect(
            self.slotExportBackgroundClicked)
        self.updateExportBackground(setdb['export_background'])

        # set correct format
        self.fmtradios[ext].click()

        # regexp for comma separated ranges
        valre = qt.QRegExp(
            r'^[0-9]+(\s*-\s*[0-9]+)?(\s*,\s*[0-9]+(\s*-\s*[0-9]+)?)*$')
        valid = qt.QRegExpValidator(valre, self)
        self.editPagePages.setValidator(valid)

        # set page mode
        {
            'range': self.radioPageSingle, # compatibility
            'single': self.radioPageSingle,
            'all': self.radioPageAll,
            'pages': self.radioPagePages,
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

        for c in (self.exportAntialias, self.exportDPI, self.labelDPI,
                  self.exportBackgroundButton, self.labelBackgroundButton):
            c.setVisible(fmt in bitmapformats)

        for c in (self.exportDPIPDF, self.labelDPIPDF,
                  self.exportColor, self.labelColor):
            c.setVisible(fmt in ('pdf', 'ps', 'eps'))

        for c in (self.exportQuality, self.labelQuality):
            c.setVisible(fmt == 'jpg')

        for c in (self.exportSVGTextAsText, self.labelSVGTextAsText,
                  self.exportDPISVG, self.labelDPISVG):
            c.setVisible(fmt == 'svg')

        self.updateSingleMulti()
        filename = os.path.splitext(self.editFileName.text())[0] + '.' + fmt
        self.editFileName.setText(filename)

    def pageClicked(self, page):
        """If page type is set."""
        setting.settingdb['export_page'] = page
        self.pageselected = page
        self.updateSingleMulti()

        self.editPagePages.setEnabled(page=='pages')

    def browseClicked(self):
        """Browse for file."""

        setdb = setting.settingdb

        # File types we can export to in the form ([extensions], Name)
        fd = qt.QFileDialog(self, _('Export page'))

        filename = self.editFileName.text()
        dirname = os.path.dirname(self.editFileName.text())
        fd.setDirectory(dirname if dirname else self.dirname)

        fd.setFileMode(qt.QFileDialog.AnyFile)
        fd.setAcceptMode(qt.QFileDialog.AcceptSave)
        fd.setOptions(qt.QFileDialog.DontConfirmOverwrite)

        # Create a mapping between a format string and extensions
        filtertoext = {}
        # convert extensions to filter
        exttofilter = {}
        filters = []
        # a list of extensions which are allowed
        validextns = []
        formats = document.AsyncExport.getFormats()
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

        if fd.exec_() == qt.QDialog.Accepted:
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
        multifile = self.pageselected != 'single'
        if (self.formatselected in multipageformats and
            self.checkMultiPage.isChecked()):
            multifile = False
        return multifile

    def updateSingleMulti(self, _oldmulti=[None]):
        """Change filename according to selected single or multi button."""
        setting.settingdb['export_multipage'] = self.checkMultiPage.isChecked()
        multifile = self.isMultiFile()
        if multifile:
            templ = setting.settingdb['export_template_multi']
        else:
            templ = setting.settingdb['export_template_single']

        newfilename = os.path.join(
            self.dirname,
            templ.replace('%DOCNAME%', self.docname) + '.' + self.formatselected)

        # only change if multi format status has changed or is
        # uninitialised
        if multifile is not getattr(self, '_oldsinglemulti', None):
            self.editFileName.setText(newfilename)
            self._oldsinglemulti = multifile

    def updatePagePages(self):
        """Update widgets allowing user to set ranges of pages."""
        npages = self.document.getNumberPages()
        if npages == 0:
            return
        text = '%i-%i' % (1, npages)
        self.editPagePages.setText(text)

    @qt.pyqtSlot()
    def clearLabel(self):
        """Clear label.
        Defined as a slot to work around PyQt C++ object deleted bug. """
        self.labelStatus.clear()

    def showMessage(self, text):
        """Show a message in a label, clearing after a time."""
        self.labelStatus.setText(text)
        qt.QTimer.singleShot(3000, self.clearLabel)

    def updateExportBackground(self, colorname):
        """Update color on export background."""
        pixmap = qt.QPixmap(16, 16)
        col = self.document.evaluate.colors.get(colorname)
        pixmap.fill(col)

        # update button (storing color in button itself - what fun!)
        self.exportBackgroundButton.setIcon(qt.QIcon(pixmap))
        self.exportBackgroundButton.iconcolor = colorname

    def slotExportBackgroundClicked(self):
        """Button clicked to change background."""
        qcolor = self.document.evaluate.colors.get(
            self.exportBackgroundButton.iconcolor)
        color = qt.QColorDialog.getColor(
            qcolor,
            self,
            "Choose color",
            qt.QColorDialog.ShowAlphaChannel )
        if color.isValid():
            self.updateExportBackground(utils.extendedColorFromQColor(color))

    def getPagePages(self):
        """Get list of entered pages."""
        visible = set(self.document.getVisiblePages())

        txt = self.editPagePages.text()
        parts = txt.split(',')
        pages = []
        for p in parts:
            p = p.replace(' ', '')
            try:
                if p.find('-')>=0:
                    rng = p.split('-')
                    for pg in range(int(rng[0])-1, int(rng[1])):
                        if pg in visible:
                            pages.append(pg)
                else:
                    pg = int(p)-1
                    if pg in visible:
                        pages.append(pg)
            except ValueError:
                # convertsion error
                raise RuntimeError(_('Error: invalid list of pages'))
        # check in range
        for pg in pages:
            if pg<0 or pg>=self.document.getNumberPages():
                raise RuntimeError(_('Error: pages out of range'))
        return pages

    def accept(self):
        """Do the export"""

        if not self.document.getVisiblePages():
            self.showMessage(_('Error: no visible pages in document'))
            return

        filename = self.editFileName.text()
        if (self.isMultiFile() and
            '%PAGENAME%' not in filename and
            '%PAGE%' not in filename and
            '%PAGE00%' not in filename and
            '%PAGE000%' not in filename):
            self.showMessage(
                _('Error: page name or number must be in filename'))
            return

        # add extension to filename if missing
        if not os.path.splitext(filename)[1]:
            filename += '.' + self.formatselected

        if self.pageselected == 'single':
            pages = [self.mainwindow.plot.getPageNumber()]
        elif self.pageselected == 'all':
            pages = self.document.getVisiblePages()
        elif self.pageselected == 'pages':
            try:
                pages = self.getPagePages()
            except RuntimeError as e:
                self.showMessage(str(e))
                return

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
        for cntrl, setn in (
                (self.exportDPI, 'export_DPI'),
                (self.exportDPIPDF, 'export_DPI_PDF'),
                (self.exportDPISVG, 'export_DPI_SVG')):
            try:
                text = cntrl.currentText()
                valid = cntrl.validator().validate(text, 0)[0]
                if valid == qt.QValidator.Acceptable:
                    setdb[setn] = int(text)
            except ValueError:
                pass

        export = document.AsyncExport(
            self.document,
            bitmapdpi=setdb['export_DPI'],
            pdfdpi=setdb['export_DPI_PDF'],
            antialias=setdb['export_antialias'],
            color=setdb['export_color'],
            quality=setdb['export_quality'],
            backcolor=setdb['export_background'],
            svgtextastext=setdb['export_SVG_text_as_text'],
            svgdpi=setdb['export_DPI_SVG'],
        )

        def _overwriteQuestion(filename):
            """Ask user whether file can be overwritten."""
            retn = qt.QMessageBox.question(
                self,
                _("Overwrite file?"),
                _("The file %s already exists") % os.path.basename(filename),
                qt.QMessageBox.Save | qt.QMessageBox.Cancel,
                qt.QMessageBox.Cancel)
            return retn == qt.QMessageBox.Save

        # count exported pages (in list so can be modified in function)
        pagecount = [0]
        def _checkAndExport(fname, pages):
            """Check whether file exists and export if ok."""
            if os.path.exists(fname):
                if not setdb['export_overwrite']:
                    if not _overwriteQuestion(fname):
                        return

            # show busy cursor
            #qt.QApplication.setOverrideCursor(qt.QCursor(qt.Qt.WaitCursor))
            # delete file if already exists
            try:
                os.unlink(fname)
            except EnvironmentError:
                pass

            # actually do the export
            export.add(fname, pages)
            pagecount[0] += len(pages)

            ext = os.path.splitext(fname)[1]
            if ext:
                utils.feedback.exportcts[ext] += 1

        if self.isMultiFile() or len(pages)==1:
            # write pages to multiple files
            for page in pages:
                pagename = self.document.getPage(page).name

                pg = page+1
                fname = filename.replace('%PAGE%', str(pg))
                fname = fname.replace('%PAGE00%', '%02i' % pg)
                fname = fname.replace('%PAGE000%', '%03i' % pg)
                fname = fname.replace('%PAGENAME%', pagename)
                _checkAndExport(fname, [page])
        else:
            # write page/pages to single file
            fname = filename.replace('%PAGE%', _('none'))
            fname = fname.replace('%PAGE00%', _('none'))
            fname = fname.replace('%PAGE000%', _('none'))
            fname = fname.replace('%PAGENAME%', _('none'))

            _checkAndExport(fname, pages)

        dirname = os.path.dirname(filename)
        if dirname:
            setting.settingdb['dirname_export'] = dirname

        # format feedback
        self.showMessage(_('Processing...'))

        self.buttonBox.button(qt.QDialogButtonBox.Close).setEnabled(False)
        self.buttonBox.button(qt.QDialogButtonBox.Save).setEnabled(False)

        def checkDone():
            """Check whether exporting has finished."""
            if not export.haveDone():
                return
            try:
                export.finish()
            except (RuntimeError, EnvironmentError) as e:
                # errors from the export
                if isinstance(e, EnvironmentError):
                    msg = cstrerror(e)
                else:
                    msg = cstr(e)
                #qt.QApplication.restoreOverrideCursor()
                qt.QMessageBox.critical(
                    self, _("Error - Veusz"),
                    _("Error exporting to file '%s'\n\n%s") %
                    (fname, msg))
            else:
                if pagecount[0] > 0:
                    self.showMessage(_('Exported %i page(s)') % pagecount[0])

            self.buttonBox.button(qt.QDialogButtonBox.Close).setEnabled(True)
            self.buttonBox.button(qt.QDialogButtonBox.Save).setEnabled(True)
            self.checktimer.stop()

        self.checktimer = qt.QTimer(self)
        self.checktimer.timeout.connect(checkDone)
        self.checktimer.start(20)
