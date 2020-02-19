#    Copyright (C) 2009 Jeremy S. Sanders
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

"""Veusz data capture dialog."""

from __future__ import division
from ..compat import citems, cstr, cstrerror
from .. import qtall as qt
from .. import setting
from ..dataimport import capture, simpleread
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context="CaptureDialog"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class CaptureDialog(VeuszDialog):
    """Capture dialog.

    This allows the user to set the various capture options."""

    def __init__(self, document, mainwindow):
        VeuszDialog.__init__(self, mainwindow, 'capture.ui')
        self.document = document

        # set values of edit controls from previous invocation (if any)
        d = setting.settingdb

        # Validate edit controls
        validator = qt.QIntValidator(1, 65535, self)
        self.portEdit.setValidator(validator)
        validator = qt.QIntValidator(1, 1000000000, self)
        self.numLinesStopEdit.setValidator(validator)
        self.timeStopEdit.setValidator(validator)
        self.tailEdit.setValidator(validator)

        # floating point values for interval
        self.updateIntervalsEdit.setValidator(
            qt.QDoubleValidator(1e-2, 10000000, 2, self))

        # add completion for filenames
        c = self.filenamecompleter = qt.QCompleter(self)
        model = qt.QDirModel(c)
        c.setModel(model)
        self.filenameEdit.setCompleter(c)

        # get notification of change of capture method
        self.methodBG = qt.QButtonGroup(self)
        self.methodBG.addButton( self.captureFileButton, 0 )
        self.methodBG.addButton( self.captureInternetButton, 1 )
        self.methodBG.addButton( self.captureProgramButton, 2 )
        self.methodBG.buttonClicked[int].connect(self.slotMethodChanged)
        # restore previously clicked button
        self.methodBG.button( d.get('CaptureDialog_method', 0) ).click()

        # get notification of change of stop method
        self.stopBG = qt.QButtonGroup(self)
        self.stopBG.addButton( self.clickingStopButton, 0 )
        self.stopBG.addButton( self.numLinesStopButton, 1 )
        self.stopBG.addButton( self.timeStopButton, 2 )
        self.stopBG.buttonClicked[int].connect(self.slotStopChanged)
        self.stopBG.button( d.get('CaptureDialog_stop', 0) ).click()

        # update interval
        self.updateIntervalsCheck.toggled.connect(
            self.updateIntervalsEdit.setEnabled)

        # tail data
        self.tailCheck.toggled.connect(self.tailEdit.setEnabled)

        # user starts capture
        self.captureButton = self.buttonBox.addButton(
            _("Ca&pture"), qt.QDialogButtonBox.ApplyRole )

        self.captureButton.clicked.connect(self.slotCaptureClicked)

        # filename browse button clicked
        self.browseButton.clicked.connect(self.slotBrowseClicked)

    def done(self, r):
        """Dialog is closed."""
        VeuszDialog.done(self, r)

        # record values for next time dialog is opened
        d = setting.settingdb
        d['CaptureDialog_method'] = self.methodBG.checkedId()
        d['CaptureDialog_stop'] = self.stopBG.checkedId()

    def slotMethodChanged(self, buttonid):
        """Enable/disable correct controls in methodBG."""
        # enable correct buttons
        fc = buttonid==0
        self.filenameEdit.setEnabled(fc)
        self.browseButton.setEnabled(fc)

        ic = buttonid==1
        self.hostEdit.setEnabled(ic)
        self.portEdit.setEnabled(ic)

        xc = buttonid==2
        self.commandLineEdit.setEnabled(xc)

    def slotStopChanged(self, buttonid):
        """Enable/disable correct controls in stopBG."""

        ns = buttonid == 1
        self.numLinesStopEdit.setEnabled(ns)

        ts = buttonid == 2
        self.timeStopEdit.setEnabled(ts)

    def slotBrowseClicked(self):
        """Browse for a data file."""

        fd = qt.QFileDialog(self, 'Browse data file or socket')
        fd.setFileMode( qt.QFileDialog.ExistingFile )

        # update filename if changed
        if fd.exec_() == qt.QDialog.Accepted:
            self.filenameEdit.replaceAndAddHistory( fd.selectedFiles()[0] )

    def slotCaptureClicked(self):
        """User requested capture."""

        # object to interpret data from stream
        descriptor = self.descriptorEdit.text()
        simprd = simpleread.SimpleRead(descriptor)

        maxlines = None
        timeout = None
        updateinterval = None
        tail = None
        try:
            stop = self.stopBG.checkedId()
            if stop == 1:
                # number of lines to read before stopping
                maxlines = int( self.numLinesStopEdit.text() )
            elif stop == 2:
                # maximum time period before stopping
                timeout = int( self.timeStopEdit.text() )

            # whether to do an update periodically
            if self.updateIntervalsCheck.isChecked():
                updateinterval = float( self.updateIntervalsEdit.text() )

            # whether to only retain N values
            if self.tailCheck.isChecked():
                tail = int( self.tailEdit.text() )

        except ValueError:
            qt.QMessageBox.critical(self, _("Invalid number"), _("Invalid number"))
            return
            
        # get method of getting data
        method = self.methodBG.checkedId()
        try:
            # create stream
            if method == 0:
                # file/socket
                stream = capture.FileCaptureStream(self.filenameEdit.text())
            elif method == 1:
                # internet socket
                stream = capture.SocketCaptureStream(
                    self.hostEdit.text(),
                    int(self.portEdit.text()) )
            elif method == 2:
                # external program
                stream = capture.CommandCaptureStream(
                    self.commandLineEdit.text())
        except EnvironmentError as e:
            # problem opening stream
            qt.QMessageBox.critical(
                self, _("Cannot open input"),
                _("Cannot open input:\n %s (error %i)") % (
                    cstrerror(e), e.errno)
            )
            return

        stream.maxlines = maxlines
        stream.timeout = timeout
        simprd.tail = tail
        cd = CapturingDialog(self.document, simprd, stream, self,
                             updateinterval=updateinterval)
        self.mainwindow.showDialog(cd)

########################################################################

class CapturingDialog(VeuszDialog):
    """Capturing data dialog.
    Shows progress to user."""

    def __init__(self, document, simprd, stream, parent,
                 updateinterval = None):
        """Initialse capture dialog:
        document: document to send data to
        simprd: object to interpret data
        stream: capturestream to read data from
        parent: parent widget
        updateinterval: if set, interval of seconds to update data in doc
        """

        VeuszDialog.__init__(self, parent, 'capturing.ui')

        self.document = document
        self.simpleread = simprd
        self.stream = stream

        # connect buttons
        self.finishButton.clicked.connect(self.slotFinish)
        self.cancelButton.clicked.connect(self.slotCancel)

        # timer which governs reading from source
        self.readtimer = qt.QTimer(self)
        self.readtimer.timeout.connect(self.slotReadTimer)

        # record time capture started
        self.starttime = qt.QTime()
        self.starttime.start()

        # sort tree by dataset name
        self.datasetTreeWidget.sortItems(0, qt.Qt.AscendingOrder)

        # timer for updating display
        self.displaytimer = qt.QTimer(self)
        self.displaytimer.timeout.connect(self.slotDisplayTimer)
        self.sourceLabel.setText( self.sourceLabel.text() %
                                  stream.name )
        self.txt_statusLabel = self.statusLabel.text()
        self.slotDisplayTimer() # initialise label

        # timer to update document
        self.updatetimer = qt.QTimer(self)
        self.updateoperation = None
        if updateinterval:
            self.updatetimer.timeout.connect(self.slotUpdateTimer)
            self.updatetimer.start( int(updateinterval*1000) )

        # start display and read timers
        self.displaytimer.start(1000)
        self.readtimer.start(10)

    def slotReadTimer(self):
        """Time to read more data."""
        try:
            self.simpleread.readData(self.stream)
        except capture.CaptureFinishException as e:
            # stream tells us it's time to finish
            self.streamCaptureFinished( cstr(e) )

    def slotDisplayTimer(self):
        """Time to update information about data source."""
        self.statusLabel.setText( self.txt_statusLabel %
                                  (self.stream.bytesread,
                                   self.starttime.elapsed() // 1000) )

        tree = self.datasetTreeWidget
        cts = self.simpleread.getDatasetCounts()

        # iterate over each dataset
        for name, length in citems(cts):
            find = tree.findItems(name, qt.Qt.MatchExactly, 0)
            if find:
                # if already in tree, update number of counts
                find[0].setText(1, str(length))
            else:
                # add new item
                tree.addTopLevelItem( qt.QTreeWidgetItem([name, str(length)]))

    def slotUpdateTimer(self):
        """Called to update document while data is being captured."""

        # undo any previous update
        if self.updateoperation:
            self.updateoperation.undo(self.document)

        # create new one
        self.updateoperation = capture.OperationDataCaptureSet(
            self.simpleread)

        # apply it (bypass history here - urgh)
        self.updateoperation.do(self.document)
        self.document.setModified()

    def streamCaptureFinished(self, message):
        """Stop timers, close stream and display message
        about finished stream."""

        # stop reading / displaying
        self.readtimer.stop()
        self.displaytimer.stop()
        self.updatetimer.stop()
        if self.stream:
            # update stats
            self.slotDisplayTimer()
            # close stream
            self.stream.close()
            self.stream = None
        # show message from stream
        self.statusLabel.setText(message)

    def slotFinish(self):
        """Finish capturing and save the results."""

        # close down timers
        self.streamCaptureFinished('')

        # undo any in-progress update
        if self.updateoperation:
            self.updateoperation.undo(self.document)

        # apply real document operation update
        op = capture.OperationDataCaptureSet(self.simpleread)
        self.document.applyOperation(op)

        # close dialog
        self.close()

    def slotCancel(self):
        """Cancel capturing."""

        # close down timers
        self.streamCaptureFinished('')

        # undo any in-progress update
        if self.updateoperation:
            self.updateoperation.undo(self.document)
            self.document.setModified()

        # close dialog
        self.close()

 
