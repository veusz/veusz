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

# $Id$

"""Veusz data capture dialog."""

import os.path

import veusz.qtall as qt4
import veusz.utils as utils
import veusz.document as document
import veusz.setting as setting

class CaptureDialog(qt4.QDialog):
    """Capture dialog.

    This allows the user to set the various capture options."""

    def __init__(self, document, *args):
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'capture.ui'),
                   self)

        self.document = document

        # set values of edit controls from previous invocation (if any)
        d = setting.settingdb
        self.descriptorEdit.setText( d.get('capture_descriptor', '') )
        self.filenameEdit.setText( d.get('capture_filename', '') )
        self.hostEdit.setText( d.get('capture_host', 'localhost') )
        self.portEdit.setText( d.get('capture_port', '10000') )
        self.commandLineEdit.setText( d.get('capture_commandline', '') )
        self.numPtsStopEdit.setText( d.get('capture_numptsstop', '1000') )
        self.timeStopEdit.setText( d.get('capture_timestop', '60') )

        # validate edit controls
        validator = qt4.QIntValidator(1, 65535, self)
        self.portEdit.setValidator(validator)
        validator = qt4.QIntValidator(1, 1000000000, self)
        self.numPtsStopEdit.setValidator(validator)
        self.timeStopEdit.setValidator(validator)

        # get notification of change of capture method
        self.methodBG = qt4.QButtonGroup(self)
        self.methodBG.addButton( self.captureFileButton, 0 )
        self.methodBG.addButton( self.captureInternetButton, 1 )
        self.methodBG.addButton( self.captureProgramButton, 2 )
        self.connect(self.methodBG, qt4.SIGNAL('buttonClicked(int)'),
                     self.slotMethodChanged)
        # restore previously clicked button
        self.methodBG.button( d.get('capture_method', 0) ).click()

        # get notification of change of stop method
        self.stopBG = qt4.QButtonGroup(self)
        self.stopBG.addButton( self.clickingStopButton, 0 )
        self.stopBG.addButton( self.numPtsStopButton, 1 )
        self.stopBG.addButton( self.timeStopButton, 2 )
        self.connect(self.stopBG, qt4.SIGNAL('buttonClicked(int)'),
                     self.slotStopChanged)
        self.stopBG.button( d.get('capture_stop', 0) ).click()

        # user starts capture
        self.connect(self.captureButton, qt4.SIGNAL('clicked()'),
                     self.slotCaptureClicked)
        # filename browse button clicked
        self.connect(self.browseButton, qt4.SIGNAL('clicked()'),
                     self.slotBrowseClicked)

    def done(self, r):
        """Dialog is closed."""
        qt4.QDialog.done(self, r)

        # record values for next time dialog is opened
        d = setting.settingdb
        d['capture_descriptor'] = unicode( self.descriptorEdit.text() )
        d['capture_filename'] = unicode( self.filenameEdit.text() )
        d['capture_host'] = unicode( self.hostEdit.text() )
        d['capture_port'] = unicode( self.portEdit.text() )
        d['capture_commandline'] = unicode( self.commandLineEdit.text() )
        d['capture_numptsstop'] = unicode( self.numPtsStopEdit.text() )
        d['capture_timestop'] = unicode( self.timeStopEdit.text() )
        d['capture_method'] = self.methodBG.checkedId()
        d['capture_stop'] = self.stopBG.checkedId()

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
        self.numPtsStopEdit.setEnabled(ns)

        ts = buttonid == 2
        self.timeStopEdit.setEnabled(ts)

    def slotBrowseClicked(self):
        """Browse for a data file."""

        fd = qt4.QFileDialog(self, 'Browse data file or socket')
        fd.setFileMode( qt4.QFileDialog.ExistingFile )

        # update filename if changed
        if fd.exec_() == qt4.QDialog.Accepted:
            self.filenameEdit.setText( fd.selectedFiles()[0] )

    def slotCaptureClicked(self):
        """User requested capture."""

        # object to interpret data from stream
        descriptor = unicode( self.descriptorEdit.text() )
        simpleread = document.SimpleRead(descriptor)

        method = self.methodBG.checkedId()
        try:
            # create stream
            if method == 0:
                # file/socket
                stream = document.FileCaptureStream(
                    unicode(self.filenameEdit.text()) )
            elif method == 1:
                # internet socket
                pass
            elif method == 2:
                # external program
                stream = document.CommandCaptureStream(
                    unicode(self.commandLineEdit.text()) )
        except EnvironmentError, e:
            # problem opening stream
            qt4.QMessageBox("Cannot open input",
                            "Cannot open input:\n"
                            " %s (error %i)" % (e.strerror, e.errno),
                            qt4.QMessageBox.Critical, qt4.QMessageBox.Ok,
                            qt4.QMessageBox.NoButton, qt4.QMessageBox.NoButton,
                            self).exec_()
            return

        cd = CapturingDialog(self.document, simpleread, stream, self)
        cd.show()

########################################################################

class CapturingDialog(qt4.QDialog):
    """Capturing data dialog.
    Shows progress to user."""

    def __init__(self, document, simpleread, stream, *args):
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'capturing.ui'),
                   self)

        self.document = document
        self.simpleread = simpleread
        self.stream = stream

        # connect buttons
        self.connect( self.finishButton, qt4.SIGNAL('clicked()'),
                      self.slotFinish )
        self.connect( self.cancelButton, qt4.SIGNAL('clicked()'),
                      self.slotCancel )

        # timer which governs reading from source
        self.readtimer = qt4.QTimer(self)
        self.connect( self.readtimer, qt4.SIGNAL('timeout()'),
                      self.slotReadTimer )

        # record time capture started
        self.starttime = qt4.QTime()
        self.starttime.start()

        # sort tree by dataset name
        self.datasetTreeWidget.sortItems(0, qt4.Qt.AscendingOrder)

        # timer for updating display
        self.displaytimer = qt4.QTimer(self)
        self.connect( self.displaytimer, qt4.SIGNAL('timeout()'),
                      self.slotDisplayTimer )
        self.sourceLabel.setText( unicode(self.sourceLabel.text()) %
                                  stream.name )
        self.txt_statusLabel = unicode(self.statusLabel.text())
        self.slotDisplayTimer() # initialise label

        # start timers
        self.displaytimer.start(1000)
        self.readtimer.start(10)

    def slotReadTimer(self):
        """Time to read more data."""
        self.simpleread.readData(self.stream)

    def slotDisplayTimer(self):
        """Time to update information about data source."""
        self.statusLabel.setText( self.txt_statusLabel %
                                  (self.stream.bytesread,
                                   self.starttime.elapsed() // 1000) )

        tree = self.datasetTreeWidget
        cts = self.simpleread.getDatasetCounts()

        # iterate over each dataset
        for name, length in cts.iteritems():
            find = tree.findItems(name, qt4.Qt.MatchExactly, 0)
            if find:
                # if already in tree, update number of counts
                find[0].setText(1, str(length))
            else:
                # add new item
                tree.addTopLevelItem( qt4.QTreeWidgetItem([name, str(length)]))

    def _finishUp(self):
        """Some cleanups."""
        # stop reading
        self.readtimer.stop()
        self.displaytimer.stop()
        # close the stream
        self.stream.close()
        # close the dialog
        self.close()

    def slotFinish(self):
        """Finish capturing and save the results."""
        self.simpleread.setInDocument(self.document)
        self._finishUp()

    def slotCancel(self):
        """Cancel capturing."""
        self._finishUp()
 
