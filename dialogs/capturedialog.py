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

class CaptureDialog(qt4.QDialog):
    """Capture dialog."""

    def __init__(self, document, *args):
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'capture.ui'),
                   self)

        self.document = document

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
        self.slotMethodChanged(0)

        # get notification of change of stop method
        self.stopBG = qt4.QButtonGroup(self)
        self.stopBG.addButton( self.clickingStopButton, 0 )
        self.stopBG.addButton( self.numPtsStopButton, 1 )
        self.stopBG.addButton( self.timeStopButton, 2 )
        self.connect(self.stopBG, qt4.SIGNAL('buttonClicked(int)'),
                     self.slotStopChanged)
        self.slotStopChanged(0)

        # user starts capture
        self.connect(self.captureButton, qt4.SIGNAL('clicked()'),
                     self.slotCaptureClicked)
        # filename browse button clicked
        self.connect(self.browseButton, qt4.SIGNAL('clicked()'),
                     self.slotBrowseClicked)

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
                pass
        except Exception, e:
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

class CapturingDialog(qt4.QDialog):
    """In progress of capturing data dialog."""

    def __init__(self, document, simpleread, stream, *args):
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'capturing.ui'),
                   self)

        self.document = document
        self.simpleread = simpleread
        self.stream = stream

        # timer which governs reading from source
        self.readtimer = qt4.QTimer(self)
        self.connect( self.readtimer, qt4.SIGNAL('timeout()'),
                      self.slotReadTimer )

        # recored when time started
        self.starttime = qt4.QTime()
        self.starttime.start()

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
        for name in sorted(cts.keys()):
            length = str( cts[name] )
            find = tree.findItems(name, qt4.Qt.MatchExactly, 0)
            if find:
                find[0].setText(1, length)
            else:
                self.datasetTreeWidget.addTopLevelItem(
                    qt4.QTreeWidgetItem([name, length]) )
