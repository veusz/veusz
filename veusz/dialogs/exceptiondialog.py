#    Copyright (C) 2006 Jeremy S. Sanders
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

'''Dialog to pop up if an exception occurs in Veusz.
This allows the user to send a bug report.'''

from __future__ import division, print_function
import sys
import time
import traceback
import re
import base64

import numpy
import sip

from ..compat import citems, curlrequest, cexceptionuser
from .. import qtall as qt
from .. import utils
from .veuszdialog import VeuszDialog

def _(text, disambiguation=None, context="ExceptionDialog"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

_emailUrl ='https://barmag.net/veusz-mail.php'

_reportformat = \
'''Veusz version: %s
Python version: %s
Python platform: %s
Numpy version: %s
Qt version: %s
PyQt version: %s
SIP version: %s
Date: %s

%s
'''

_sendformat = \
'''Email: %s

Error report
------------
%s

What the user was doing before the crash
----------------------------------------
%s
'''

def createReportText(exception):
    return _reportformat % (
        utils.version(),
        sys.version,
        sys.platform,
        numpy.__version__,
        qt.qVersion(),
        qt.PYQT_VERSION_STR,
        sip.SIP_VERSION_STR,
        time.strftime('%a, %d %b %Y %H:%M:%S +0000', time.gmtime()),
        cexceptionuser(exception),
    )

class ExceptionSendDialog(VeuszDialog):
    """Dialog to send debugging report."""

    def __init__(self, exception, parent):

        VeuszDialog.__init__(self, parent, 'exceptionsend.ui')

        # debugging report text
        self.text = createReportText(exception)
        self.detailstosend.setPlainText(self.text)

    def accept(self):
        """Send text."""
        # build up the text of the message
        text = (
            _sendformat % (
                self.emailedit.text(),
                self.text,
                self.detailsedit.toPlainText()
                ))

        # send the message as base-64 encoded utf-8
        text = base64.b64encode(text.encode('utf8'))

        try:
            # send the message
            curlrequest.urlopen(_emailUrl, b'message='+text)
        except:
            # something went wrong...
            qt.QMessageBox.critical(
                None, _("Veusz"),
                _("Failed to connect to error server "
                  "to send report. Is your internet connected?"))
            return

        qt.QMessageBox.information(
            self, _("Submitted"),
            _("Thank you for submitting an error report"))
        VeuszDialog.accept(self)

def _raiseIgnoreException():
    """Ignore this exception to clear out stack frame of previous exception."""
    raise utils.IgnoreException()

def formatLocals(exception):
    """Return local variables."""

    alreadyself = set()

    tb = exception[2]
    outlines = []
    while tb:
        frame = tb.tb_frame
        tb = tb.tb_next

        outlines.append('')
        outlines.append(
            'Frame %s (File %s, line %s)' %
            (frame.f_code.co_name,
             frame.f_code.co_filename,
             frame.f_lineno))

        # get local variables for frame
        for key, value in citems(frame.f_locals):
            # print out variables in frame
            try:
                v = repr(value)
            except:
                v = '<???>'
            if len(v) > 128:
                v = v[:120] + '...'
            outlines.append(' %s = %s' % (key, v))

            # print out attributes if item is self
            if key == 'self' and id(value) not in alreadyself:
                alreadyself.add(id(value))
                for attr in sorted( dir(value) ):
                    try:
                        v = getattr(value, attr)
                    except:
                        # can sometimes get type error
                        continue
                    if hasattr(v, '__call__'):
                        # skip callables, to cut down output
                        continue
                    try:
                        sv = repr(v)
                    except:
                        sv = '<???>'
                    if len(sv) > 128:
                        sv = sv[:120] + '...'

                    outlines.append('  self.%s = %s' % (attr, sv))

    return '\n'.join(outlines)

class ExceptionDialog(VeuszDialog):
    """Choose an exception to send to developers."""

    ignore_exceptions = set()

    def __init__(self, exception, parent):

        VeuszDialog.__init__(self, parent, 'exceptionlist.ui')

        # get text for traceback and locals
        self.fmtexcept = ''.join(traceback.format_exception(*exception))
        self.backtrace = self.fmtexcept + formatLocals(exception)

        self.errortextedit.setPlainText(self.backtrace)

        # set critical pixmap to left of dialog
        icon = qt.qApp.style().standardIcon(
            qt.QStyle.SP_MessageBoxCritical, None, self)
        self.erroriconlabel.setPixmap(icon.pixmap(32))

        self.ignoreSessionButton.clicked.connect(self.ignoreSessionSlot)
        self.saveButton.clicked.connect(self.saveButtonSlot)

        self.checkVeuszVersion()

        if not _emailUrl:
            self.okButton.hide()

    def checkVeuszVersion(self):
        """See whether there is a later version of veusz and inform the
        user."""

        newver = utils.latestVersion()
        thisver = utils.version()
        if not newver:
            msg = _('Could not check the latest Veusz version')
        else:
            # convert to tuples for comparison
            newver_tup = utils.versionToTuple(newver)
            thisver_tup = utils.versionToTuple(thisver)

            if thisver_tup == newver_tup:
                msg = _('You are running the latest released Veusz version')
            elif thisver_tup > newver_tup:
                msg = _('You are running an unreleased Veusz version')
            else:
                msg = (
                    _('<b>Your current version of Veusz is old. '
                      'Veusz %s is available.</b>') % newver)

        self.veuszversionlabel.setText(msg)

    def accept(self):
        """Accept by opening send dialog."""
        d = ExceptionSendDialog(self.backtrace, self)
        if d.exec_() == qt.QDialog.Accepted:
            VeuszDialog.accept(self)

    def ignoreSessionSlot(self):
        """Ignore exception for session."""
        ExceptionDialog.ignore_exceptions.add(self.fmtexcept)
        self.reject()

    def saveButtonSlot(self):
        filename = qt.QFileDialog.getSaveFileName(self, 'Save File')
        if filename[0]:
            f = open(filename[0], 'w')
            f.write(createReportText(self.backtrace))
            f.close()

            self.close()

    def exec_(self):
        """Exec dialog if exception is not ignored."""
        if self.fmtexcept not in ExceptionDialog.ignore_exceptions:
            VeuszDialog.exec_(self)

        # send another exception shortly - this clears out the current one
        # so the stack frame of the current exception is released
        qt.QTimer.singleShot(0, _raiseIgnoreException)
