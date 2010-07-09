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
 
# $Id$

'''Dialog to pop up if an exception occurs in Veusz.
This allows the user to send a bug report in via email.'''

import sys
import os.path
import time
import traceback
import urllib2
import sip

import numpy

import veusz.qtall as qt4
import veusz.utils as utils

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

class ExceptionSendDialog(qt4.QDialog):
    """Dialog to send debugging report."""
    
    def __init__(self, exception, *args):

        # load up UI
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'exceptionsend.ui'),
                   self)

        # debugging report text
        self.text = _reportformat % (
            utils.version(),
            sys.version,
            sys.platform,
            numpy.__version__,
            qt4.qVersion(),
            qt4.PYQT_VERSION_STR,
            sip.SIP_VERSION_STR,
            time.strftime('%a, %d %b %Y %H:%M:%S +0000', time.gmtime()),
            exception
            )
        self.detailstosend.setPlainText(self.text)
        
    def accept(self):
        """Send text."""
        # build up the text of the message
        text = ( _sendformat % (
                unicode(self.emailedit.text()),
                self.text,
                unicode(self.detailsedit.toPlainText())
                ))

        # send the message as base-64 encoded utf-8
        text = str( qt4.QString(text).toUtf8().toBase64() )

        try:
            # send the message
            urllib2.urlopen('http://barmag.net/veusz-mail.php',
                            'message=%s' % text)

        except urllib2.URLError:
            # something went wrong...
            qt4.QMessageBox.critical(None, "Veusz",
                                     "Failed to connect to error server "
                                     "to send report. Is your internet "
                                     "connected?")
            return

        qt4.QDialog.accept(self)

class ExceptionDialog(qt4.QDialog):
    """Choose an exception to send to developers."""
    
    ignore_exceptions = set()

    def __init__(self, exception, *args):

        # load up UI
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(utils.veuszDirectory, 'dialogs',
                                'exceptionlist.ui'),
                   self)

        # create backtrace text from exception, and add to list
        self.backtrace = ''.join(traceback.format_exception(*exception)).strip()
        self.errortextedit.setPlainText(self.backtrace)

        # set critical pixmap to left of dialog
        icon = qt4.qApp.style().standardIcon(qt4.QStyle.SP_MessageBoxCritical,
                                             None, self)
        self.erroriconlabel.setPixmap(icon.pixmap(32))

        self.connect(self.ignoreSessionButton, qt4.SIGNAL('clicked()'),
                     self.ignoreSessionSlot)
        
    def accept(self):
        """Accept by opening send dialog."""
        d = ExceptionSendDialog(self.backtrace, self)
        if d.exec_() == qt4.QDialog.Accepted:
            qt4.QDialog.accept(self)
        
    def ignoreSessionSlot(self):
        """Ignore exception for session."""
        ExceptionDialog.ignore_exceptions.add(self.backtrace)
        self.reject()

    def exec_(self):
        """Exec dialog if exception is not ignored."""
        if self.backtrace not in ExceptionDialog.ignore_exceptions:
            qt4.QDialog.exec_(self)

