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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################
 
# $Id$

'''Dialog to pop up if an exception occurs in Veusz.
This allows the user to send a bug report in via email.'''

import sys
import os
import os.path
import time
import traceback
import smtplib
import email.MIMEText

import numarray

import veusz.qtall as qt4
import veusz.utils as utils

_reportformat='''
Veusz version: %s
Python version: %s
Numarray version: %s
Date: %s

%s
'''

_emailformat = '''
Veusz exception report
----------------------

%s

-----------------------------------------
What the user was doing before the crash:
%s
'''

_to_address = 'veusz-exception-reports@gna.org'
#_to_address = 'jss@ast.cam.ac.uk'

class ExceptionSendDialog(qt4.QDialog):
    """Dialog to send debugging report."""
    
    def __init__(self, exception, *args):

        # load up UI
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'exceptionsend.ui'),
                   self)

        # debugging report text
        self.text = _reportformat % (
            utils.version(),
            sys.version,
            numarray.__version__,
            time.strftime('%a, %d %b %Y %H:%M:%S +0000', time.gmtime()),
            exception
            )
        self.detailstosend.setPlainText(self.text)
        
    def accept(self):
        """Send text."""
        # build up the text of the message
        text = _emailformat % (
            self.text,
            unicode(self.detailsedit.toPlainText())
            )

        fromaddress = unicode(self.emailedit.text())

        # construct the message object
        msg = email.MIMEText.MIMEText(text)
        msg['Subject'] = 'Veusz exception report'
        msg['From'] = fromaddress
        msg['To'] = _to_address
        msg['X-Veusz-Exception'] = 'Yes'

        # send the message
        try:
            s = smtplib.SMTP()
            s.connect()
            s.sendmail(fromaddress, [_to_address], msg.as_string())
            s.close()
        except smtplib.SMTPException:
            # something went wrong...
            mb = qt4.QMessageBox("Veusz",
                                 "Failed to send message",
                                 qt.QMessageBox.Critical,
                                 qt.QMessageBox.Ok | qt.QMessageBox.Default,
                                 qt.QMessageBox.NoButton,
                                 qt.QMessageBox.NoButton)
            mb.exec_loop()
            return

        qt4.QDialog.accept(self)

class ExceptionDialog(qt4.QDialog):
    """Choose an exception to send to developers."""
    
    exceptions = []

    def __init__(self, exception, *args):

        # load up UI
        qt4.QDialog.__init__(self, *args)
        qt4.loadUi(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'exceptionlist.ui'),
                   self)

        # create backtrace text from exception, and add to list
        backtrace = ''.join(traceback.format_exception(*exception)).strip()
        for i in ExceptionDialog.exceptions:
            if i[0] == backtrace:
                i[1] += 1
                break
        else:
            ExceptionDialog.exceptions.append( [backtrace, 1] )

        # fill in exception table
        e = self.exceptionlist
        e.clear()
        e.setHeaderLabels(['Problem', 'Count'])

        items = []
        thisrow = None
        for row, (exc, num) in enumerate(ExceptionDialog.exceptions):
            item = qt4.QTreeWidgetItem([exc, str(num)])
            if exc == backtrace:
                thisrow = item
            items.insert(0, item)

        e.addTopLevelItems(items)
        e.resizeColumnToContents(0)
        e.setCurrentItem(thisrow)
        
    def accept(self):
        """Accept by opening send dialog."""

        # get text for selected item
        selitems = self.exceptionlist.selectedItems()
        excepttext = unicode(selitems[0].text(0))

        d = ExceptionSendDialog(excepttext, self)
        if d.exec_() == qt4.QDialog.Accepted:
            qt4.QDialog.accept(self)
        
