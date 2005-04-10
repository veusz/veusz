#    Copyright (C) 2005 Jeremy S. Sanders
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

import traceback
import time
import smtplib
import email.MIMEText

import qt

import utils

class _ExceptionItem(qt.QListViewItem):
    def __init__(self, parent, exceptiontext):
        qt.QListViewItem.__init__(self, parent)
        self.setMultiLinesEnabled(True)

        self.exceptiontext = exceptiontext
        self.count = 1

    def text(self, column):
        '''Return the text in a column.'''
        if column == 0:
            return self.exceptiontext
        else:
            return str(self.count)
        
class _ExceptionDialog(qt.QDialog):
    '''A dialog to show exceptions, and from there, send in bugs.'''

    def __init__(self, parent):
        qt.QDialog.__init__(self, parent)
        self.setCaption('Exceptions - Veusz')

        spacing = self.fontMetrics().height() / 2
        self.layout = qt.QVBoxLayout(self, spacing)
        l = qt.QLabel('An exception occured in Veusz. This probably '
                      'indicates a bug. You can help fix bugs by sending '
                      'in a bug report to the developers.', self)
        l.setAlignment( l.alignment() | qt.Qt.WordBreak )
        self.layout.addWidget(l)

        el = self.exceptionlist = qt.QListView(self)
        el.addColumn('Exception')
        el.addColumn('Count')
        self.layout.addWidget(self.exceptionlist)
        self.items = []

        bhbox = qt.QHBox(self)
        bhbox.setSpacing(spacing)
        self.layout.addWidget(bhbox)
        
        sendbutton = qt.QPushButton("&Send report...", bhbox)
        sendbutton.setDefault( True )
        closebutton = qt.QPushButton("&Close", bhbox)
        self.connect( sendbutton, qt.SIGNAL('clicked()'),
                      self.slotSend )
        self.connect( closebutton, qt.SIGNAL('clicked()'),
                      self.slotClose )

    def sizeHint(self):
        """Returns recommended size of dialog."""
        return qt.QSize(600, 400)

    def slotClose(self):
        """Hides the dialog."""
        self.hide()

    def slotSend(self):
        """Sends a bug report."""

        d = _SendDialog(self,
                        self.exceptionlist.selectedItem().exceptiontext)
        d.exec_loop()

    def addException(self, exception):
        '''Show the exception in the dialog.'''

        # create backtrace text from exception
        backtrace = ''.join(traceback.format_exception(*exception)).strip()

        # see whether exception occured before
        found = None
        for i in self.items:
            if i.exceptiontext == backtrace:
                found = i
                break

        if found != None:
            # if so, increment the count
            found.count += 1
            found.repaint()
            found.setSelected(True)
        else:
            # else, add an item
            i = _ExceptionItem(self.exceptionlist, backtrace)
            i.setSelected(True)
            self.exceptionlist.insertItem(i)
            self.items.append(i)

_reportformat='''
Veusz version: %s
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

class _SendDialog(qt.QDialog):
    '''A dialog to allow the user to send the exception.'''

    def __init__(self, parent, text):
        '''Initialise the dialog.'''
        qt.QDialog.__init__(self, parent)
        self.setCaption('Send exception - Veusz')

        spacing = self.fontMetrics().height() / 2
        self.layout = qt.QVBoxLayout(self, spacing)

        l = qt.QLabel('Please give your email address:', self)
        self.layout.addWidget(l)
        self.emailedit = qt.QLineEdit('', self)
        self.layout.addWidget(self.emailedit)

        l = qt.QLabel('Please say what you were doing before'
                      ' the error:', self)
        self.layout.addWidget(l)
        self.doingedit = qt.QTextEdit(self)
        self.doingedit.setTextFormat(qt.Qt.PlainText)
        self.layout.addWidget(self.doingedit)
        
        l = qt.QLabel('Exception text:', self)
        self.layout.addWidget(l)

        # make a text edit with the text to send (read only)
        self.text = _reportformat % (
            utils.version(),
            time.strftime('%a, %d %b %Y %H:%M:%S +0000', time.gmtime()),
            text
            )
        te = self.textedit = qt.QTextEdit(self)
        te.setText(self.text)
        te.setTextFormat(qt.Qt.PlainText)
        te.setReadOnly(True)
        self.layout.addWidget(te)

        l = qt.QLabel('No personal information will be sent other than '
                      'that listed here. Your mailserver, however, '
                      'could add additional information.', self)
        l.setAlignment( l.alignment() | qt.Qt.WordBreak )
        self.layout.addWidget(l)

        bhbox = qt.QHBox(self)
        bhbox.setSpacing(spacing)
        self.layout.addWidget(bhbox)

        sendbutton = qt.QPushButton("&Send", bhbox)
        cancelbutton = qt.QPushButton("&Cancel", bhbox)
        self.connect( sendbutton, qt.SIGNAL('clicked()'),
                      self.slotSend )
        self.connect( cancelbutton, qt.SIGNAL('clicked()'),
                      self.reject )

    def sizeHint(self):
        """Returns recommended size of dialog."""
        return qt.QSize(600, 400)

    def slotSend(self):
        '''Actually send the email.'''

        # build up the text of the message
        text = _emailformat % (
            self.text,
            str(self.doingedit.text())
            )

        fromaddress = str(self.emailedit.text())

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
            mb = qt.QMessageBox("Veusz",
                                "Failed to send message",
                                qt.QMessageBox.Critical,
                                qt.QMessageBox.Ok | qt.QMessageBox.Default,
                                qt.QMessageBox.NoButton,
                                qt.QMessageBox.NoButton)
            mb.exec_loop()
            return

        self.accept()

_dialog = None
def showException(exception):
    '''Show the exception given in the exception dialog.'''

    global _dialog

    # create the dialog if it doesn't exist
    if _dialog == None:
        _dialog = _ExceptionDialog(qt.qApp.mainWidget())

    _dialog.show()
    
    # add the exception to the dialog
    _dialog.addException(exception)
