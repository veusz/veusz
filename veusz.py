#!/usr/bin/env python

# veusz.py
# Main veusz program file

#    Copyright (C) 2004 Jeremy S. Sanders
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

import sys
import os.path
import signal

import qt

# this allows modules relative to this one to be used,
# allowing this program to be run from python, or using this script
# please suggest a replacement for this
sys.path.insert( 0, os.path.dirname(__file__) )

import utils
import windows.mainwindow

copyr='''Veusz version %s
Copyright (C) Jeremy Sanders 2003-2005 <jeremy@jeremysanders.net>
Licenced under the GNU General Public Licence (version 2 or greater)
'''

def write_details():
    '''Write the copyright details.'''
    sys.stderr.write(copyr % utils.version())
    
def handleIntSignal(signum, frame):
    '''Ask windows to close if Ctrl+C pressed.'''
    qt.qApp.closeAllWindows()

def run():
    '''Run the main application.'''

    app = qt.QApplication(sys.argv)

    # process command line arguments
    cmdline = [str(app.argv()[i]) for i in range(1, app.argc())]

    if '--help' in cmdline:
        write_details()
        sys.stderr.write('\nUsage: \n veusz saved.vsz ...\n')
        sys.stderr.write('Optional arguments --help, --version\n')
        sys.exit(0)
    elif '--version' in cmdline:
        write_details()
        sys.exit(0)

    # register a signal handler to catch ctrl+c
    signal.signal(signal.SIGINT, handleIntSignal)

    # open the main window
    win = windows.mainwindow.MainWindow()
    win.show()
    app.connect(app, qt.SIGNAL("lastWindowClosed()"),
                app, qt.SLOT("quit()"))

    # load in filename given
    if len(cmdline) != 0:
        win.openFile(cmdline[0])

        # open up more windows if multiple files given
        for i in cmdline[1:]:
            win = windows.mainwindow.MainWindow()
            win.show()
            win.openFile(i)

    app.exec_loop()

# if ran as a program
if __name__ == '__main__':
    run()

