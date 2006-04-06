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
import optparse

import qt

# this allows modules relative to this one to be used,
# allowing this program to be run from python, or using this script
# please suggest a replacement for this
sys.path.insert( 0, os.path.dirname(__file__) )

import utils
import windows.mainwindow
from application import Application

copyr='''Veusz %s

Copyright (C) Jeremy Sanders 2003-2006 <jeremy@jeremysanders.net>
Licenced under the GNU General Public Licence (version 2 or greater)
'''

def handleIntSignal(signum, frame):
    '''Ask windows to close if Ctrl+C pressed.'''
    qt.qApp.closeAllWindows()

def run():
    '''Run the main application.'''

    app = Application(sys.argv)

    # register a signal handler to catch ctrl+c
    signal.signal(signal.SIGINT, handleIntSignal)

    # call these functions on startup
    for function in utils.callonstartup:
        function()
    
    if app.argv():

        parser = optparse.OptionParser(
            usage="%prog [options] filename.vsz ...",
            version=copyr % utils.version())

        options, args = parser.parse_args( app.argv() )

        filelist = args[1:]
 
        # load in filename given
        if filelist:
            for filename in filelist:
                #XXX - need error handling here...
                windows.mainwindow.MainWindow.CreateWindow(filename)
        else:
            windows.mainwindow.MainWindow.CreateWindow()
    else:
        windows.mainwindow.MainWindow.CreateWindow()
    
    app.connect(app, qt.SIGNAL("lastWindowClosed()"),
                app, qt.SLOT("quit()"))

    app.exec_loop()

# if ran as a program
if __name__ == '__main__':
    #profile.run('run()')
    run()

