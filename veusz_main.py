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

# Allow veusz to be run even if not installed into PYTHONPATH
try:
    import veusz
except ImportError:
    # load in the veusz module, but change its path to
    # the veusz directory, and insert it into sys.modules    
    import __init__ as veusz
    thisdir = os.path.dirname( os.path.abspath(__file__) )
    veusz.__path__ = [thisdir]
    veusz.__name__ = 'veusz'
    sys.modules['veusz'] = veusz

import veusz.qtall as qt4

import veusz.utils as utils
from veusz.windows.mainwindow import MainWindow
from veusz.application import Application
import veusz.widgets

copyr='''Veusz %s

Copyright (C) Jeremy Sanders 2003-2006 <jeremy@jeremysanders.net>
Licenced under the GNU General Public Licence (version 2 or greater)
'''

def handleIntSignal(signum, frame):
    '''Ask windows to close if Ctrl+C pressed.'''
    qt4.qApp.closeAllWindows()

def run():
    '''Run the main application.'''

    app = Application(sys.argv)

    # register a signal handler to catch ctrl+c
    signal.signal(signal.SIGINT, handleIntSignal)

    # handle arguments
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
                MainWindow.CreateWindow(filename)
        else:
            MainWindow.CreateWindow()
    else:
        MainWindow.CreateWindow()
    
    app.connect(app, qt4.SIGNAL("lastWindowClosed()"),
                app, qt4.SLOT("quit()"))

    app.exec_()

# if ran as a program
if __name__ == '__main__':
    #import profile
    #profile.run('run()')
    run()

