# mainwindow.py
# the main window of the application

#    Copyright (C) 2003 Jeremy S. Sanders
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

import qt
import sys

import consolewindow
import plotwindow
import document
import dialogs.aboutdialog

class MainWindow(qt.QMainWindow):
    """ The main window class for the application."""

    def __init__(self, *args):
        qt.QMainWindow.__init__(self, *args)

        self.document = document.Document()

        self.setCaption("Veusz")

        self._defineMenus()

        self.plot = plotwindow.PlotWindow(self.document, self)

        # make the console window a dock
        self.console = consolewindow.ConsoleWindow(self.document,
                                                   self)
        self.moveDockWindow( self.console, qt.Qt.DockBottom )

        # the plot window is the central window
        self.setCentralWidget( self.plot )

        self.statusBar().message("Ready", 2000)

    def _defineMenus(self):
        self.menuFile = qt.QPopupMenu( self )
        self.menuBar().insertItem( "&File",
                                   self.menuFile)

        self.actionQuit = qt.QAction("Quit the program",
                                     "&Quit", qt.QKeySequence(), self)
        qt.QObject.connect( self.actionQuit, qt.SIGNAL( "activated()" ),
                            qt.qApp, qt.SLOT( "closeAllWindows()" ))

        self.actionQuit.addTo( self.menuFile )

        self.menuHelp = qt.QPopupMenu( self )
        self.menuBar().insertItem( "&Help",
                                   self.menuHelp )

        self.actionAbout = qt.QAction("Displays information about the program",
                                      "About...", qt.QKeySequence(), self)
        qt.QObject.connect( self.actionAbout, qt.SIGNAL("activated()" ),
                            self.slotAbout )
        self.actionAbout.addTo( self.menuHelp )


    def slotAbout(self):
        """Show about dialog."""
        d = dialogs.aboutdialog.AboutDialog(self)
        d.exec_loop()

