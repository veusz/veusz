#!/usr/bin/env python

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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

'''Main Veusz executable.'''

from __future__ import division
import sys
import os.path
import signal
import argparse

import veusz
from veusz.compat import czip, cbytes, cstr
from veusz import qtall as qt
from veusz import utils

copyr='''Veusz %s

Copyright (C) Jeremy Sanders 2003-2017 <jeremy@jeremysanders.net>
 and contributors
Licenced under the GNU General Public Licence (version 2 or greater)
'''

splashcopyr='''<b><font color="purple">Veusz %s<br></font></b>
Copyright (C) Jeremy Sanders 2003-2017 and contributors<br>
Licenced under the GPL (version 2 or greater)
'''

def handleIntSignal(signum, frame):
    '''Ask windows to close if Ctrl+C pressed.'''
    qt.qApp.closeAllWindows()

def makeSplashLogo():
    '''Make a splash screen logo.'''

    splash = qt.QSplashScreen()
    splash.setStyleSheet("background-color:white;")

    # draw logo on pixmap
    layout = qt.QVBoxLayout(splash)
    pm = utils.getPixmap('logo.png')
    logo = qt.QLabel()
    logo.setPixmap(pm)
    logo.setAlignment(qt.Qt.AlignCenter)
    layout.addWidget(logo)

    # add copyright text
    message = qt.QLabel()
    message.setText(splashcopyr % utils.version())
    message.setAlignment(qt.Qt.AlignCenter)
    # increase size of font
    font = message.font()
    font.setPointSize(font.pointSize()*1.5)
    message.setFont(font)
    layout.addWidget(message)
    h = qt.QFontMetrics(font).height()
    layout.setContentsMargins(h,h,h,h)

    # Center the spash screen
    splash.setGeometry(5, 5, 100, 100)
    screen = qt.QDesktopWidget().screenGeometry()
    splash.move((screen.width()-layout.sizeHint().width())/2,
        (screen.height()-layout.sizeHint().height())/2)

    # make sure dialog goes away - avoid problem if a message box pops
    # up before it is removed
    qt.QTimer.singleShot(2000, splash.hide)

    return splash

def excepthook(excepttype, exceptvalue, tracebackobj):
    '''Show exception dialog if an exception occurs.'''
    sys.setrecursionlimit(sys.getrecursionlimit()+1000)

    from veusz.dialogs.exceptiondialog import ExceptionDialog
    if not isinstance(exceptvalue, utils.IgnoreException):
        # next exception is ignored to clear out the stack frame of the
        # previous exception - yuck
        d = ExceptionDialog((excepttype, exceptvalue, tracebackobj), None)
        d.exec_()

def listen(docs, quiet):
    '''For running with --listen option.'''
    from veusz.veusz_listen import openWindow
    openWindow(docs, quiet=quiet)

def export(exports, docs):
    '''A shortcut to load a set of files and export them.'''
    from veusz import document
    from veusz import utils
    for expfn, vsz in czip(exports, docs):
        doc = document.Document()
        ci = document.CommandInterpreter(doc)
        ci.Load(vsz)
        ci.run('Export(%s)' % repr(expfn))

def convertArgsUnicode(args):
    '''Convert set of arguments to unicode (for Python 2).
    Arguments in argv use current file system encoding
    '''
    enc = sys.getfilesystemencoding()
    # bail out if not supported
    if enc is None:
        return args
    out = []
    for a in args:
        if isinstance(a, cbytes):
            out.append(a.decode(enc))
        else:
            out.append(a)
    return out

class ImportThread(qt.QThread):
    '''Do import of main code within another thread.
    Main application runs when this is done
    '''
    def run(self):
        from veusz import setting
        from veusz import widgets
        from veusz import dataimport

class VeuszApp(qt.QApplication):
    """Event which can open mac files."""

    def __init__(self):
        qt.QApplication.__init__(self, sys.argv)

        self.lastWindowClosed.connect(self.quit)

        # register a signal handler to catch ctrl+C
        signal.signal(signal.SIGINT, handleIntSignal)

        # parse command line options
        parser = argparse.ArgumentParser(
            description='Veusz scientific plotting package')
        parser.add_argument(
            '--version', action='version',
            version=copyr % utils.version())
        parser.add_argument(
            '--unsafe-mode',
            action='store_true',
            help='disable safety checks when running documents'
            ' or scripts')
        parser.add_argument(
            '--listen',
            action='store_true',
            help='read and execute Veusz commands from stdin,'
            ' replacing veusz_listen')
        parser.add_argument(
            '--quiet',
            action='store_true',
            help='if in listening mode, do not open a window but'
            ' execute commands quietly')
        parser.add_argument(
            '--export', action='append', metavar='FILE',
            help='export the next document to this'
            ' output image file, exiting when finished')
        parser.add_argument(
            '--embed-remote',
            action='store_true',
            help='(internal - not for external use)')
        parser.add_argument(
            '--plugin', action='append', metavar='FILE',
            help='load the plugin from the file given for '
            'the session')
        parser.add_argument(
            '--translation', metavar='FILE',
            help='load the translation .qm file given')
        parser.add_argument(
            'docs', metavar='FILE', nargs='*',
            help='document to load')

        self.args = args = parser.parse_args()

        args.docs = convertArgsUnicode(args.docs)

        # export files to make images
        if args.export:
            if len(args.export) != len(args.docs):
                parser.error(
                    'export option needs same number of documents and '
                    'output files')
            args.export = convertArgsUnicode(args.export)

        self.openeventfiles = []
        self.startupdone = False
        self.splash = None
        self.trans = None

    def openMainWindow(self, docs):
        """Open the main window with any loaded files."""
        from veusz.windows.mainwindow import MainWindow

        emptywins = []
        for w in self.topLevelWidgets():
            if isinstance(w, MainWindow) and w.document.isBlank():
                emptywins.append(w)

        if docs:
            # load in filenames given
            for filename in docs:
                if not emptywins:
                    MainWindow.CreateWindow(filename)
                else:
                    emptywins[0].openFile(filename)
        else:
            # create blank window
            MainWindow.CreateWindow()

    def openPendingFiles(self):
        """If startup complete, open any files."""
        if self.startupdone:
            self.openMainWindow([None] + self.openeventfiles)
            del self.openeventfiles[:]
        else:
            qt.QTimer.singleShot(100, self.openPendingFiles)

    def event(self, event):
        """Handle events. This is the only way to get the FileOpen event.
        FileOpen is used by MacOS to open files.
        """
        if event.type() == qt.QEvent.FileOpen:
            self.openeventfiles.append(event.file())
            # need to wait until startup has finished
            qt.QTimer.singleShot(100, self.openPendingFiles)
            return True
        return qt.QApplication.event(self, event)

    def startup(self):
        """Do startup."""

        if not (self.args.listen or self.args.export):
            # show the splash screen on normal start
            self.splash = makeSplashLogo()
            self.splash.show()

        self.thread = ImportThread()
        self.thread.finished.connect(self.slotStartApplication)
        self.thread.start()

    def slotStartApplication(self):
        """Start app, after modules imported."""

        args = self.args

        from veusz.utils import vzdbus, vzsamp
        vzdbus.setup()
        vzsamp.setup()

        # add text if we want to display an error after startup
        startuperrors = []

        from veusz import document
        from veusz import setting

        # install exception hook after thread has finished
        sys.excepthook = excepthook

        # for people who want to run any old script
        setting.transient_settings['unsafe_mode'] = bool(
            args.unsafe_mode)

        # optionally load a translation
        txfile = args.translation or setting.settingdb['translation_file']
        if txfile:
            self.trans = qt.QTranslator()
            if self.trans.load(txfile):
                self.installTranslator(self.trans)
            else:
                startuperrors.append(
                    'Error loading translation "%s"' % txfile)

        # add directories to path
        if setting.settingdb['external_pythonpath']:
            sys.path += setting.settingdb['external_pythonpath'].split(':')

        # load any requested plugins
        if args.plugin:
            try:
                document.Document.loadPlugins(pluginlist=args.plugin)
            except RuntimeError as e:
                startuperrors.append(cstr(e))

        # different modes
        if args.listen:
            # listen to incoming commands
            listen(args.docs, quiet=args.quiet)
        elif args.export:
            export(args.export, args.docs)
            self.quit()
            sys.exit(0)
        else:
            # standard start main window
            self.openMainWindow(args.docs)
            self.startupdone = True

        # clear splash when startup done
        if self.splash is not None:
            self.splash.finish(self.topLevelWidgets()[0])

        # this has to be displayed after the main window is created,
        # otherwise it never gets shown
        for error in startuperrors:
            qt.QMessageBox.critical(None, "Veusz", error)

def run():
    '''Run the main application.'''

    # jump to the embedding client entry point if required
    if len(sys.argv) == 2 and sys.argv[1] == '--embed-remote':
        from veusz.embed_remote import runremote
        runremote()
        return

    # this function is spaghetti-like and has nasty code paths.
    # the idea is to postpone the imports until the splash screen
    # is shown

    app = VeuszApp()
    app.startup()
    app.exec_()

# if ran as a program
if __name__ == '__main__':
    #import cProfile
    #cProfile.run('run()', 'outprofile.dat')
    run()
