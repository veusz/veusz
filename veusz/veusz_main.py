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
import optparse

# trick to make sure veusz is on the path, if being run as a script
try:
    import veusz
except ImportError:
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) )
    import veusz

from veusz.compat import czip, cbytes
from veusz import qtall as qt4
from veusz import utils

copyr='''Veusz %s

Copyright (C) Jeremy Sanders 2003-2016 <jeremy@jeremysanders.net> and contributors
Licenced under the GNU General Public Licence (version 2 or greater)
'''

splashcopyr='''<b><font color="purple">Veusz %s<br></font></b>
Copyright (C) Jeremy Sanders 2003-2016 and contributors<br>
Licenced under the GPL (version 2 or greater)
'''

def handleIntSignal(signum, frame):
    '''Ask windows to close if Ctrl+C pressed.'''
    qt4.qApp.closeAllWindows()

def makeSplashLogo():
    '''Make a splash screen logo.'''

    splash = qt4.QSplashScreen()
    splash.setStyleSheet("background-color:white;")
    
    # draw logo on pixmap
    layout = qt4.QVBoxLayout(splash)
    pm = utils.getPixmap('logo.png')
    logo = qt4.QLabel()            
    logo.setPixmap(pm)
    logo.setAlignment(qt4.Qt.AlignCenter)
    layout.addWidget(logo)
    
    # add copyright text
    message = qt4.QLabel()
    message.setText(splashcopyr % utils.version())
    message.setAlignment(qt4.Qt.AlignCenter)
    # increase size of font
    font = message.font()
    font.setPointSize(font.pointSize()*1.5)
    message.setFont(font)
    layout.addWidget(message)
    h = qt4.QFontMetrics(font).height()
    layout.setContentsMargins(h,h,h,h)

    # Center the spash screen
    splash.setGeometry(5, 5, 100, 100)
    screen = qt4.QDesktopWidget().screenGeometry()
    splash.move((screen.width()-layout.sizeHint().width())/2, 
        (screen.height()-layout.sizeHint().height())/2)

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

def listen(args, quiet):
    '''For running with --listen option.'''
    from veusz.veusz_listen import openWindow
    openWindow(args, quiet=quiet)

def export(exports, args):
    '''A shortcut to load a set of files and export them.'''
    from veusz import document
    from veusz import utils
    for expfn, vsz in czip(exports, args[1:]):
        doc = document.Document()
        ci = document.CommandInterpreter(doc)
        ci.Load(vsz)
        ci.run('Export(%s)' % repr(expfn))

def convertArgsUnicode(args):
    '''Convert set of arguments to unicode.
    Arguments in argv use current file system encoding
    '''
    enc = sys.getfilesystemencoding()
    # bail out if not supported
    if enc is None:
        return args
    out = []
    for a in args:
        if isinstance(a, cbytes):
            out.append( a.decode(enc) )
        else:
            out.append(a)
    return out

class ImportThread(qt4.QThread):
    '''Do import of main code within another thread.
    Main application runs when this is done
    '''
    def run(self):
        from veusz import setting
        from veusz import widgets
        from veusz import dataimport

class VeuszApp(qt4.QApplication):
    """Event which can open mac files."""

    def __init__(self, args):
        qt4.QApplication.__init__(self, args)

        self.lastWindowClosed.connect(self.quit)

        # register a signal handler to catch ctrl+C
        signal.signal(signal.SIGINT, handleIntSignal)

        # parse command line options
        parser = optparse.OptionParser(
            usage='%prog [options] filename.vsz ...',
            version=copyr % utils.version())
        parser.add_option('--unsafe-mode', action='store_true',
                          help='disable safety checks when running documents'
                          ' or scripts')
        parser.add_option('--listen', action='store_true',
                          help='read and execute Veusz commands from stdin,'
                          ' replacing veusz_listen')
        parser.add_option('--quiet', action='store_true',
                          help='if in listening mode, do not open a window but'
                          ' execute commands quietly')
        parser.add_option('--export', action='append', metavar='FILE',
                          help='export the next document to this'
                          ' output image file, exiting when finished')
        parser.add_option('--embed-remote', action='store_true',
                          help=optparse.SUPPRESS_HELP)
        parser.add_option('--plugin', action='append', metavar='FILE',
                          help='load the plugin from the file given for '
                          'the session')
        parser.add_option('--translation', metavar='FILE',
                          help='load the translation .qm file given')
        options, args = parser.parse_args(self.arguments())

        # export files to make images
        if options.export and len(options.export) != len(args)-1:
            parser.error(
                'export option needs same number of documents and '
                'output files')

        # convert args to unicode from filesystem strings
        self.args = convertArgsUnicode(args)
        self.options = options

        self.openeventfiles = []
        self.startupdone = False
        self.splash = None

    def openMainWindow(self, args):
        """Open the main window with any loaded files."""
        from veusz.windows.mainwindow import MainWindow

        emptywins = []
        for w in self.topLevelWidgets():
            if isinstance(w, MainWindow) and w.document.isBlank():
                emptywins.append(w)

        if len(args) > 1:
            # load in filenames given
            for filename in args[1:]:
                if not emptywins:
                    MainWindow.CreateWindow(filename)
                else:
                    emptywins[0].openFile(filename)
        else:
            # create blank window
            MainWindow.CreateWindow()

    def checkOpen(self):
        """If startup complete, open any files."""
        if self.startupdone:
            self.openMainWindow([None] + self.openeventfiles)
            del self.openeventfiles[:]
        else:
            qt4.QTimer.singleShot(100, self.checkOpen)

    def event(self, event):
        """Handle events. This is the only way to get the FileOpen event."""
        if event.type() == qt4.QEvent.FileOpen:
            self.openeventfiles.append(event.file())
            # need to wait until startup has finished
            qt4.QTimer.singleShot(100, self.checkOpen)
            return True
        return qt4.QApplication.event(self, event)

    def startup(self):
        """Do startup."""

        if not (self.options.listen or self.options.export):
            # show the splash screen on normal start
            self.splash = makeSplashLogo()
            self.splash.show()

        # optionally load a translation
        if self.options.translation:
            trans = qt4.QTranslator()
            trans.load(self.options.translation)
            self.installTranslator(trans)

        self.thread = ImportThread()
        self.thread.finished.connect(self.slotStartApplication)
        self.thread.start()

    def slotStartApplication(self):
        """Start app, after modules imported."""

        options = self.options
        args = self.args

        from veusz.utils import vzdbus, vzsamp
        vzdbus.setup()
        vzsamp.setup()

        from veusz import document
        from veusz import setting

        # install exception hook after thread has finished
        sys.excepthook = excepthook

        # for people who want to run any old script
        setting.transient_settings['unsafe_mode'] = bool(
            options.unsafe_mode)

        # add directories to path
        if setting.settingdb['external_pythonpath']:
            sys.path += setting.settingdb['external_pythonpath'].split(':')

        # load any requested plugins
        if options.plugin:
            document.Document.loadPlugins(pluginlist=options.plugin)

        # different modes
        if options.listen:
            # listen to incoming commands
            listen(args, quiet=options.quiet)
        elif options.export:
            export(options.export, args)
            self.quit()
            sys.exit(0)
        else:
            # standard start main window
            self.openMainWindow(args)
            self.startupdone = True

        # clear splash when startup done
        if self.splash is not None:
            self.splash.finish(self.topLevelWidgets()[0])

def run():
    '''Run the main application.'''

    # nasty workaround for bug that causes non-modal windows not to
    # appear on mac see
    # https://github.com/jeremysanders/veusz/issues/39
    if sys.platform == 'darwin':
        import glob
        for f in glob.glob(os.environ['HOME'] + '/Library/Saved Application State/org.python.veusz.*/*'):
            os.unlink(f)

    # jump to the embedding client entry point if required
    if len(sys.argv) == 2 and sys.argv[1] == '--embed-remote':
        from veusz.embed_remote import runremote
        runremote()
        return

    # this function is spaghetti-like and has nasty code paths.
    # the idea is to postpone the imports until the splash screen
    # is shown

    app = VeuszApp(sys.argv)
    app.startup()
    app.exec_()

# if ran as a program
if __name__ == '__main__':
    #import cProfile
    #cProfile.run('run()', 'outprofile.dat')
    run()
