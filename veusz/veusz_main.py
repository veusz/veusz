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

import sys
import signal
import argparse
import re

import veusz
from veusz import qtall as qt
from veusz import utils

if sys.version_info[0] < 3:
    raise RuntimeError('Veusz only supports Python 3')

copyr='''Veusz %s

Copyright (C) Jeremy Sanders 2003-2025 <jeremy@jeremysanders.net>
 and contributors
Licenced under the GNU General Public Licence (version 2 or greater)
'''

splashcopyr='''<b><font color="purple">Veusz %s<br></font></b>
Copyright (C) Jeremy Sanders 2003-2025 and contributors<br>
Licenced under the GPL (version 2 or greater)
'''

def _(text, disambiguation=None, context='Application'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

def handleIntSignal(signum, frame):
    '''Ask windows to close if Ctrl+C pressed.'''
    qt.QApplication.instance().closeAllWindows()

def makeSplash(app):
    '''Make a splash screen logo.'''

    splash = qt.QSplashScreen()
    splash.setStyleSheet("background-color:white; color: black;")

    # draw logo on pixmap
    layout = qt.QVBoxLayout(splash)
    logo = qt.QLabel()
    logo.setPixmap(utils.getPixmap('logo.png'))
    logo.setAlignment(qt.Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(logo)

    # add copyright text
    message = qt.QLabel()
    message.setText(splashcopyr % utils.version())
    message.setAlignment(qt.Qt.AlignmentFlag.AlignCenter)
    # increase size of font
    font = message.font()
    font.setPointSizeF(font.pointSize()*1.5)
    message.setFont(font)
    layout.addWidget(message)
    h = qt.QFontMetrics(font).height()
    layout.setContentsMargins(h,h,h,h)

    # Center the spash screen
    screen = splash.screen().size()
    splash.move(
        (screen.width()-layout.sizeHint().width())//2,
        (screen.height()-layout.sizeHint().height())//2
    )

    # make sure dialog goes away - avoid problem if a message box pops
    # up before it is removed
    qt.QTimer.singleShot(2000, splash.hide)

    return splash

def excepthook(excepttype, exceptvalue, tracebackobj):
    '''Show exception dialog if an exception occurs.'''

    # exception dialog doesnt work if not in main thread, so we send
    # the exception to the application to display
    app = qt.QGuiApplication.instance()
    if app.thread is not qt.QThread.currentThread():
        app.signalException.emit(excepttype, exceptvalue, tracebackobj)
        return

    sys.setrecursionlimit(sys.getrecursionlimit()+1000)

    from veusz.dialogs.exceptiondialog import ExceptionDialog
    if not isinstance(exceptvalue, utils.IgnoreException):
        # next exception is ignored to clear out the stack frame of the
        # previous exception - yuck
        d = ExceptionDialog((excepttype, exceptvalue, tracebackobj), None)
        d.exec()

def listen(docs, quiet):
    '''For running with --listen option.'''
    from veusz.veusz_listen import openWindow
    openWindow(docs, quiet=quiet)

def export(exports, docs, options):
    '''A shortcut to load a set of files and export them.'''
    from veusz import document
    from veusz import utils

    # TODO: validate options
    opttxt = ', '.join(options) if options else ''

    for expfn, vsz in zip(exports, docs):
        doc = document.Document()
        ci = document.CommandInterpreter(doc)
        ci.Load(vsz)
        ci.run('Export(%s, %s)' % (repr(expfn), opttxt))

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
        if isinstance(a, bytes):
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

    signalException = qt.pyqtSignal(object, object, object)

    def __init__(self):
        qt.QApplication.__init__(self, sys.argv)

        self.lastWindowClosed.connect(self.quit)
        self.signalException.connect(self.showException)

        # Bind desktop file to display icon in wayland
        qt.QGuiApplication.setDesktopFileName("veusz")

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
            '--export-option', action='append', metavar='VAL',
            help='add option when exporting file')
        parser.add_argument(
            '--embed-remote',
            action='store_true',
            help='(internal - not for external use)')
        parser.add_argument(
            '--veusz-plugin', action='append', metavar='FILE',
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
        from veusz.document import Document, PluginLoadError

        emptywins = []
        for w in self.topLevelWidgets():
            if isinstance(w, MainWindow) and w.document.isBlank():
                emptywins.append(w)

        if docs:
            # load in filenames given
            for filename in docs:
                if not emptywins:
                    MainWindow.CreateWindow(filename)
                elif filename:
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
        if event.type() == qt.QEvent.Type.FileOpen:
            self.openeventfiles.append(event.file())
            # need to wait until startup has finished
            qt.QTimer.singleShot(100, self.openPendingFiles)
            return True
        return qt.QApplication.event(self, event)

    def startup(self):
        """Do startup."""

        if not (self.args.listen or self.args.export):
            # show the splash screen on normal start
            self.splash = makeSplash(self)
            # this seems necessary on MacOS
            self.splash.resize(self.splash.sizeHint())
            self.splash.show()
            self.splash.raise_()

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
        global defaultexcepthook
        defaultexcepthook = sys.excepthook
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
            # We want a list of items separated by colons
            # Unfortunately on windows there can be a colon and drive letter,
            # so we avoid splitting colons which look like a:\foo or B:/bar
            parts = re.findall(
                r'[A-Za-z]:[\\/][^:]+|[^:]+',
                setting.settingdb['external_pythonpath'])
            sys.path += list(parts)

        try:
            # load plugins from settings
            document.Document.loadPlugins()
            if args.veusz_plugin:
                # load plugins on command line
                document.Document.loadPlugins(pluginlist=args.veusz_plugin)
        except document.PluginLoadError as e:
            startuperrors.append(str(e))

        # color theme
        scheme = setting.settingdb['color_scheme']
        hascolorscheme = hasattr(self.styleHints(), 'setColorScheme') # qt>=6.5
        if scheme == 'default':
            pass
        elif scheme == 'system-light' and hascolorscheme:
            self.styleHints().setColorScheme(qt.Qt.ColorScheme.Light)
        elif scheme == 'system-dark' and hascolorscheme:
            self.styleHints().setColorScheme(qt.Qt.ColorScheme.Dark)
        else:
            pal = utils.getPalette(scheme)
            if pal is not None:
                # palettes only work typically in fusion style
                self.setStyle("fusion")
                self.setPalette(pal)

        # different modes
        if args.listen:
            # listen to incoming commands
            listen(args.docs, quiet=args.quiet)
        elif args.export:
            export(args.export, args.docs, args.export_option)
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
            qt.QMessageBox.critical(None, _("Error starting - Veusz"), error)

    def showException(self, excepttype, exceptvalue, tracebackobj):
        """Show an exception dialog (raised from another thread)."""
        from veusz.dialogs.exceptiondialog import ExceptionDialog
        if not isinstance(exceptvalue, utils.IgnoreException):
            # next exception is ignored to clear out the stack frame of the
            # previous exception - yuck
            d = ExceptionDialog((excepttype, exceptvalue, tracebackobj), None)
            d.exec()

def run():
    '''Run the main application.'''

    # high DPI support
    try:
        qt.QApplication.setHighDpiScaleFactorRoundingPolicy(
            qt.QApplication.highDpiScaleFactorRoundingPolicy().PassThrough)
    except AttributeError:
        # old qt versions
        pass

    # jump to the embedding client entry point if required
    if len(sys.argv) == 2 and sys.argv[1] == '--embed-remote':
        from veusz.embed_remote import runremote
        runremote()
        return

    # start me up
    app = VeuszApp()
    app.startup()
    app.exec()

# if ran as a program
if __name__ == '__main__':
    #import cProfile
    #cProfile.run('run()', 'outprofile.dat')
    run()
