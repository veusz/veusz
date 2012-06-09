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

copyr='''Veusz %s

Copyright (C) Jeremy Sanders 2003-2012 <jeremy@jeremysanders.net> and contributors
Licenced under the GNU General Public Licence (version 2 or greater)
'''

splashcopyr='''<b><font color="purple">Veusz %s<br></font></b>
Copyright (C) Jeremy Sanders 2003-2012 and contributors<br>
Licenced under the GPL (version 2 or greater)
'''

def handleIntSignal(signum, frame):
    '''Ask windows to close if Ctrl+C pressed.'''
    qt4.qApp.closeAllWindows()

def makeSplashLogo():
    '''Make a splash screen logo.'''
    border = 16
    xw, yw = 520, 240
    pix = qt4.QPixmap(xw, yw)
    pix.fill()
    p = qt4.QPainter(pix)

    # draw logo on pixmap
    logo = utils.getPixmap('logo.png')
    p.drawPixmap( xw/2 - logo.width()/2, border, logo )

    # add copyright text
    doc = qt4.QTextDocument()
    doc.setPageSize( qt4.QSizeF(xw, yw - 3*border - logo.height()) )
    f = qt4.qApp.font()
    f.setPointSize(14)
    doc.setDefaultFont(f)
    doc.setDefaultTextOption( qt4.QTextOption(qt4.Qt.AlignCenter) )
    doc.setHtml(splashcopyr % utils.version())
    p.translate(0, 2*border + logo.height())
    doc.drawContents(p)
    p.end()
    return pix

def excepthook(excepttype, exceptvalue, tracebackobj):
    '''Show exception dialog if an exception occurs.'''
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
    import veusz.document as document
    for expfn, vsz in zip(exports, args[1:]):
        doc = document.Document()
        ci = document.CommandInterpreter(doc)
        ci.Load(vsz)
        ci.run('Export(%s)' % repr(expfn))

def mainwindow(args):
    '''Open the main window with any loaded files.'''
    from veusz.windows.mainwindow import MainWindow
    if len(args) > 1:
        # load in filenames given
        for filename in args[1:]:
            MainWindow.CreateWindow(filename)
    else:
        # create blank window
        MainWindow.CreateWindow()

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
        if isinstance(a, str):
            out.append( a.decode(enc) )
        else:
            out.append(a)
    return out

class ImportThread(qt4.QThread):
    '''Do import of main code within another thread.
    Main application runs when this is done
    '''
    def run(self):
        import veusz.setting
        import veusz.widgets
        import veusz.utils.vzdbus as vzdbus

        vzdbus.setup()

class AppRunner(qt4.QObject):
    '''Object to run application. We have to do this to get an
    event loop while importing, etc.'''

    def __init__(self, options, args):

        qt4.QObject.__init__(self)

        self.splash = None
        if not (options.listen or options.export):
            # show the splash screen on normal start
            self.splash = qt4.QSplashScreen(makeSplashLogo())
            self.splash.show()

        self.options = options
        self.args = args

        # optionally load a translation
        if options.translation:
            trans = qt4.QTranslator()
            trans.load(options.translation)
            qt4.qApp.installTranslator(trans)

        self.thread = ImportThread()
        self.connect( self.thread, qt4.SIGNAL('finished()'),
                      self.slotStartApplication )
        self.thread.start()

    def slotStartApplication(self):
        '''Main start of application.'''

        options = self.options
        args = self.args

        import veusz.document
        import veusz.setting

        # for people who want to run any old script
        veusz.setting.transient_settings['unsafe_mode'] = bool(
            options.unsafe_mode)

        # load any requested plugins
        if options.plugin:
            veusz.document.Document.loadPlugins(pluginlist=options.plugin)

        # different modes
        if options.listen:
            # listen to incoming commands
            listen(args, quiet=options.quiet)
        elif options.export:
            # export files to make images
            if len(options.export) != len(args)-1:
                parser.error(
                    'export option needs same number of documents and '
                    'output files')
            export(options.export, args)
            qt4.qApp.quit()
        else:
            # standard start main window
            mainwindow(args)

        # clear splash when startup done
        if self.splash is not None:
            self.splash.finish(qt4.qApp.topLevelWidgets()[0])

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

    app = qt4.QApplication(sys.argv)
    app.connect(app, qt4.SIGNAL('lastWindowClosed()'),
                app, qt4.SLOT('quit()'))
    sys.excepthook = excepthook

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
    options, args = parser.parse_args( app.argv() )

    # convert args to unicode from filesystem strings
    args = convertArgsUnicode(args)

    s = AppRunner(options, args)
    app.exec_()

# if ran as a program
if __name__ == '__main__':
    #import cProfile
    #cProfile.run('run()', 'outprofile.dat')
    run()

