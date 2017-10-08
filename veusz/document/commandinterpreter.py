# commandinterpreter.py
# this module handles the command line interface interpreter

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

"""
A module for the execution of user 'macro' code inside a special
environment. That way the commandline can be used to interact with
the app without worrying about the app internals.

The way this works is to create an evironment specific to the class
consisting of globals & locals.

Commands can be run inside the environment. Python errors are trapped
and dumped out to stderr.

stderr and stdout can be reassigned in the environment to point to
an alternative interface. They should point to objects providing
a write() interface.

This class is modelled on the one described in
'GUI Programming in Python: QT Edition' (Boudewijn Rempt)
"""

from __future__ import division, print_function

# get globals before things are imported
_globals = globals()

import sys
import traceback
import io
import os.path

from ..compat import pickle, cexec
from .commandinterface import CommandInterface
from .. import utils

class CommandInterpreter(object):
    """Class for executing commands in the Veusz command line language."""

    def __init__(self, document):
        """ Initialise object with the document it interfaces."""
        self.document = document

        # set up interface to document
        self.interface = CommandInterface(document)

        # initialise environment (make a copy from inital globals)
        self.globals = _globals.copy()

        # save the stdout & stderr
        self.write_stdout = sys.stdout
        self.write_stderr = sys.stderr
        self.read_stdin = sys.stdin

        # import numpy into the environment
        cexec("from numpy import *", self.globals)

        # define root object
        self.globals['Root'] = self.interface.Root

        # shortcut
        ifc = self.interface

        # define commands for interface
        self.cmds = {}
        for cmd in (
                CommandInterface.safe_commands +
                CommandInterface.unsafe_commands +
                CommandInterface.import_commands):
            self.cmds[cmd] = getattr(ifc, cmd)
        self.cmds['GPL'] = self.GPL
        self.cmds['Load'] = self.Load

        self.globals.update( self.cmds )

    def addCommand(self, name, command):
        """Add the given command to the list of available commands."""
        self.cmds[name] = command
        self.globals[name] = command

    def setFiles(self, stdout, stderr, stdin):
        """Assign the environment input/output files."""
        self.write_stdout = stdout
        self.write_stderr = stderr
        self.read_stdin = stdin

    def _pythonise(self, text):
        """Internal routine to convert commands in the form Cmd a b c into Cmd(a,b,c)."""

        out = ''
        # iterate over lines
        for line in text.split('\n'):
            parts = line.split()

            # turn Cmd a b c into Cmd(a,b,c)
            if len(parts) != 0 and parts[0] in self.cmds:
                line = utils.pythonise(line)

            out += line + '\n'

        return out

    def run(self, inputcmds, filename = None):
        """ Run a set of commands inside the preserved environment.

        inputcmds: a string with the commands to run
        filename: a filename to report if there are errors
        """

        if filename is None:
            filename = '<string>'

        # pythonise!
        inputcmds = self._pythonise(inputcmds)

        # ignore if blank
        if len(inputcmds.strip()) == 0:
            return

        # preserve output streams
        saved = sys.stdout, sys.stderr, sys.stdin
        sys.stdout, sys.stderr, sys.stdin = (
            self.write_stdout, self.write_stderr, self.read_stdin)

        # count number of newlines in expression
        # If it's 2, then execute as a single statement (print out result)
        if inputcmds.count('\n') == 2:
            stattype = 'single'
        else:
            stattype = 'exec'

        # first compile the code to check for syntax errors
        try:
            c = compile(inputcmds, filename, stattype)
        except (OverflowError, ValueError, SyntaxError):
            info = sys.exc_info()
            backtrace = traceback.format_exception(*info)
            for line in backtrace:
                sys.stderr.write(line)

        else:
            # block update signals from document while updating
            with self.document.suspend():
                try:
                    # execute the code
                    cexec(c, self.globals)
                except:
                    # print out the backtrace to stderr
                    info = sys.exc_info()
                    backtrace = traceback.format_exception(*info)
                    for line in backtrace:
                        sys.stderr.write(line)

        # return output streams
        sys.stdout, sys.stderr, sys.stdin = saved

    def Load(self, filename):
        """Replace the document with a new one from the filename."""

        with io.open(filename, 'rU', encoding='utf8') as f:
            self.document.wipe()
            self.interface.To('/')
            oldfile = self.globals['__file__']
            self.globals['__file__'] = os.path.abspath(filename)

            self.interface.importpath.append(
                os.path.dirname(os.path.abspath(filename)))
            self.runFile(f)
            self.interface.importpath.pop()
            self.globals['__file__'] = oldfile
            self.document.setModified()
            self.document.setModified(False)
            self.document.clearHistory()

    def runFile(self, fileobject):
        """ Run a file in the preserved environment."""

        # preserve output streams
        temp_stdout = sys.stdout
        temp_stderr = sys.stderr
        sys.stdout = self.write_stdout
        sys.stderr = self.write_stderr

        with self.document.suspend():
            # actually run the code
            try:
                cexec(fileobject.read(), self.globals)
            except Exception:
                # print out the backtrace to stderr
                info = sys.exc_info()
                backtrace = traceback.format_exception(*info)
                for line in backtrace:
                    sys.stderr.write(line)

        # return output streams
        sys.stdout = temp_stdout
        sys.stderr = temp_stderr

    def evaluate(self, expression):
        """Evaluate an expression in the environment."""

        # preserve output streams
        temp_stdout = sys.stdout
        temp_stderr = sys.stderr
        sys.stdout = self.write_stdout
        sys.stderr = self.write_stderr

        # actually run the code
        try:
            retn = eval(expression, self.globals)
        except Exception:
            # print out the backtrace to stderr
            info = sys.exc_info()
            backtrace = traceback.format_exception(*info)
            for line in backtrace:
                sys.stderr.write(line)
            retn = None

        # return output streams
        sys.stdout = temp_stdout
        sys.stderr = temp_stderr

        return retn

    def GPL(self):
        """Write the GPL to the console window."""
        sys.stdout.write( utils.getLicense() )

    def runPickle(self, command):
        """Run a pickled command given as arguments.

        command should consist of following:
        dumps( (name, args, namedargs) )
        name is name of function to execute in environment
        args are the arguments (list)
        namedargs are the named arguments (dict).
        """

        name, args, namedargs = pickle.loads(command)
        self.globals['_tmp_args0'] = args
        self.globals['_tmp_args1'] = namedargs

        #print(name, args, namedargs)
        try:
            retn = eval('%s(*_tmp_args0, **_tmp_args1)' % name)
        except Exception as e:
            # return exception picked if exception
            retn = e

        del self.globals['_tmp_args0']
        del self.globals['_tmp_args1']

        return pickle.dumps(retn)
