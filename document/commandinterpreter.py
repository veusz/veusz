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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##############################################################################

# $Id$

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

import sys
import traceback
import string

import commandinterface
import utils

class CommandInterpreter:
    def __init__(self, document):
        """ Initialise object with the document it interfaces."""
        self.document = document

        # set up interface to document
        self.interface = commandinterface.CommandInterface(document)

        # initialise environment
        self.globals = globals()
        self.locals = locals()

        # save the stdout & stderr
        self.write_stdout = sys.stdout
        self.write_stderr = sys.stderr

        # import numarray into the environment
        exec "from numarray import *" in self.globals, self.locals

        # shortcut
        i = self.interface

        # define commands for interface
        self.cmds = { 'GPL': self.GPL,
                      'SetVerbose': i.SetVerbose,
                      'Add': i.Add,
                      'To': i.To,
                      'List': i.List,
                      'Get': i.Get,
                      'Set': i.Set,
                      'SetData': i.SetData,
                      'Print': i.Print,
                      'WriteEPS': i.WriteEPS,
                      'Resize': i.Resize }

        for i in self.cmds.items():
            self.globals[ i[0] ] = i[1]

    def setOutputs(self, stdout, stderr):
        """ Assign the environment output files."""
        self.write_stdout = stdout
        self.write_stderr = stderr

    def _pythonise(self, text):
        """Internal routine to convert commands in the form Cmd a b c into Cmd(a,b,c)."""
        
        out = ''
        # iterate over lines
        for l in string.split(text, '\n'):
            parts = string.split(l)

            # turn Cmd a b c into Cmd(a,b,c)
            if len(parts) != 0 and parts[0] in self.cmds.keys():
                l = utils.pythonise(l)

            out += l + '\n'

        return out
        
    def run(self, input, filename = None):
        """ Run a set of commands inside the preserved environment.

        input: a string with the commands to run
        filename: a filename to report if there are errors
        """
        
        if filename == None:
            filename = '<string>'

        # pythonise!
        input = self._pythonise(input)

        # ignore if blank
        if len(string.strip(input)) == 0:
            return

        # preserve output streams
        temp_stdout = sys.stdout
        temp_stderr = sys.stderr
        sys.stdout = self.write_stdout
        sys.stderr = self.write_stderr

        # count number of newlines in expression
        # If it's 2, then execute as a single statement (print out result)
        if string.count(input, '\n') == 2:
            stattype = 'single'
        else:
            stattype = 'exec'

        # first compile the code to check for syntax errors
        try:
            c = compile(input, filename, stattype)
        except (OverflowError, ValueError, SyntaxError), e:
            i = sys.exc_info()
            backtrace = traceback.format_exception( *i )
            for l in backtrace:
                sys.stderr.write(l)

        else:
            # execute the code
            try:
                exec c in self.globals, self.locals
            except Exception, e:
                # print out the backtrace to stderr
                i = sys.exc_info()
                backtrace = traceback.format_exception( *i )
                for l in backtrace:
                    sys.stderr.write(l)            

        # return output streams
        sys.stdout = temp_stdout
        sys.stderr = temp_stderr

    def runFile(self, fileobject):
        """ Run a file in the preserved environment."""

        # preserve output streams
        temp_stdout = sys.stdout
        temp_stderr = sys.stderr
        sys.stdout = self.write_stdout
        sys.stderr = self.write_stderr

        # actually run the code
        try:
            exec fileobject in self.globals, self.locals
        except Exception, e:
            # print out the backtrace to stderr
            i = sys.exc_info()
            backtrace = traceback.format_exception( *i )
            for l in backtrace:
                sys.stderr.write(l)            

        # return output streams
        sys.stdout = temp_stdout
        sys.stderr = temp_stderr

    # FIXME: need a version of this that can throw exceptions instead
    def evaluate(self, expression):
        """ Evaluate an expression in the environment."""

        # preserve output streams
        temp_stdout = sys.stdout
        temp_stderr = sys.stderr
        sys.stdout = self.write_stdout
        sys.stderr = self.write_stderr

        # actually run the code
        try:
            retn = eval(expression, self.globals, self.locals)
        except Exception, e:
            # print out the backtrace to stderr
            i = sys.exc_info()
            backtrace = traceback.format_exception( *i )
            for l in backtrace:
                sys.stderr.write(l)
            retn = None

        # return output streams
        sys.stdout = temp_stdout
        sys.stderr = temp_stderr

        return retn

    def GPL(self):
        """Write the GPL to the console window."""
        # FIXME: This should open up a separate window
        # FIXME: We need to incode the path to the file somewhere
        file = open('COPYING', 'r')
        for line in file:
            sys.stdout.write(line)

