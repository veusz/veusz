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

# get globals before things are imported
_globals = globals()

import sys
import traceback
import pickle
import os.path

import commandinterface
import veusz.utils as utils

class CommandInterpreter:
    """Class for executing commands in the Veusz command line language."""

    def __init__(self, document):
        """ Initialise object with the document it interfaces."""
        self.document = document

        # set up interface to document
        self.interface = commandinterface.CommandInterface(document)

        # initialise environment (make a copy from inital globals)
        self.globals = _globals.copy()

        # save the stdout & stderr
        self.write_stdout = sys.stdout
        self.write_stderr = sys.stderr

        # import numarray into the environment
        exec "from numarray import *" in self.globals

        # shortcut
        i = self.interface

        # define commands for interface
        self.cmds = { 
            'Action': i.Action,
            'Add': i.Add,
            'Export': i.Export,
            'Get': i.Get,
            'GetChildren': i.GetChildren,
            'GetData': i.GetData,
            'GetDatasets': i.GetDatasets,
            'GPL': self.GPL,
            'ImportString': i.ImportString,
            'ImportString2D': i.ImportString2D,
            'ImportFile': i.ImportFile,
            'ImportFile2D': i.ImportFile2D,
            'ImportFileCSV': i.ImportFileCSV,
            'ImportFITSFile': i.ImportFITSFile,
            'List': i.List,
            'Load': self.Load,
            'Print': i.Print,
            'ReloadData': i.ReloadData,
            'Rename': i.Rename,
            'Remove': i.Remove,
            'Save': i.Save,
            'Set': i.Set,
            'SetData': i.SetData,
            'SetDataExpression': i.SetDataExpression,
            'SetData2D': i.SetData2D,
            'SetVerbose': i.SetVerbose,
            'To': i.To
            }

        for name, val in self.cmds.items():
            self.globals[name] = val

    def addCommand(self, name, command):
        """Add the given command to the list of available commands."""
        self.cmds[name] = command
        self.globals[name] = command

    def setOutputs(self, stdout, stderr):
        """ Assign the environment output files."""
        self.write_stdout = stdout
        self.write_stderr = stderr

    def _pythonise(self, text):
        """Internal routine to convert commands in the form Cmd a b c into Cmd(a,b,c)."""
        
        out = ''
        # iterate over lines
        for l in text.split('\n'):
            parts = l.split()

            # turn Cmd a b c into Cmd(a,b,c)
            if len(parts) != 0 and parts[0] in self.cmds.keys():
                l = utils.pythonise(l)

            out += l + '\n'

        return out
        
    def run(self, inputcmds, filename = None):
        """ Run a set of commands inside the preserved environment.

        inputcmds: a string with the commands to run
        filename: a filename to report if there are errors
        """
        
        if filename == None:
            filename = '<string>'

        # pythonise!
        inputcmds = self._pythonise(inputcmds)

        # ignore if blank
        if len(inputcmds.strip()) == 0:
            return

        # preserve output streams
        temp_stdout = sys.stdout
        temp_stderr = sys.stderr
        sys.stdout = self.write_stdout
        sys.stderr = self.write_stderr

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
            i = sys.exc_info()
            backtrace = traceback.format_exception( *i )
            for l in backtrace:
                sys.stderr.write(l)

        else:
            # execute the code
            try:
                exec c in self.globals
            except Exception:
                # print out the backtrace to stderr
                i = sys.exc_info()
                backtrace = traceback.format_exception( *i )
                for l in backtrace:
                    sys.stderr.write(l)            

        # return output streams
        sys.stdout = temp_stdout
        sys.stderr = temp_stderr

    def Load(self, filename):
        """Replace the document with a new one from the filename."""

        # FIXME: should update filename in main window
        f = open(filename, 'r')
        self.document.wipe()
        self.interface.To('/')
        self.runFile(f)
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

        # actually run the code
        try:
            exec fileobject in self.globals
        except Exception:
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
            retn = eval(expression, self.globals)
        except Exception:
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
        dirname = os.path.dirname(__file__)
        f = open(os.path.join(dirname, '..', 'COPYING'), 'rU')

        for line in f:
            sys.stdout.write(line)

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

        print name, args, namedargs
        try:
            retn = eval('%s(*_tmp_args0, **_tmp_args1)' % name)
        except Exception, e:
            # return exception picked if exception
            retn = e
            
        del self.globals['_tmp_args0']
        del self.globals['_tmp_args1']

        return pickle.dumps(retn)
    
