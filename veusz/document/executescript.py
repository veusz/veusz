#    Copyright (C) 2014 Jeremy S. Sanders
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

# note: no future statements here for backward compatibility

import sys
import os.path
import traceback

from .. import qtall as qt4
from .. import setting
from .. import utils

from ..compat import cexec, cstr
from .commandinterface import CommandInterface

def _(text, disambiguation=None, context='ExecuteScript'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

class ExecuteError(RuntimeError):
    """Error when loading document."""
    def __init__(self, text, backtrace=''):
        RuntimeError.__init__(self, text)
        self.backtrace = backtrace

def executeScript(thedoc, filename, script, callbackunsafe=None):
    """Execute a script for the document.

    This handles setting up the environment and checking for unsafe
    commands in the execution.

    filename: filename to supply in __filename__
    script: text to execute
    callbackunsafe: should be set to a function to ask the user whether it is
      ok to execute any unsafe commands found. Return True if ok.
    """

    def genexception(exc):
        info = sys.exc_info()
        backtrace = ''.join(traceback.format_exception(*info))
        return ExecuteError(cstr(exc), backtrace=backtrace)

    # compile script and check for security (if reqd)
    unsafe = [setting.transient_settings['unsafe_mode']]
    while True:
        try:
            compiled = utils.compileChecked(
                script, mode='exec', filename=filename,
                ignoresecurity=unsafe[0])
            break
        except utils.SafeEvalException:
            if callbackunsafe is None or not callbackunsafe():
                raise ExecuteError(_("Unsafe command in script"))
            # repeat with unsafe mode switched on
            unsafe[0] = True
        except Exception as e:
            raise genexception(e)

    env = thedoc.eval_context.copy()
    interface = CommandInterface(thedoc)

    # allow safe commands as-is
    for cmd in interface.safe_commands:
        env[cmd] = getattr(interface, cmd)

    # define root node
    env['Root'] = interface.Root

    # wrap unsafe calls with a function to check whether ok
    def _unsafecaller(func):
        def wrapped(*args, **argsk):
            if not unsafe[0]:
                if callbackunsafe is None or not callbackunsafe():
                    raise ExecuteError(_("Unsafe command in script"))
                unsafe[0] = True
                func(*args, **argsk)
        return wrapped
    for name in interface.unsafe_commands:
        env[name] = _unsafecaller(getattr(interface, name))

    # get ready for loading document
    env['__file__'] = filename
    # allow import to happen relative to loaded file
    interface.AddImportPath( os.path.dirname(os.path.abspath(filename)) )

    thedoc.wipe()
    thedoc.suspendUpdates()

    try:
        # actually run script text
        cexec(compiled, env)
    except ExecuteError:
        thedoc.enableUpdates()
        raise
    except Exception as e:
        thedoc.enableUpdates()
        raise genexception(e)

    # success!!
    thedoc.enableUpdates()
    thedoc.setModified(False)
    thedoc.clearHistory()
