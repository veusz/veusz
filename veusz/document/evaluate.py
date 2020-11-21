#    Copyright (C) 2016 Jeremy S. Sanders
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

from __future__ import division
from collections import defaultdict
import os.path
import re
import datetime

import numpy as N

from . import colors

from ..compat import citems, cstr, cexec
from .. import setting
from .. import utils
from .. import datasets
from .. import qtall as qt

# python identifier
identifier_re = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
# for splitting
identifier_split_re = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

# python module
module_re = re.compile(r'^[A-Za-z_\.]+$')

# function(arg1, arg2...) for custom functions
# not quite correct as doesn't check for commas in correct places
function_re = re.compile(r'''
^([A-Za-z_][A-Za-z0-9_]*)[ ]*  # identifier
\((                            # begin args
(?: [ ]* ,? [ ]* [A-Za-z_][A-Za-z0-9_]* )*     # named args
(?: [ ]* ,? [ ]* \*[A-Za-z_][A-Za-z0-9_]* )?   # *args
(?: [ ]* ,? [ ]* \*\*[A-Za-z_][A-Za-z0-9_]* )? # **kwargs
)\)$                           # endargs''', re.VERBOSE)

def _(text, disambiguation=None, context="Evaluate"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

# Notes on Security
# -----------------

# Security states:
#  * Secure
#    - if new document
#    - if loaded from secure location
#    - or allow set in dialog
#  * Insecure (skip in dialog)

# Security context:
#  * Executing statements when loading
#  * Importing functions
#  * Evaluating expressions (non checking Python)

class Evaluate:
    """Class to manage evaluation of expressions in a special environment."""

    def __init__(self, doc):
        self.doc = doc

        # directories to examine when importing
        self.importpath = []

        self.wipe()

    def wipe(self):
        """Clear current customs."""

        # store custom functions and constants
        # consists of tuples of (name, type, value)
        # type is constant or function
        # we use this format to preserve evaluation order
        self.def_imports = []
        self.def_definitions = []
        self.def_colors = []
        self.def_colormaps = []

        #self.customs = []

        # this is the context used to evaluate expressions
        self.context = {}

        # copy default colormaps
        self.colormaps = utils.ColorMaps()
        self.colors = colors.Colors()

        self.update()

        # copies of validated compiled expressions
        self.compiled = {}
        self.compfailed = set()
        self.compfailedchangeset = -1

        # cached expressions which have been already evaluated as datasets
        self.exprdscache = {}
        self.exprdscachechangeset = None

        # whether we hit security tests
        self.setSecurity(False)

    def update(self):
        """To be called after custom constants or functions are changed.
        This sets up a safe environment where things can be evaluated
        """

        c = self.context
        c.clear()

        # add numpy things
        # we try to avoid various bits and pieces for safety
        for name, val in citems(N.__dict__):
            if ( (callable(val) or type(val)==float) and
                 name not in __builtins__ and
                 name[:1] != '_' and name[-1:] != '_' ):
                c[name] = val

        # safe functions
        c['os_path_join'] = os.path.join
        c['os_path_dirname'] = os.path.dirname
        c['veusz_markercodes'] = tuple(utils.MarkerCodes)

        # helpful functions for expansion
        c['ENVIRON'] = dict(os.environ)
        c['DATE'] = self._evalformatdate
        c['TIME'] = self._evalformattime
        c['DATA'] = self._evaldata
        c['FILENAME'] = self._evalfilename
        c['BASENAME'] = self._evalbasename
        c['ESCAPE'] = utils.latexEscape
        c['SETTING'] = self._evalsetting
        c['LANG'] = self._evallang

        for name, val in self.def_imports:
            self._updateImport(name, val)

        for name, val in self.def_definitions:
            self._updateDefinition(name, val)

        self.colors.wipe()
        for name, val in self.def_colors:
            self.colors.addColor(name, val)
        self.colors.updateModel()

        self.colormaps.wipe()
        for name, val in self.def_colormaps:
            self._updateColormap(name, val)

    def setSecurity(self, secure):
        """Updated the security context."""
        oldsecure = getattr(self, 'secure_document', False)

        self.secure_document = secure
        self.doc.sigSecuritySet.emit(secure)

        if not oldsecure and secure:
            # if we're now secure, and were not previously, update
            # context
            self.exprdscache = {}
            self.exprdscachechangeset = None
            self.update()

    def updateSecurityFromPath(self):
        """Make document secure if in a secure location."""
        filename = self.doc.filename
        absfilename = os.path.abspath(filename)
        paths = setting.settingdb['secure_dirs'] + [
            utils.exampleDirectory]
        for dirname in paths:
            absdirname = os.path.abspath(dirname)
            if absfilename.startswith(absdirname + os.sep):
                self.setSecurity(True)

    def inSecureMode(self):
        """Is the document in a safe location?"""
        return (
            setting.transient_settings['unsafe_mode'] or
            self.secure_document
        )

    def _updateImport(self, module, val):
        """Add an import statement to the eval function context."""
        if module_re.match(module):
            # work out what is safe to import
            symbols = identifier_split_re.findall(val)
            if self._checkImportsSafe():
                if symbols:
                    defn = 'from %s import %s' % (
                        module, ', '.join(symbols))
                    try:
                        cexec(defn, self.context)
                    except Exception:
                        self.doc.log(_(
                            "Failed to import '%s' from module '%s'") % (
                                ', '.join(toimport), module))
                        return
                else:
                    defn = 'import %s' % module
                    try:
                        cexec(defn, self.context)
                    except Exception:
                        self.doc.log(_(
                            "Failed to import module '%s'") % module)
                        return
            else:
                if not symbols:
                    self.doc.log(_("Did not import module '%s'") % module)
                else:
                    self.doc.log(_(
                        "Did not import '%s' from module '%s'") % (
                            ', '.join(list(symbols)), module))

        else:
            self.doc.log( _("Invalid module name '%s'") % module )

    def validateProcessColormap(self, colormap):
        """Validate and process a colormap value.

        Returns a list of B,G,R,alpha tuples or raises ValueError if a problem."""

        try:
            if len(colormap) < 2:
                raise ValueError( _("Need at least two entries in colormap") )
        except TypeError:
            raise ValueError( _("Invalid type for colormap") )

        out = []
        for entry in colormap:
            if entry == (-1,0,0,0):
                out.append(entry)
                continue

            for v in entry:
                try:
                    v - 0
                except TypeError:
                    raise ValueError(
                        _("Colormap entries should be numerical") )
                if v < 0 or v > 255:
                    raise ValueError(
                        _("Colormap entries should be between 0 and 255") )

            if len(entry) == 3:
                out.append( (int(entry[2]), int(entry[1]), int(entry[0]),
                             255) )
            elif len(entry) == 4:
                out.append( (int(entry[2]), int(entry[1]), int(entry[0]),
                             int(entry[3])) )
            else:
                raise ValueError( _("Each colormap entry consists of R,G,B "
                                    "and optionally alpha values") )

        return tuple(out)

    def _updateColormap(self, name, val):
        """Add a colormap entry."""

        try:
            cmap = self.validateProcessColormap(val)
        except ValueError as e:
            self.doc.log( cstr(e) )
        else:
            self.colormaps[ cstr(name) ] = cmap

    def _updateDefinition(self, name, val):
        """Update a function or constant in eval function context."""

        if identifier_re.match(name):
            defn = val
        else:
            m = function_re.match(name)
            if not m:
                self.doc.log(
                    _("Invalid function or constant specification '%s'") %
                    name)
                return
            name = m.group(1)
            args = m.group(2)
            defn = 'lambda %s: %s' % (args, val)

        # evaluate, but we ignore any unsafe commands or exceptions
        comp = self.compileCheckedExpression(defn)
        if comp is None:
            return
        try:
            self.context[name] = eval(comp, self.context)
        except Exception as e:
            self.doc.log( _(
                "Error evaluating '%s': '%s'") % (name, cstr(e)) )

    def compileCheckedExpression(self, expr, origexpr=None, log=True):
        """Compile expression and check for errors.

        origexpr is an expression to show in error messages. This is
        used if replacements have been done, etc.
        """

        try:
            return self.compiled[expr]
        except KeyError:
            pass

        # track failed compilations, so we only print them once
        if self.compfailedchangeset != self.doc.changeset:
            self.compfailedchangeset = self.doc.changeset
            self.compfailed.clear()
        elif expr in self.compfailed:
            return None

        if origexpr is None:
            origexpr = expr

        try:
            checked = utils.compileChecked(
                expr,
                ignoresecurity=self.inSecureMode(),
            )
        except utils.SafeEvalException as e:
            if log:
                self.doc.log(
                    _("Unsafe expression '%s': %s") % (origexpr, cstr(e)))
            self.compfailed.add(expr)
            return None
        except Exception as e:
            if log:
                self.doc.log(
                    _("Error in expression '%s': %s") % (origexpr, cstr(e)))
            return None
        else:
            self.compiled[expr] = checked
            return checked

    @staticmethod
    def _evalformatdate(fmt=None):
        """DATE() eval: return date with optional format."""
        d = datetime.date.today()
        return d.isoformat() if fmt is None else d.strftime(fmt)

    @staticmethod
    def _evalformattime(fmt=None):
        """TIME() eval: return time with optional format."""
        t = datetime.datetime.now()
        return t.isoformat() if fmt is None else t.strftime(fmt)

    def _evaldata(self, name, part='data'):
        """DATA(name, [part]) eval: return dataset as array."""
        if part not in ('data', 'perr', 'serr', 'nerr'):
            raise RuntimeError("Invalid dataset part '%s'" % part)
        if name not in self.doc.data:
            raise RuntimeError("Dataset '%s' does not exist" % name)
        data = getattr(self.doc.data[name], part)

        if isinstance(data, N.ndarray):
            return N.array(data)
        elif isinstance(data, list):
            return list(data)
        return data

    def _evalfilename(self):
        """FILENAME() eval: returns filename."""
        return utils.latexEscape(self.doc.filename)

    def _evalbasename(self):
        """BASENAME() eval: returns base filename."""
        return utils.latexEscape(os.path.basename(self.doc.filename))

    def _evalsetting(self, path):
        """SETTING() eval: return setting given full path."""
        return self.doc.resolveSettingPath(None, path).get()

    @staticmethod
    def _evallang(opts):
        lang = qt.QLocale().name()
        if lang in opts:
            return opts[lang]
        majorl = lang.split('_')[0]
        if majorl in opts:
            return opts[majorl]
        if 'default' in opts:
            return opts['default']
        return utils.latexEscape('NOLANG:%s' % str(lang))

    def evalDatasetExpression(self, expr, part='data', datatype='numeric',
                              dimensions=1):
        """Return dataset after evaluating a dataset expression.
        part is 'data', 'serr', 'perr' or 'nerr' - these are the
        dataset parts which are evaluated by the expression

        None is returned on error
        """

        key = (expr, part, datatype, dimensions)
        if self.exprdscachechangeset != self.doc.changeset:
            self.exprdscachechangeset = self.doc.changeset
            self.exprdscache.clear()
        elif key in self.exprdscache:
            return self.exprdscache[key]

        self.exprdscache[key] = ds = datasets.evalDatasetExpression(
            self.doc, expr, part=part, datatype=datatype, dimensions=dimensions)
        return ds

    def _checkImportsSafe(self):
        """Check whether symbols are safe to import."""

        # do import anyway
        if self.inSecureMode():
            return True

        self.doc.sigAllowedImports.emit()
        return self.inSecureMode()

    def getColormap(self, name, invert):
        """Get colormap with name given (returning grey if does not exist)."""
        cmap = self.colormaps.get(name, self.colormaps['grey'])
        if invert:
            if cmap[0][0] >= 0:
                return cmap[::-1]
            else:
                # ignore marker at beginning for stepped maps
                return tuple([cmap[0]] + list(cmap[-1:0:-1]))
        return cmap

    def saveCustomDefinitions(self, fileobj):
        """Save custom constants and functions."""
        for ctype, defns in (
                ('import', self.def_imports),
                ('definition', self.def_definitions),
                ('color', self.def_colors),
                ('colormap', self.def_colormaps)):

            for val in defns:
                fileobj.write(
                    'AddCustom(%s, %s, %s)\n' % (
                        utils.rrepr(ctype),
                        utils.rrepr(val[0]),
                        utils.rrepr(val[1])))

    def saveCustomFile(self, fileobj):
        """Export the custom settings to a file."""

        self.doc._writeFileHeader(fileobj, 'custom definitions')
        self.saveCustomDefinitions(fileobj)
