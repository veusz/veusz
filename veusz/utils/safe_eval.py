#    Copyright (C) 2013 Jeremy S. Sanders
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
###############################################################################

"""
'Safe' python code evaluation

The idea is to examine the compiled ast tree and chack for invalid
entries
"""

# don't do this as it affects imported files
# from __future__ import division
import ast

from ..compat import cstr, cpy3, cbuiltins
from .. import qtall as qt

def _(text, disambiguation=None, context='SafeEval'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

# blacklist of nodes
forbidden_nodes = set((
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ))

if hasattr(ast, 'Exec'):
    forbidden_nodes.add(ast.Exec)

# whitelist of allowed builtins
allowed_builtins = frozenset((
        'ArithmeticError',
        'AttributeError',
        'BaseException',
        'Exception',
        'False',
        'FloatingPointError',
        'IndexError',
        'KeyError',
        'NameError',
        'None',
        'OverflowError',
        'RuntimeError',
        'StandardError',
        'StopIteration',
        'True',
        'TypeError',
        'ValueError',
        'ZeroDivisionError',
        'abs',
        'all',
        'any',
        'apply',
        'basestring',
        'bin',
        'bool',
        'bytes',
        'callable',
        'chr',
        'cmp',
        'complex',
        'dict',
        'divmod',
        'enumerate',
        'filter',
        'float',
        'format',
        'frozenset',
        'hash',
        'hex',
        'id',
        'int',
        'isinstance',
        'issubclass',
        'iter',
        'len',
        'list',
        'long',
        'map',
        'max',
        'min',
        'next',
        'object',
        'oct',
        'ord',
        'pow',
        'print',
        'property',
        'range',
        'reduce',
        'repr',
        'reversed',
        'round',
        'set',
        'slice',
        'sorted',
        'str',
        'sum',
        'tuple',
        'unichr',
        'unicode',
        'xrange',
        'zip'
        ))

numpy_forbidden = set((
        'frombuffer',
        'fromfile',
        'getbuffer',
        'getbufsize',
        'load',
        'loads',
        'loadtxt',
        'ndfromtxt',
        'newbuffer',
        'pkgload',
        'recfromcsv',
        'recfromtxt',
        'save',
        'savetxt',
        'savez',
        'savez_compressed',
        'setbufsize',
        'seterr',
        'seterrcall',
        'seterrobj',
        ))

# blacklist using whitelist above
forbidden_builtins = ( set(cbuiltins.__dict__.keys()) - allowed_builtins |
                       numpy_forbidden )

class SafeEvalException(Exception):
    """Raised by safety errors in code."""
    pass

class CheckNodeVisitor(ast.NodeVisitor):
    """Visit ast nodes to look for unsafe entries."""

    def generic_visit(self, node):
        if type(node) in forbidden_nodes:
            raise SafeEvalException(_("%s not safe") % type(node))
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Name(self, name):
        if name.id[:2] == '__' or name.id in forbidden_builtins:
            raise SafeEvalException(
                _('Access to special names not allowed: "%s"') % name.id)
        self.generic_visit(name)

    def visit_Call(self, call):
        if not hasattr(call.func, 'id'):
            raise SafeEvalException(_("Function has no identifier"))

        if call.func.id[:2] == '__' or call.func.id in forbidden_builtins:
            raise SafeEvalException(
                _('Access to special functions not allowed: "%s"') %
                call.func.id)
        self.generic_visit(call)

    def visit_Attribute(self, attr):
        if not hasattr(attr, 'attr'):
            raise SafeEvalException(_('Access denied to attribute'))
        if ( attr.attr[:2] == '__' or attr.attr[:5] == 'func_' or
             attr.attr[:3] == 'im_' or attr.attr[:3] == 'tb_' ):
            raise SafeEvalException(
                _('Access to special attributes not allowed: "%s"') %
                attr.attr)
        self.generic_visit(attr)

def compileChecked(code, mode='eval', filename='<string>',
                   ignoresecurity=False):
    """Compile code, checking for security errors.

    Returns a compiled code object.
    mode = 'exec' or 'eval'
    """

    # python2 needs filename encoded
    if not cpy3:
        filename = filename.encode('utf-8')

    try:
        tree = ast.parse(code, filename, mode)
    except Exception as e:
        raise ValueError(_('Unable to parse file: %s') % cstr(e))

    if not ignoresecurity:
        visitor = CheckNodeVisitor()
        visitor.visit(tree)

    compiled = compile(tree, filename, mode)

    return compiled
