#    Copyright (C) 2007 Jeremy S. Sanders
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

Based on the public domain code of Babar K. Zafar
http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/496746
(version 0.1 or 1.2 May 27 2006)

The idea is to examine the compiled ast tree and chack for invalid
entries

I have removed the timeout checking as this probably isn't a serious
problem for veusz documents
"""

import inspect, compiler.ast
import thread, time
import __builtin__
import os.path

import numpy as N

#----------------------------------------------------------------------
# Module globals.
#----------------------------------------------------------------------

# Toggle module level debugging mode.
DEBUG = False

# List of all AST node classes in compiler/ast.py.
all_ast_nodes = [name for (name, obj) in inspect.getmembers(compiler.ast)
                 if inspect.isclass(obj) and
                 issubclass(obj, compiler.ast.Node)]

# List of all builtin functions and types (ignoring exception classes).
all_builtins = [name for (name, obj) in inspect.getmembers(__builtin__)
                if inspect.isbuiltin(obj) or
                (inspect.isclass(obj) and not issubclass(obj, Exception))]

#----------------------------------------------------------------------
# Utilties.
#----------------------------------------------------------------------

def classname(obj):
    return obj.__class__.__name__

def get_node_lineno(node):
    return (node.lineno) and node.lineno or 0
       
#----------------------------------------------------------------------
# Restricted AST nodes & builtins.
#----------------------------------------------------------------------

# Deny evaluation of code if the AST contain any of the following nodes:
unallowed_ast_nodes = (
    #   'Add', 'And',
    #   'AssAttr', 'AssList', 'AssName', 'AssTuple',
    #   'Assert', 'Assign', 'AugAssign',
    'Backquote',
    #   'Bitand', 'Bitor', 'Bitxor', 'Break',
    #   'CallFunc', 'Class', 'Compare', 'Const', 'Continue',
    #   'Decorators', 'Dict', 'Discard', 'Div',
    #   'Ellipsis', 'EmptyNode',
    'Exec',
    #   'Expression', 'FloorDiv',
    #   'For',
    'From',
    #   'Function',
    #   'GenExpr', 'GenExprFor', 'GenExprIf', 'GenExprInner',
    #   'Getattr', 'Global', 'If',
    'Import',
    #   'Invert',
    #   'Keyword', 'Lambda', 'LeftShift',
    #   'List', 'ListComp', 'ListCompFor', 'ListCompIf', 'Mod',
    #   'Module',
    #   'Mul', 'Name', 'Node', 'Not', 'Or', 'Pass', 'Power',
    #   'Print', 'Printnl',
    'Raise',
    #    'Return', 'RightShift', 'Slice', 'Sliceobj',
    #   'Stmt', 'Sub', 'Subscript',
    'TryExcept', 'TryFinally',
    #   'Tuple', 'UnaryAdd', 'UnarySub',
    #   'While','Yield'
    )

# Deny evaluation of code if it tries to access any of the following builtins:
unallowed_builtins = (
    '__import__',
    #   'abs', 'apply', 'basestring', 'bool', 'buffer',
    #   'callable', 'chr', 'classmethod', 'cmp', 'coerce',
    'compile',
    #   'complex',
    'delattr',
    #   'dict',
    'dir',
    #   'divmod', 'enumerate',
    'eval', 'execfile', 'file',
    #   'filter', 'float', 'frozenset',
    'getattr', 'globals', 'hasattr',
    #    'hash', 'hex', 'id',
    'input',
    #   'int', 'intern', 'isinstance', 'issubclass', 'iter',
    #   'len', 'list',
    'locals',
    #   'long', 'map', 'max', 'min', 'object', 'oct',
    'open',
    #   'ord', 'pow', 'property', 'range',
    'raw_input',
    #   'reduce',
    'reload',
    #   'repr', 'reversed', 'round', 'set',
    'setattr',
    #   'slice', 'sorted', 'staticmethod',  'str', 'sum', 'super',
    #   'tuple', 'type', 'unichr', 'unicode',
    'vars',
    #    'xrange', 'zip'
    )

# checks there are no obvious mistakes above
#for ast_name in unallowed_ast_nodes:
#    assert ast_name in all_ast_nodes
#for name in unallowed_builtins:
#    assert name in all_builtins

# faster lookup
unallowed_ast_nodes = dict( (i, True) for i in unallowed_ast_nodes )
unallowed_builtins = dict( (i, True) for i in unallowed_builtins )

#----------------------------------------------------------------------
# Restricted attributes.
#----------------------------------------------------------------------

# In addition to these we deny access to all lowlevel attrs (__xxx__).
unallowed_attr = (
    'im_class', 'im_func', 'im_self',
    'func_code', 'func_defaults', 'func_globals', 'func_name',
    'tb_frame', 'tb_next',
    'f_back', 'f_builtins', 'f_code', 'f_exc_traceback',
    'f_exc_type', 'f_exc_value', 'f_globals', 'f_locals' )
unallowed_attr = dict( (i, True) for i in unallowed_attr )

def is_unallowed_attr(name):
    if name == '__file__':
        return False
    return ( (name[:2] == '__' and name[-2:] == '__') or
             (name in unallowed_attr) )

#----------------------------------------------------------------------
# SafeEvalVisitor.
#----------------------------------------------------------------------

class SafeEvalError(object):
    """
    Base class for all which occur while walking the AST.

    Attributes:
      errmsg = short decription about the nature of the error
      lineno = line offset to where error occured in source code
    """
    def __init__(self, errmsg, lineno):
        self.errmsg, self.lineno = errmsg, lineno
    def __str__(self):
        return "line %d : %s" % (self.lineno, self.errmsg)

class SafeEvalASTNodeError(SafeEvalError):
    "Expression/statement in AST evaluates to a restricted AST node type."
    pass
class SafeEvalBuiltinError(SafeEvalError):
    "Expression/statement in tried to access a restricted builtin."
    pass
class SafeEvalAttrError(SafeEvalError):
    "Expression/statement in tried to access a restricted attribute."
    pass

class SafeEvalVisitor(object):
    """
    Data-driven visitor which walks the AST for some code and makes
    sure it doesn't contain any expression/statements which are
    declared as restricted in 'unallowed_ast_nodes'. We'll also make
    sure that there aren't any attempts to access/lookup restricted
    builtin declared in 'unallowed_builtins'. By default we also won't
    allow access to lowlevel stuff which can be used to dynamically
    access non-local envrioments.

    Interface:
      walk(ast) = validate AST and return True if AST is 'safe'

    Attributes:
      errors = list of SafeEvalError if walk() returned False

    Implementation:
    
    The visitor will automatically generate methods for all of the
    available AST node types and redirect them to self.ok or self.fail
    reflecting the configuration in 'unallowed_ast_nodes'. While
    walking the AST we simply forward the validating step to each of
    node callbacks which take care of reporting errors.
    """

    def __init__(self):
        "Initialize visitor by generating callbacks for all AST node types."
        self.errors = []
        for ast_name in all_ast_nodes:
            # Don't reset any overridden callbacks.
            if not getattr(self, 'visit' + ast_name, None):
                if ast_name in unallowed_ast_nodes:
                    setattr(self, 'visit' + ast_name, self.fail)
                else:
                    setattr(self, 'visit' + ast_name, self.ok)

    def walk(self, ast):
        "Validate each node in AST and return True if AST is 'safe'."
        self.visit(ast)
        return self.errors == []
        
    def visit(self, node, *args):
        "Recursively validate node and all of its children."
        fn = getattr(self, 'visit' + classname(node))
        if DEBUG: self.trace(node)
        fn(node, *args)
        for child in node.getChildNodes():
            self.visit(child, *args)

    def visitName(self, node, *args):
        "Disallow any attempts to access a restricted builtin/attr."
        name = node.getChildren()[0]
        lineno = get_node_lineno(node)
        if name in unallowed_builtins:
            self.errors.append(SafeEvalBuiltinError( \
                "access to builtin '%s' is denied" % name, lineno))
        elif is_unallowed_attr(name):
            self.errors.append(SafeEvalAttrError( \
                "access to attribute '%s' is denied" % name, lineno))
               
    def visitGetattr(self, node, *args):
        "Disallow any attempts to access a restricted attribute."
        name = node.attrname
        lineno = get_node_lineno(node)
        if is_unallowed_attr(name):
            self.errors.append(SafeEvalAttrError( \
                "access to attribute '%s' is denied" % name, lineno))
            
    def ok(self, node, *args):
        "Default callback for 'harmless' AST nodes."
        pass
    
    def fail(self, node, *args):
        "Default callback for unallowed AST nodes."
        lineno = get_node_lineno(node)
        self.errors.append(SafeEvalASTNodeError( \
            "execution of '%s' statements is denied" % classname(node),
            lineno))

    def trace(self, node):
        "Debugging utility for tracing the validation of AST nodes."
        print classname(node)
        for attr in dir(node):
            if attr[:2] != '__':
                print ' ' * 4, "%-15.15s" % attr, getattr(node, attr)

##########################################################################
# Veusz evaluation functions
##########################################################################

def checkContextOkay(context):
    """Check the context statements will be executed in.

    Returns True if context is okay
    """
    
    ctx_errkeys, ctx_errors = [], []
    for (key, obj) in context.items():
        if inspect.isbuiltin(obj):
            ctx_errkeys.append(key)
            ctx_errors.append("key '%s' : unallowed builtin %s" % (key, obj))
        if inspect.ismodule(obj):
            ctx_errkeys.append(key)
            ctx_errors.append("key '%s' : unallowed module %s" % (key, obj))

    if ctx_errors:
        raise SafeEvalContextException(ctx_errkeys, ctx_errors)

# # set up environment in dict
# veusz_eval_context = {}

# # add callables (not modules) and floats which don't override builtins
# for name, val in N.__dict__.iteritems():
#     if ( (callable(val) or type(val)==float) and
#          name not in __builtins__ and
#          name[:1] != '_' and name[-1:] != '_' ):
#         veusz_eval_context[name] = val

# # useful safe functions
# veusz_eval_context['os_path_join'] = os.path.join
# veusz_eval_context['os_path_dirname'] = os.path.dirname

def _filterExceptions(errs, securityonly):
    """Remove python exceptions from error list."""
    if securityonly:
        errs = [e for e in errs if (isinstance(e, SafeEvalException) or
                                    isinstance(e, SafeEvalError))]
    if errs:
        return errs
    else:
        return None

def checkCode(code, securityonly=False):
    """Check code, returning errors (if any) or None if okay.

    if securityonly is set, then don't return errors from Python
    exceptions.
    """

    # compiler can't parse strings with unicode
    code = code.encode('utf8')
    try:
        ast = compiler.parse(code)
    except SyntaxError, e:
        return _filterExceptions([e], securityonly)

    checker = SafeEvalVisitor()
    checker.walk(ast)
    return _filterExceptions(checker.errors, securityonly)

#----------------------------------------------------------------------
# Safe 'eval' replacement.
#----------------------------------------------------------------------

class SafeEvalException(Exception):
    "Base class for all safe-eval related errors."
    pass

class SafeEvalCodeException(SafeEvalException):
    """
    Exception class for reporting all errors which occured while
    validating AST for source code in safe_eval().

    Attributes:
      code   = raw source code which failed to validate
      errors = list of SafeEvalError
    """
    def __init__(self, code, errors):
        self.code, self.errors = code, errors
    def __str__(self):
        return '\n'.join([str(err) for err in self.errors])

class SafeEvalContextException(SafeEvalException):
    """
    Exception class for reporting unallowed objects found in the dict
    intended to be used as the local enviroment in safe_eval().

    Attributes:
      keys   = list of keys of the unallowed objects
      errors = list of strings describing the nature of the error
               for each key in 'keys'
    """
    def __init__(self, keys, errors):
        self.keys, self.errors = keys, errors
    def __str__(self):
        return '\n'.join([str(err) for err in self.errors])
        
class SafeEvalTimeoutException(SafeEvalException):
    """
    Exception class for reporting that code evaluation execeeded
    the given timelimit.

    Attributes:
      timeout = time limit in seconds
    """
    def __init__(self, timeout):
        self.timeout = timeout
    def __str__(self):
        return "Timeout limit execeeded (%s secs) during exec" % self.timeout

def exec_timed(code, context, timeout_secs):
    """
    Dynamically execute 'code' using 'context' as the global enviroment.
    SafeEvalTimeoutException is raised if execution does not finish within
    the given timelimit.
    """
    assert(timeout_secs > 0)

    signal_finished = False
    
    def alarm(secs):
        def wait(secs):
            for n in xrange(timeout_secs):
                time.sleep(1)
                if signal_finished: break
            else:
                thread.interrupt_main()
        thread.start_new_thread(wait, (secs,))

    try:
        alarm(timeout_secs)
        exec code in context
        signal_finished = True
    except KeyboardInterrupt:
        raise SafeEvalTimeoutException(timeout_secs)

def timed_safe_eval(code, context = {}, timeout_secs = 5):
    """
    Validate source code and make sure it contains no unauthorized
    expression/statements as configured via 'unallowed_ast_nodes' and
    'unallowed_builtins'. By default this means that code is not
    allowed import modules or access dangerous builtins like 'open' or
    'eval'. If code is considered 'safe' it will be executed via
    'exec' using 'context' as the global environment. More details on
    how code is executed can be found in the Python Reference Manual
    section 6.14 (ignore the remark on '__builtins__'). The 'context'
    enviroment is also validated and is not allowed to contain modules
    or builtins. The following exception will be raised on errors:

      if 'context' contains unallowed objects = 
        SafeEvalContextException

      if code is didn't validate and is considered 'unsafe' = 
        SafeEvalCodeException

      if code did not execute within the given timelimit =
        SafeEvalTimeoutException
    """   
    ctx_errkeys, ctx_errors = [], []
    for (key, obj) in context.items():
        if inspect.isbuiltin(obj):
            ctx_errkeys.append(key)
            ctx_errors.append("key '%s' : unallowed builtin %s" % (key, obj))
        if inspect.ismodule(obj):
            ctx_errkeys.append(key)
            ctx_errors.append("key '%s' : unallowed module %s" % (key, obj))

    if ctx_errors:
        raise SafeEvalContextException(ctx_errkeys, ctx_errors)

    ast = compiler.parse(code)
    checker = SafeEvalVisitor()

    if checker.walk(ast):
        exec_timed(code, context, timeout_secs)
    else:
        raise SafeEvalCodeException(code, checker.errors)
       
#----------------------------------------------------------------------
# Basic tests.
#----------------------------------------------------------------------

import unittest

class TestSafeEval(unittest.TestCase):
    def test_builtin(self):
        # attempt to access a unsafe builtin
        self.assertRaises(SafeEvalException,
            timed_safe_eval, "open('test.txt', 'w')")

    def test_getattr(self):
        # attempt to get arround direct attr access
        self.assertRaises(SafeEvalException, \
            timed_safe_eval, "getattr(int, '__abs__')")

    def test_func_globals(self):
        # attempt to access global enviroment where fun was defined
        self.assertRaises(SafeEvalException, \
            timed_safe_eval, "def x(): pass; print x.func_globals")

    def test_lowlevel(self):
        # lowlevel tricks to access 'object'
        self.assertRaises(SafeEvalException, \
            timed_safe_eval, "().__class__.mro()[1].__subclasses__()")

    def test_timeout_ok(self):
        # attempt to exectute 'slow' code which finishes within timelimit
        def test(): time.sleep(2)
        env = {'test':test}
        timed_safe_eval("test()", env, timeout_secs = 5)

    def test_timeout_exceed(self):
        # attempt to exectute code which never teminates
        self.assertRaises(SafeEvalException, \
            timed_safe_eval, "while 1: pass")

    def test_invalid_context(self):
        # can't pass an enviroment with modules or builtins
        env = {'f' : __builtins__.open, 'g' : time}
        self.assertRaises(SafeEvalException, \
            timed_safe_eval, "print 1", env)

    def test_callback(self):
        # modify local variable via callback
        self.value = 0
        def test(): self.value = 1
        env = {'test':test}
        timed_safe_eval("test()", env)
        self.assertEqual(self.value, 1)

if __name__ == "__main__":
    unittest.main()

#----------------------------------------------------------------------
# End unittests
#----------------------------------------------------------------------
