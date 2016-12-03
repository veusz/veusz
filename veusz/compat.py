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
six-like compatibility module between python2 and python3

Rolled own, because I can control the naming better (everying starts
with a 'c')
"""

import itertools
import locale
import os
import sys

cpy3 = sys.version_info[0] == 3

if cpy3:
    # py3

    # builtins
    import builtins as cbuiltins
    from io import StringIO as CStringIO, BytesIO as CBytesIO
    import urllib.request as curlrequest

    # imports
    import pickle

    # range function
    crange = range

    # zip function
    czip = zip

    # function to create user strings
    cstr = str

    # base string type
    cbasestr = str

    # bytes-like object
    cbytes = bytes

    # unicode-like object
    cunicode = str

    # iterate over dict
    def citems(d):
        return d.items()
    def ckeys(d):
        return d.keys()
    def cvalues(d):
        return d.values()

    # next iterator
    cnext = next

    # python3 compatible iterator
    CIterator = object

    # python3 object with __bool__/__nonzero__
    class CBool(object):
        def __bool__(self):
            return self.cbool()

    # exec function
    cexec = getattr(cbuiltins, 'exec')

    # execfile
    def cexecfile(filename, globaldict):
        with open(filename) as f:
            code = compile(f.read(), filename, 'exec')
        cexec(code, globaldict)

    # convert strerror exception to string
    def cstrerror(ex):
        return ex.strerror

    # text repr (compatible with python2)
    def crepr(v):
        if isinstance(v, str):
            return 'u' + repr(v)
        return repr(v)

    # convert exception to a user string
    def cexceptionuser(ex):
        return str(ex)

    # get current directory (as unicode)
    cgetcwd = os.getcwd

else:
    # py2

    # builtins
    import __builtin__ as cbuiltins

    # imports
    import cPickle as pickle
    from StringIO import StringIO as CStringIO
    from io import BytesIO as CBytesIO
    import urllib2 as curlrequest

    # range function
    crange = xrange

    # zip function
    czip = itertools.izip

    # function to create user strings
    cstr = unicode

    # base string
    cbasestr = basestring

    # bytes-like object
    cbytes = str

    # unicode-like string
    cunicode = unicode

    # iterate over dict
    def citems(d):
        return d.iteritems()
    def ckeys(d):
        return d.iterkeys()
    def cvalues(d):
        return d.itervalues()

    # next iterator
    def cnext(i):
        return i.next()

    # python3 compatible iterator
    class CIterator(object):
        def next(self):
            return type(self).__next__(self)

    # python3 object with __bool__/__nonzero__
    class CBool(object):
        def __nonzero__(self):
            return self.cbool()

    # exec function
    def cexec(text, globdict):
        """An exec-like function.

        As veusz always supplies a globals and no locals, we simplify this."""

        # this is done like this to avoid a compile-time error in py3
        code = 'exec text in globdict'
        exec(code)

    # execfile
    def cexecfile(filename, globaldict):
        execfile(filename, globaldict)

    # convert strerror exception to string
    def cstrerror(ex):
        if isinstance(ex.strerror, str):
            deflocale = locale.getdefaultlocale()[1] or 'ascii'
            return ex.strerror.decode(deflocale)
        else:
            return ex.strerror

    # sometimes exceptions come as unicode, sometimes as strings
    # encoded in ascii, so we have to decode
    def cexceptionuser(ex):
        if hasattr(ex, 'strerror') and isinstance(ex.strerror, str):
            # comes from operating system as encoded
            deflocale = locale.getdefaultlocale()[1] or 'ascii'
            return str(ex).decode(deflocale)
        else:
            # let's hope this works
            return unicode(ex)

    # py2/3 repr
    crepr = repr

    # unicode getcwd
    cgetcwd = os.getcwdu
