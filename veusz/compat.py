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

import sys
import itertools

cpy3 = sys.version_info[0] == 3

if cpy3:
    # py3
    crange = range

    # zip function
    czip = zip

    # function to create user strings
    cstr = str

    # base string type
    cstrbase = str

    # iterate over dict
    def citems(d):
        return d.items()
    def ckeys(d):
        return d.keys()
    def cvalues(d):
        return d.values()

else:
    # py2
    crange = xrange

    # zip function
    czip = itertools.izip

    # function to create user strings
    cstr = unicode

    # iterate over dict
    def citems(d):
        return d.iteritems()
    def ckeys(d):
        return d.iterkeys()
    def cvalues(d):
        return d.itervalues()
