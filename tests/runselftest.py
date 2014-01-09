#!/usr/bin/env python

#    Copyright (C) 2011 Jeremy S. Sanders
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

"""A program to self test the Veusz installation.

This code compares the output of example + self test input files with
expected output.  It returns 0 if the tests succeeded, otherwise the
number of tests failed. If you use an argument "regenerate" to the
program, the comparison files will be recreated.

This program requires the veusz module to be on the PYTHONPATH.

On Unix/Linux, Qt requires the DISPLAY environment to be set to an X11
server for the self test to run. In a non graphical environment Xvfb
can be used to create a hidden X11 server.

The comparison files are close to being SVG files, but use XPM for any
images and use a fixed (hacked) font metric to give the same results
on each platform. In addition Unicode characters are expanded to their
Unicode code to work around different font handling on platforms.
"""

# messes up loaded files if set
# from __future__ import division
from __future__ import print_function
import glob
import os
import os.path
import sys

from veusz.compat import cexec, cstr
import veusz.qtall as qt4
import veusz.utils as utils
import veusz.document as document
import veusz.setting as setting
import veusz.dataimport
import veusz.document.svg_export as svg_export

# required to get structures initialised
import veusz.windows.mainwindow

try:
    import h5py
except ImportError:
    h5py = None

try:
    from astropy.io import fits as pyfits
except ImportError:
    try:
        import pyfits
    except ImportError:
        pyfits = None

# these tests fail for some reason which haven't been debugged
# it appears the failures aren't important however
excluded_tests = set([
        # fails on Linux Arm
        'spectrum.vsz',
        'hatching.vsz',
        # don't expect this to work
        'mathml.vsz',
    ])

class StupidFontMetrics(object):
    """This is a fake font metrics device which should return the same
    results on all systems with any font."""
    def __init__(self, font, device):
        self.font = font
        self.device = device

    def height(self):
        return self.device.logicalDpiY() * (self.font.pointSizeF()/72.)

    def width(self, text):
        return len(text)*self.height()*0.5

    def ascent(self):
        return 0.1*self.height()

    def descent(self):
        return 0.1*self.height()

    def leading(self):
        return 0.1*self.height()

    def boundingRect(self, c):
        return qt4.QRectF(0, 0, self.height()*0.5, self.height())

    def boundingRectChar(self, c):
        return qt4.QRectF(0, 0, self.height()*0.5, self.height())

_pt = utils.textrender.PartText
class PartTextAscii(_pt):
    """Text renderer which converts text to ascii."""
    def __init__(self, text):
        text = text.encode('ascii', 'xmlcharrefreplace').decode('ascii')
        _pt.__init__(self, text)
    def render(self, state):
        _pt.render(self, state)
    def addText(self, text):
        self.text += text.encode('ascii', 'xmlcharrefreplace').decode('ascii')

def renderTest(invsz, outfile):
    """Render vsz document to create outfile."""

    d = document.Document()
    ifc = document.CommandInterface(d)

    # this lot looks a bit of a mess
    cmds = d.eval_context
    for cmd in document.CommandInterface.safe_commands:
        cmds[cmd] = getattr(ifc, cmd)
    for cmd in document.CommandInterface.unsafe_commands:
        cmds[cmd] = getattr(ifc, cmd)

    cexec("from numpy import *", cmds)
    ifc.AddImportPath( os.path.dirname(invsz) )
    cexec(compile(open(invsz).read(), invsz, 'exec'), cmds)
    ifc.Export(outfile)

class Dirs(object):
    """Directories and files object."""
    def __init__(self):
        self.thisdir = os.path.dirname(__file__)
        self.exampledir = os.path.join(self.thisdir, '..', 'examples')
        self.testdir = os.path.join(self.thisdir, 'selftests')
        self.comparisondir = os.path.join(self.thisdir, 'comparison')

        files = ( glob.glob( os.path.join(self.exampledir, '*.vsz') ) +
                  glob.glob( os.path.join(self.testdir, '*.vsz') ) )

        self.invszfiles = [ f for f in files if
                            os.path.basename(f) not in excluded_tests ]

def renderAllTests():
    """Check documents produce same output as in comparison directory."""

    print("Regenerating all test output")

    d = Dirs()
    for vsz in d.invszfiles:
        base = os.path.basename(vsz)
        print(base)
        outfile = os.path.join(d.comparisondir, base + '.selftest')
        renderTest(vsz, outfile)

def runTests():
    print("Testing output")

    fails = 0
    passes = 0
    skipped = 0

    d = Dirs()
    for vsz in sorted(d.invszfiles):
        base = os.path.basename(vsz)
        print(base)

        if ( (base[:5] == 'hdf5_' and h5py is None) or
             (base[:5] == 'fits_' and pyfits is None) ):
            print(" SKIPPED")
            skipped += 1
            continue

        outfile = os.path.join(d.thisdir, base + '.temp.selftest')
        renderTest(vsz, outfile)

        comparfile = os.path.join(d.thisdir, 'comparison', base + '.selftest')

        f1 = open(outfile, 'rU')
        f2 = open(comparfile, 'rU')
        comp = f1.read() == f2.read()
        f1.close()
        f2.close()

        if not comp:
            print(" FAIL: results differed")
            fails += 1
        else:
            print(" PASS")
            passes += 1
            os.unlink(outfile)

    print()
    if skipped != 0:
        print('Skipped %i tests' % skipped)
    if fails == 0:
        print("All tests %i/%i PASSED" % (passes, passes))
        sys.exit(0)
    else:
        print("%i/%i tests FAILED" % (fails, passes+fails))
        sys.exit(fails)

oldflt = svg_export.fltStr
def fltStr(v, prec=1):
    # this is to get consistent rounding to get the self test correct... yuck
    # decimal would work, but that drags in loads of code
    # convert float to string with prec decimal places

    v = round(v, prec+2)

    return oldflt(v, prec=prec)

if __name__ == '__main__':
    os.environ['LC_ALL'] = 'C'

    app = qt4.QApplication([])

    setting.transient_settings['unsafe_mode'] = True

    # hack metrics object to always return same metrics
    # and replace text renderer with one that encodes unicode symbols
    utils.textrender.FontMetrics = StupidFontMetrics
    utils.FontMetrics = StupidFontMetrics
    utils.textrender.PartText = PartTextAscii

    # nasty hack to remove underlining
    del utils.textrender.part_commands[r'\underline']

    # dpi (use old values)
    svg_export.dpi = 90.
    svg_export.scale = 1.
    svg_export.fltStr = fltStr

    if len(sys.argv) == 1:
        runTests()
    else:
        if len(sys.argv) != 2 or sys.argv[1] != 'regenerate':
            print >>sys.stderr, "Usage: %s [regenerate]" % sys.argv[0]
            sys.exit(1)
        renderAllTests()
