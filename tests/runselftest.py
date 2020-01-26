#!/usr/bin/env python3

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
can be used to create a hidden X11 server.  Alternatively, set the
environment variable QT_QPA_PLATFORM=minimal to avoid the X11
dependency.

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
import subprocess
import optparse

# this needs to be set before main imports
os.environ['LC_ALL'] = 'C'

try:
    import h5py
except ImportError:
    h5py = None

from veusz.compat import cexec, cstr, copenuniversal
import veusz.qtall as qt
import veusz.utils as utils
import veusz.document as document
import veusz.setting as setting
import veusz.dataimport
import veusz.document.svg_export as svg_export

# required to get structures initialised
import veusz.windows.mainwindow

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

        # fails on suse / fedora
        'contour_labels.vsz',
        # new arm self test failures
        'example_import.vsz',
        'profile.vsz',
        '1dto2d.vsz',

        # don't expect this to work
        'mathml.vsz',

        # 3d rendering needs more work
        '3d_errors.vsz',
        '3d_function.vsz',
        '3d_points.vsz',
        '3d_surface.vsz',
        '3d_volume.vsz',
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
        return qt.QRectF(0, 0, self.height()*0.5, self.height())

    def boundingRectChar(self, c):
        return qt.QRectF(0, 0, self.height()*0.5, self.height())

    def lineSpacing(self):
        return 0.1*self.height()

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

def renderVszTest(invsz, outfile, test_saves=False, test_unlink=False):
    """Render vsz document to create outfile."""

    doc = document.Document()
    mode = 'hdf5' if os.path.splitext(invsz)[1] == '.vszh5' else 'vsz'
    doc.load(invsz, mode=mode)

    if test_unlink:
        for d in doc.data:
            doc.data[d].linked = None

    if test_saves and h5py is not None:
        tempfilename = 'self-test-temporary.vszh5'
        doc.save(tempfilename, mode='hdf5')
        doc = document.Document()
        doc.load(tempfilename, mode='hdf5')
        os.unlink(tempfilename)

    if test_saves:
        tempfilename = 'self-test-temporary.vsz'
        doc.save(tempfilename, mode='vsz')
        doc = document.Document()
        doc.load(tempfilename, mode='vsz')
        os.unlink(tempfilename)

    ifc = document.CommandInterface(doc)
    ifc.Export(outfile)

def renderPyTest(inpy, outfile):
    """Render py embedded script to create outfile."""
    retn = subprocess.call([sys.executable, inpy, outfile])
    return retn == 0

class Dirs(object):
    """Directories and files object."""
    def __init__(self):
        self.thisdir = os.path.dirname(__file__)
        self.exampledir = os.path.join(self.thisdir, '..', 'examples')
        self.testdir = os.path.join(self.thisdir, 'selftests')
        self.comparisondir = os.path.join(self.thisdir, 'comparison')

        self.infiles = (
            glob.glob( os.path.join(self.exampledir, '*.vsz') ) +
            glob.glob( os.path.join(self.testdir, '*.vsz') ) +
            glob.glob( os.path.join(self.testdir, '*.vszh5') ) )
        self.infiles += glob.glob(os.path.join(self.testdir, '*.py'))

def renderAllTests():
    """Check documents produce same output as in comparison directory."""

    print("Regenerating all test output")

    d = Dirs()
    for infile in d.infiles:
        base = os.path.basename(infile)
        print(base)
        outfile = os.path.join(d.comparisondir, base + '.selftest')
        ext = os.path.splitext(base)[1]
        if ext == '.vsz' or ext == '.vszh5':
            renderVszTest(infile, outfile)
        elif ext == '.py':
            renderPyTest(infile, outfile)

def runTests(test_saves=False, test_unlink=False):
    print("Testing output")

    fails = 0
    passes = 0
    skipped_support = 0
    skipped_wip = 0

    d = Dirs()
    for infile in sorted(d.infiles):
        base = os.path.basename(infile)
        print(base)

        ext = os.path.splitext(infile)[1]

        if ( (base[:5] == 'hdf5_' and h5py is None) or
             (base[:5] == 'fits_' and pyfits is None) or
             (ext == '.vszh5' and h5py is None) ):
            print(" SKIPPED: missing support module")
            skipped_support += 1
            continue

        outfile = os.path.join(d.thisdir, base + '.temp.selftest')

        if ext == '.vsz' or ext == '.vszh5':
            renderVszTest(infile, outfile, test_saves=test_saves,
                          test_unlink=test_unlink)
        elif ext == '.py':
            if not renderPyTest(infile, outfile):
                print(" FAIL: did not execute cleanly")
                fails += 1
                continue
        else:
            raise RuntimeError('Invalid input file')

        if base in excluded_tests:
            print(" SKIPWIP: rendered, but comparison skipped")
            skipped_wip += 1
            os.unlink(outfile)
            continue

        comparfile = os.path.join(d.thisdir, 'comparison', base + '.selftest')
        with copenuniversal(outfile) as f1:
            with copenuniversal(comparfile) as f2:
                comp = f1.read() == f2.read()

        if not comp:
            print(" FAIL: results differed")
            fails += 1
        else:
            print(" PASS")
            passes += 1
            os.unlink(outfile)

    print()
    if skipped_support != 0:
        print('Skipped %i tests (missing support)' % skipped_support)
    if skipped_wip != 0:
        print('Skipped %i comparisons (work in progress)' % skipped_wip)
    if fails == 0:
        print("All tests %i/%i PASSED" % (passes, passes))
        sys.exit(0)
    else:
        print("%i/%i tests FAILED" % (fails, passes+fails))
        sys.exit(fails)

oldflt = svg_export.fltStr
def fltStr(v, prec=1):
    """Only output floats to 1 dp."""
    return oldflt(v, prec=prec)

if __name__ == '__main__':
    app = qt.QApplication([])

    setting.transient_settings['unsafe_mode'] = True

    # hack metrics object to always return same metrics
    # and replace text renderer with one that encodes unicode symbols
    utils.textrender.FontMetrics = StupidFontMetrics
    utils.FontMetrics = StupidFontMetrics
    utils.textrender.PartText = PartTextAscii

    # nasty hack to remove underlining
    del utils.textrender.part_commands[r'\underline']

    # dpi (use old values)
    svg_export.fltStr = fltStr

    parser = optparse.OptionParser()
    parser.add_option("", "--test-saves", action="store_true",
                      help="tests saving documents and reloading them")
    parser.add_option("", "--test-unlink", action="store_true",
                      help="unlinks data from files before --test-saves")

    options, args = parser.parse_args()
    if len(args) == 0:
        runTests(test_saves=options.test_saves,
                 test_unlink=options.test_unlink)
    elif args == ['regenerate']:
        renderAllTests()
    else:
        parser.error("argument must be empty or 'regenerate'")
