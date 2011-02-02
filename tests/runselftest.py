import glob
import os.path
import sys

import veusz.qtall as qt4
import veusz.utils.textrender
import veusz.document as document
import veusz.setting as setting
import veusz.windows.mainwindow

excluded_tests = set([
        # the 2pi x in the axis gives different positions depending on font
        'inside.vsz', 
        # for some reason more points in polyline: clipping issue?
        'histo.vsz',
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

class AsciiRenderer(veusz.utils.textrender.Renderer):
    """Text renderer which replaces text with text with only ascii.
    Mac OS likes to split up writing characters otherwise
    """

    def __init__(self, *args, **argsv):
        # replace text with utf8 encoded version
        newargs = list(args)
        # replace text with unicode encoded
        newargs[4] = unicode(args[4]).encode('ascii', 'xmlcharrefreplace')
        veusz.utils.textrender.Renderer.__init__(self, *newargs, **argsv)

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

    exec "from numpy import *" in cmds
    ifc.AddImportPath( os.path.dirname(invsz) )
    exec open(invsz) in cmds
    ifc.Export(outfile)

def renderAllTests():
    print "Regenerating all test output"

    thisdir = os.path.dirname(__file__)
    exampledir = os.path.join(thisdir, '..', 'examples' )
    for vsz in glob.glob( os.path.join(exampledir, '*.vsz') ):
        base = os.path.basename(vsz)
        if base in excluded_tests:
            pass
            #continue
        print base

        outfile = os.path.join(thisdir, 'comparison', base + '.selftest')
        renderTest(vsz, outfile)

def runTests():
    print "Testing output"

    fails = 0
    passes = 0

    thisdir = os.path.dirname(__file__)
    exampledir = os.path.join(thisdir, '..', 'examples' )
    for vsz in glob.glob( os.path.join(exampledir, '*.vsz') ):
        base = os.path.basename(vsz)
        if base in excluded_tests:
            continue
        print base

        outfile = os.path.join(thisdir, base + '.temp.selftest')
        renderTest(vsz, outfile)

        comparfile = os.path.join(thisdir, 'comparison', base + '.selftest')

        t1 = open(outfile, 'rU').read()
        t2 = open(comparfile, 'rU').read()
        if t1 != t2:
            print " FAIL: results differed"
            fails += 1
        else:
            print " PASS"
            passes += 1
            os.unlink(outfile)

    if fails == 0:
        print "All tests %i/%i PASSED" % (passes, passes)
        sys.exit(0)
    else:
        print "%i/%i tests FAILED" % (fails, passes+fails)
        sys.exit(1)

if __name__ == '__main__':
    app = qt4.QApplication([])

    veusz.setting.transient_settings['unsafe_mode'] = True

    # hack metrics object to always return same metrics
    # and replace text renderer with one that encodes unicode symbols
    veusz.utils.textrender.FontMetrics = StupidFontMetrics
    veusz.utils.FontMetrics = StupidFontMetrics
    veusz.utils.Renderer = AsciiRenderer

    # nasty hack to remove underlining
    del veusz.utils.textrender.part_commands[r'\underline']

    if len(sys.argv) == 1:
        runTests()
    else:
        if len(sys.argv) != 2 or sys.argv[1] != 'regenerate':
            print >>sys.stderr, "Usage: %s [regenerate]" % sys.argv[0]
            sys.exit(1)
        renderAllTests()
