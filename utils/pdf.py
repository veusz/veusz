#    Copyright (C) 2008 Jeremy S. Sanders
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
###############################################################################

# $Id$

import re

def scalePDFMediaBox(text, pagewidth,
                     requiredwidth, requiredheight):
    """Take the PDF file text and adjust the page size.
    pagewidth: full page width
    requiredwidth: width we want
    requiredheight: height we want
    """
                     
    m = re.search(r'^/MediaBox \[([0-9]+) ([0-9]+) ([0-9]+) ([0-9]+)\]$',
                  text, re.MULTILINE)

    box = [float(x) for x in m.groups()]
    widthfactor = box[2] / pagewidth
    newbox = '/MediaBox [%i %i %i %i]' % (
        0, 
        int(box[3]-widthfactor*requiredheight),
        int(widthfactor*requiredwidth),
        int(box[3]))

    text = text[:m.start()] + newbox + text[m.end():]
    return text

def fixupPDFIndices(text):
    """Fixup index table in PDF.

    Basically, we need to find the start of each obj in the file
    These indices are then placed in an xref table at the end
    The index to the xref table is placed after a startxref
    """

    # find occurences of obj in string
    indices = {}
    for m in re.finditer('([0-9]+) 0 obj', text):
        index = int(m.group(1))
        indices[index] = m.start(0)

    # build up xref block
    fmt = '%010i %05i n'
    xref = ['xref', '0 %i' % (len(indices)+1), '0000000000 65535 f']
    for i in xrange(len(indices)):
        xref.append( fmt % (indices[i+1], 0) )
    xref.append('trailer\n')
    xref = '\n'.join(xref)

    # replace current xref with this one
    xref_re = re.compile('^xref\n.*trailer\n', re.DOTALL | re.MULTILINE)
    text = xref_re.sub(xref, text)

    # find the xref block start
    xref_index = re.search('^xref\n', text, re.MULTILINE).start(0)

    # put the correct index to the xref after startxref
    startxref_re = re.compile('^startxref\n[0-9]+\n', re.DOTALL | re.MULTILINE)
    text = startxref_re.sub('startxref\n%i\n' % xref_index, text)
    
    return text
