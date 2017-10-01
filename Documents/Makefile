############################################################################
#    Copyright (C) 2017 Jeremy S. Sanders
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
############################################################################

verfile=../VERSION
RELEASE=$(shell cat $(verfile))
mansources=$(shell find manual-source -name *.rst)

all: man manual

############################################################################
# man pages

%.1: %.pod $(verfile)
	pod2man --release=$(RELEASE) --center="Veusz" $< > $@

%.man.txt: %.1
	MANWIDTH=76 man ./$< > $@

man: man-page/veusz.1 man-page/veusz.man.txt

############################################################################
# manual

manual-pdf: manual/pdf/veusz.pdf

manual-html: manual/html/index.html

manual/pdf/veusz.pdf: $(mansources) $(verfile)
	make -C manual-source latexpdf
	mkdir -p manual/pdf
	mv manual/latex/veusz.pdf manual/pdf/
	rm -rf manual/latex/

manual/html/index.html: $(mansources) $(verfile)
	make -C manual-source html

manual: manual-html manual-pdf
	rm -rf manual/doctrees/

############################################################################

clean:
	rm -f man-page/*.1 man-page/*.man.txt
	make -C manual-source clean
