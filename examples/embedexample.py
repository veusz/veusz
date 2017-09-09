#    Copyright (C) 2009 Jeremy S. Sanders
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

"""An example embedding program. Veusz needs to be installed into
the Python path for this to work (use setup.py)

This animates a sin plot, then finishes
"""

import time
import numpy
import veusz.embed as veusz

# construct a Veusz embedded window
# many of these can be opened at any time
g = veusz.Embedded('window title')
g.EnableToolbar()

# construct the plot
g.To( g.Add('page') )
g.To( g.Add('graph') )
g.Add('xy', marker='tiehorz', MarkerFill__color='green')

# this stops intelligent axis extending
g.Set('x/autoRange', 'exact')

# zoom out
g.Zoom(0.8)

# loop, changing the values of the x and y datasets
for i in range(10):
    x = numpy.arange(0+i/2., 7.+i/2., 0.05)
    y = numpy.sin(x)
    g.SetData('x', x)
    g.SetData('y', y)

    # wait to animate the graph
    time.sleep(2)

# let the user see the final result
print("Waiting for 10 seconds")
time.sleep(10)
print("Done!")

# close the window (this is not strictly necessary)
g.Close()
