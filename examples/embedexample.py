#!/usr/bin/env python

"""An example embedding program. Veusz needs to be installed into
the Python path for this to work (use setup.py)

This animates a sin plot, then finishes
"""

import time
import numarray
import veusz.embed as veusz

# construct a Veusz embedded window
# many of these can be opened at any time
g = veusz.Embedded('window title')

# construct the plot
g.To( g.Add('page') )
g.To( g.Add('graph') )
g.Add('xy', marker='tiehorz', MarkerFill__color='green')

# this stops intelligent axis extending
g.Set('x/autoExtend', False)
g.Set('x/autoExtendZero', False)

# zoom out
g.Zoom(0.8)

# loop, changing the values of the x and y datasets
for i in range(10):
    x = numarray.arange(0+i/2., 7.+i/2., 0.05)
    y = numarray.sin(x)
    g.SetData('x', x)
    g.SetData('y', y)

    # wait to animate the graph
    time.sleep(2)

# let the user see the final result
print "Waiting for 10 seconds"
time.sleep(10)
print "Done!"

# close the window (this is not strictly necessary)
g.Close()
