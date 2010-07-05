#    Copyright (C) 2010 Jeremy S. Sanders
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

# $Id$

"""Plugins for general operations."""

import random
import veusz.qtall as qt4
import field

# add an instance of your class to this list to be registered
workerpluginregistry = []

class WorkerPluginException(RuntimeError):
    """Raise this to report an error doing what was requested.
    """
    pass

class WorkerPlugin(object):
    # the plugin will get inserted into the menu in a hierarchy based on
    # the elements of this tuple
    menu = ('Base plugin',)
    name = 'Base plugin'

    author = ''
    description_short = ''
    description_full = ''

    def __init__(self):
        """Override this to declare a list of input fields if required."""
        self.fields = []

    def apply(self, commandinterface, fieldresults):
        """Override this option to do the work of the plugin.

        * commandinterface is an instance of the embedding interface,
        which also contains the Root widget node object
        * fieldresults is a dict containing the values of the fields plus
        'currentwidget' which is the path to the current widget

        * Raise an WorkerPluginException(str) to report a problem to the user
        """

#########################################################################

class RandomizePlotColors(WorkerPlugin):
    """Randomize the colors used in plotting."""

    menu = ('Colors', 'Randomize')
    name = 'Randomize colors'
    author = 'Jeremy Sanders'
    description_short = 'Randomize the colors used in plotting'
    description_full = 'Randomize the colors used in plotting markers, lines or error bars'

    def __init__(self):
        """Construct plugin."""
        self.fields = [
            field.FieldWidget("widget", descr="Start from widget",
                              default="/"),
            field.FieldCheck("randxy", descr="Randomize xy plotters",
                             default=True),
            field.FieldCheck("randfunc", descr="Randomize function plotters",
                             default=True),
            [ field.FieldFloat("minhue", descr="Hue range (0-1)",
                               default=0.),
              field.FieldFloat("maxhue", descr=" ",
                               default=1.) ],
            [ field.FieldFloat("minsat", descr="Saturation range (0-1)",
                               default=0.3),
              field.FieldFloat("maxsat", descr=" ",
                               default=1.) ],
            [ field.FieldFloat("minval", descr="Value range (0-1)",
                               default=0.3),
              field.FieldFloat("maxval", descr=" ",
                               default=1.) ]
            ]

    def getRandomColor(self, fields):
        """Return RGB value for a random color."""
        H = random.uniform(fields['minhue'], fields['maxhue'])
        S = random.uniform(fields['minsat'], fields['maxsat'])
        V = random.uniform(fields['minval'], fields['maxval'])
        col = qt4.QColor.fromHsvF(H, S, V)
        return str(col.name())

    def apply(self, ifc, fields):
        """Do the randomizing."""

        fromwidget = ifc.Root.fromPath(fields['widget'])

        if fields['randxy']:
            for node in fromwidget.WalkWidgets(widgettype='xy'):
                col = self.getRandomColor(fields)
                node.PlotLine.color.val = col
                node.MarkerFill.color.val = col
                node.ErrorBarLine.color.val = col

        if fields['randfunc']:
            for node in fromwidget.WalkWidgets(widgettype='function'):
                node.Line.color.val = self.getRandomColor(fields)

workerpluginregistry.append(RandomizePlotColors())

class Test(WorkerPlugin):
    """Randomize the colors used in plotting."""

    menu = ('test',)
    name = 'test'
    author = 'Jeremy Sanders'
    description_short = 'test'
    description_full = 'test'

    def apply(self, ifc, fieldresults):
        """Do the randomizing."""
        print fieldresults
        ifc.Root.width.val = '2cm'
        ifc.Root.height.val = '2cm'
        raise WorkerPluginException, "hello there"

workerpluginregistry.append(Test())
