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
import veusz.setting as setting
import field

# add an instance of your class to this list to be registered
toolspluginregistry = []

class ToolsPluginException(RuntimeError):
    """Raise this to report an error doing what was requested.
    """
    pass

class ToolsPlugin(object):
    # the plugin will get inserted into the menu in a hierarchy based on
    # the elements of this tuple
    menu = ('Base plugin',)
    name = 'Base plugin'

    author = ''
    description_short = ''
    description_full = ''

    # if the plugin takes no parameters, set this to False
    has_parameters = True

    def __init__(self):
        """Override this to declare a list of input fields if required."""
        self.fields = []

    def apply(self, commandinterface, fieldresults):
        """Override this option to do the work of the plugin.

        * commandinterface is an instance of the embedding interface,
        which also contains the Root widget node object
        * fieldresults is a dict containing the values of the fields plus
        'currentwidget' which is the path to the current widget

        * Raise an ToolsPluginException(str) to report a problem to the user
        """

#########################################################################

class ColorsRandomize(ToolsPlugin):
    """Randomize the colors used in plotting."""

    menu = ('Colors', 'Randomize')
    name = 'Randomize colors'
    description_short = 'Randomize the colors used in plotting'
    description_full = 'Randomize the colors used in plotting markers, lines or error bars. Random colors in hue, saturation and luminosity (HSV) are chosen between the two colors given.'

    def __init__(self):
        """Construct plugin."""
        self.fields = [
            field.FieldWidget("widget", descr="Start from widget",
                              default="/"),
            field.FieldBool("randxy", descr="Randomize xy plotters",
                            default=True),
            field.FieldBool("randfunc", descr="Randomize function plotters",
                            default=True),
            field.FieldColor('color1', descr="Start of color range",
                             default='#404040'),
            field.FieldColor('color2', descr="End of color range",
                             default='#ff0004'),
            ]

    def getRandomColor(self, col1, col2):
        """Return RGB name for a random color."""

        H1, H2 = col1.hue(), col2.hue()
        S1, S2 = col1.saturation(), col2.saturation()
        V1, V2 = col1.value(), col2.value()

        def rand(a, b):
            if a > b:
                return random.randint(b, a)
            return random.randint(a, b)
            
        col = qt4.QColor.fromHsv(rand(H1, H2), rand(S1, S2), rand(V1, V2))
        return str(col.name())

    def apply(self, ifc, fields):
        """Do the randomizing."""

        fromwidget = ifc.Root.fromPath(fields['widget'])

        col1 = qt4.QColor(fields['color1'])
        col2 = qt4.QColor(fields['color2'])

        if fields['randxy']:
            for node in fromwidget.WalkWidgets(widgettype='xy'):
                col = self.getRandomColor(col1, col2)
                node.PlotLine.color.val = col
                node.MarkerFill.color.val = col
                node.ErrorBarLine.color.val = col

        if fields['randfunc']:
            for node in fromwidget.WalkWidgets(widgettype='function'):
                node.Line.color.val = self.getRandomColor(col1, col2)

toolspluginregistry.append(ColorsRandomize)

class ColorsReplace(ToolsPlugin):
    """Randomize the colors used in plotting."""

    menu = ('Colors', 'Replace')
    name = 'Replace colors'
    description_short = 'Search and replace colors'
    description_full = 'Searches for a color and replaces it with a different color'

    def __init__(self):
        """Construct plugin."""
        self.fields = [
            field.FieldWidget("widget", descr="Start from widget",
                              default="/"),
            field.FieldBool("follow", descr="Change references and defaults",
                            default=True),
            field.FieldColor('color1', descr="Color to change",
                             default='black'),
            field.FieldColor('color2', descr="Replacement color",
                             default='red'),
            ]

    def apply(self, ifc, fields):
        """Do the color search and replace."""

        fromcol = qt4.QColor(fields['color1'])

        def walkNodes(node):
            """Walk nodes, changing values."""
            if node.type == 'setting' and node.settingtype == 'color':
                # only follow references if requested
                if node.isreference:
                    if fields['follow']:
                        node = node.resolveReference()
                    else:
                        return

                # evaluate into qcolor to make sure is a true match
                if qt4.QColor(node.val) == fromcol:
                    node.val = fields['color2']
            else:
                for c in node.children:
                    walkNodes(c)

        fromwidget = ifc.Root.fromPath(fields['widget'])
        walkNodes(fromwidget)

toolspluginregistry.append(ColorsReplace)

class TextReplace(ToolsPlugin):
    """Randomize the colors used in plotting."""

    menu = ('General', 'Replace text')
    name = 'Replace text'
    description_short = 'Search and replace text in settings'
    description_full = 'Searches for text in a setting and replaces it'

    def __init__(self):
        """Construct plugin."""
        self.fields = [
            field.FieldWidget("widget", descr="Start from widget",
                              default="/"),
            field.FieldBool("follow", descr="Change references and defaults",
                            default=True),
            field.FieldBool("onlystr", descr="Change only textual data",
                            default=False),
            field.FieldText('text1', descr="Text to change",
                            default=''),
            field.FieldText('text2', descr="Replacement text",
                            default=''),
            ]

    def apply(self, ifc, fields):
        """Do the search and replace."""

        def walkNodes(node):
            """Walk nodes, changing values."""
            if node.type == 'setting':
                # only follow references if requested
                if node.isreference:
                    if fields['follow']:
                        node = node.resolveReference()
                    else:
                        return

                val = node.val
                # try to change if a string, and not only strings or type is string
                if isinstance(val, basestring) and (not fields['onlystr'] or
                                                    node.settingtype == 'str'):
                    # update text if it changes
                    val2 = val.replace(fields['text1'], fields['text2'])
                    if val != val2:
                        try:
                            node.val = val2
                        except setting.InvalidType:
                            pass
            else:
                for c in node.children:
                    walkNodes(c)

        fromwidget = ifc.Root.fromPath(fields['widget'])
        walkNodes(fromwidget)

toolspluginregistry.append(TextReplace)

