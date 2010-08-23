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
import re
import fnmatch

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

class WidgetsClone(ToolsPlugin):
    """Take a widget and children and clone them."""

    menu = ('Widgets', 'Clone for datasets')
    name = 'Clone widgets for datasets'
    description_short = 'Clones a widget and its children for datasets'
    description_full = 'Take a widget and its children and clone it, plotting different sets of data in each clone.\nHint: Use a "*" in the name of a replacement dataset to match multiple datasets, e.g. x_*'

    def __init__(self):
        """Construct plugin."""
        self.fields = [
            field.FieldWidget("widget", descr="Clone widget",
                              default=""),
            field.FieldDataset('ds1', descr="Dataset 1 to change",
                               default=''),
            field.FieldDatasetMulti('ds1repl',
                                    descr="Replacement(s) for dataset 1"),
            field.FieldDataset('ds2', descr="Dataset 2 to change (optional)",
                               default=''),
            field.FieldDatasetMulti('ds2repl',
                                    descr="Replacement(s) for dataset 2"),
            field.FieldBool("names", descr="Build new names from datasets",
                            default=True),
            ]

    def apply(self, ifc, fields):
        """Do the cloning."""

        def expanddatasets(dslist):
            """Expand * and ? in dataset names."""
            datasets = []
            for ds in dslist:
                if ds.find('*') == -1 and ds.find('?') == -1:
                    datasets.append(ds)
                else:
                    dlist = fnmatch.filter(ifc.GetDatasets(), ds)
                    dlist.sort()
                    datasets += dlist
            return datasets

        def chainpairs(dslist1, dslist2):
            """Return pairs of datasets, repeating if necessary."""
            if not dslist1:
                dslist1 = ['']
            if not dslist2:
                dslist2 = ['']

            end1 = end2 = False
            idx1 = idx2 = 0
            while True:
                if idx1 >= len(ds1repl):
                    idx1 = 0
                    end1 = True
                if idx2 >= len(ds2repl):
                    idx2 = 0
                    end2 = True
                if end1 and end2:
                    break

                yield dslist1[idx1], dslist2[idx2]
                idx1 += 1
                idx2 += 1

        def walkNodes(node, dsname, dsrepl):
            """Walk nodes, changing datasets."""
            if node.type == 'setting':
                if node.settingtype in (
                    'dataset', 'dataset-or-floatlist', 'dataset-or-str'):
                    # handle single datasets
                    if node.val == dsname:
                        node.val = dsrepl
                elif node.settingtype == 'dataset-multi':
                    # settings with multiple datasets
                    out = list(node.val)
                    for i, v in enumerate(out):
                        if v == dsname:
                            out[i] = dsrepl
                    if tuple(out) != node.val:
                        node.val = out
            else:
                for c in node.children:
                    walkNodes(c, dsname, dsrepl)

        # get names of replacement datasets
        ds1repl = expanddatasets(fields['ds1repl'])
        ds2repl = expanddatasets(fields['ds2repl'])

        # make copies of widget and children for each pair of datasets
        widget = ifc.Root.fromPath(fields['widget'])
        for ds1r, ds2r in chainpairs(ds1repl, ds2repl):
            # construct a name
            newname = None
            if fields['names']:
                newname = widget.name
                if ds1r: newname += ' ' + ds1r
                if ds2r: newname += ' ' + ds2r

            # make the new widget (and children)
            newwidget = widget.Clone(widget.parent, newname=newname)

            # do replacement of datasets
            if fields['ds1']:
                walkNodes(newwidget, fields['ds1'], ds1r)
            if fields['ds2']:
                walkNodes(newwidget, fields['ds2'], ds2r)

class FontSize(ToolsPlugin):
    """Increase or decrease the font size."""

    def __init__(self, dirn):
        """Construct plugin.
        dirn == 1: increase sizes
        dirn == -1: decrease sizes
        """
        self.dirn = dirn
        self.fields = [
            field.FieldWidget("widget", descr="Start from widget",
                              default="/"),
            field.FieldBool("follow", descr="Change references and defaults",
                            default=True),
            field.FieldFloat("delta", descr="Change by value",
                             default=2),
            ]

    def apply(self, ifc, fields):
        """Do the search and replace."""

        pt_re = re.compile(r'^([\d.]+)[ ]*pt$')
        delta = fields['delta']
        changed = set()

        def walkNodes(node):
            """Walk nodes, changing values."""
            if node.type == 'setting':
                if node.name == 'size':
                    # find size setting with sibling font (improve this)
                    if not hasattr(node.parent, 'font'):
                        return

                    # only follow references if requested
                    if node.isreference:
                        if fields['follow']:
                            node = node.resolveReference()
                        else:
                            return

                    # avoid doing things more than once
                    p = node.path
                    if p in changed:
                        return
                    changed.add(p)

                    # change point size if requested
                    m = pt_re.match(node.val)
                    if m:
                        pt = float(m.group(1)) + delta*self.dirn
                        if pt < 0: pt = 0.1
                        node.val = '%gpt' % pt
            else:
                for c in node.children:
                    walkNodes(c)

        fromwidget = ifc.Root.fromPath(fields['widget'])
        walkNodes(fromwidget)

class FontSizeIncrease(FontSize):
    menu = ('General', 'Increase font sizes')
    name = 'Increase font sizes'
    description_short = 'Increase font sizes'
    description_full = 'Increase font sizes by number of points given'

    def __init__(self):
        FontSize.__init__(self, 1)

class FontSizeDecrease(FontSize):
    menu = ('General', 'Decrease font sizes')
    name = 'Decrease font sizes'
    description_short = 'Decrease font sizes'
    description_full = 'Decrease font sizes by number of points given'

    def __init__(self):
        FontSize.__init__(self, -1)

toolspluginregistry += [
    ColorsRandomize,
    ColorsReplace,
    TextReplace,
    WidgetsClone,
    FontSizeIncrease,
    FontSizeDecrease,
    ]
