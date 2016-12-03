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

"""Plugins for general operations."""

from __future__ import division
import random
import re
import fnmatch

import numpy as N

from ..compat import cbasestr
from .. import qtall as qt4
from .. import utils
from . import field

def _(text, disambiguation=None, context='ToolsPlugin'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

# add an instance of your class to this list to be registered
toolspluginregistry = []

class ToolsPluginException(RuntimeError):
    """Raise this to report an error doing what was requested.
    """
    pass

class ToolsPlugin(object):
    # the plugin will get inserted into the menu in a hierarchy based on
    # the elements of this tuple
    menu = (_('Base plugin'),)
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

    menu = (_('Colors'), _('Randomize'))
    name = 'Randomize colors'
    description_short = _('Randomize the colors used in plotting')
    description_full = _('Randomize the colors used in plotting markers, lines or error bars. Random colors in hue, saturation and luminosity (HSV) are chosen between the two colors given.')

    def __init__(self):
        """Construct plugin."""
        self.fields = [
            field.FieldWidget("widget", descr=_("Start from widget"),
                              default="/"),
            field.FieldBool("randxy", descr=_("Randomize xy plotters"),
                            default=True),
            field.FieldBool("randfunc", descr=_("Randomize function plotters"),
                            default=True),
            field.FieldColor('color1', descr=_("Start of color range"),
                             default='#404040'),
            field.FieldColor('color2', descr=_("End of color range"),
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
                node.PlotLine.color = col

        if fields['randfunc']:
            for node in fromwidget.WalkWidgets(widgettype='function'):
                node.Line.color.val = self.getRandomColor(col1, col2)

class ColorsSequenceCMap(ToolsPlugin):
    """Color sequence using colormap."""

    menu = (_('Colors'), _('Colormap Sequence'))
    name = 'Create color sequence using color map'
    description_short = _('Make widgets use sequence of colors in a colormap')
    description_full = _(
        'Give new colors to each widget in a sequence using a colormap. '
        'Match can be set to match names of widgets (e.g. "xy*").')

    def __init__(self):
        """Construct plugin."""
        self.fields = [
            field.FieldWidget(
                "widget", descr=_("Start from widget"), default="/"),
            field.FieldBool(
                "colorxy", descr=_("Color xy plotters"), default=True),
            field.FieldBool(
                "colorfunc", descr=_("Color function plotters"), default=True),
            field.FieldText(
                "match", descr=_("Match")),
            field.FieldBool(
                "invert", descr=_("Invert colormap"), default=False),
            field.FieldBool(
                "randomize", descr=_("Randomize order"), default=False),
            field.FieldColormap(
                "colormap", descr=_("Colormap"), default="grey"),
            ]

    def apply(self, ifc, fields):
        """Do the randomizing."""

        fromwidget = ifc.Root.fromPath(fields['widget'])
        match = fields['match'].strip()

        widgets = []
        for w in fromwidget.WalkWidgets():
            if match and not fnmatch.fnmatch(w.name, match):
                continue
            if ( (w.widgettype=='xy' and fields['colorxy']) or
                 (w.widgettype=='function' and fields['colorfunc']) ):
                widgets.append(w)

        # get list of RGBA values
        cvals = ifc.GetColormap(
            fields["colormap"], invert=fields["invert"],
            nvals=max(1, len(widgets)))

        if fields["randomize"]:
            N.random.shuffle(cvals)

        # convert colors to #XXXX format
        for idx, widget in enumerate(widgets):
            if cvals[idx,3] == 255:
                # opaque
                col = "#%02x%02x%02x" % (
                    cvals[idx,0], cvals[idx,1], cvals[idx,2])
            else:
                # with transparency
                col = "#%02x%02x%02x%02x" % (
                    cvals[idx,0], cvals[idx,1], cvals[idx,2], cvals[idx,3])

            t = widget.widgettype
            if t == 'xy':
                widget.color.val = col
            elif t == 'function':
                widget.Line.color.val = col

class ColorsSequence(ToolsPlugin):
    """Color plotters in sequence."""

    menu = (_('Colors'), _('Sequence'))
    name = 'Create color sequence'
    description_short = _('Make widgets use sequence of colors')
    description_full = _('Give new colors to each widget in a sequence between the two colors given.')

    def __init__(self):
        """Construct plugin."""
        self.fields = [
            field.FieldWidget("widget", descr=_("Start from widget"),
                              default="/"),
            field.FieldBool("randxy", descr=_("Color xy plotters"),
                            default=True),
            field.FieldBool("randfunc", descr=_("Color function plotters"),
                            default=True),
            field.FieldColor('color1', descr=_("Start of color range"),
                             default='#ff0000'),
            field.FieldColor('color2', descr=_("End of color range"),
                             default='#4000ff'),
            ]

    def apply(self, ifc, fields):
        """Do the sequence."""

        fromwidget = ifc.Root.fromPath(fields['widget'])

        col1 = qt4.QColor(fields['color1'])
        col2 = qt4.QColor(fields['color2'])
        H1, H2 = col1.hue(), col2.hue()
        S1, S2 = col1.saturation(), col2.saturation()
        V1, V2 = col1.value(), col2.value()

        # add up total number of widgets
        numwidgets = (
            len( list(fromwidget.WalkWidgets(widgettype='xy')) ) +
            len( list(fromwidget.WalkWidgets(widgettype='function')) ) )

        def colatidx(i):
            """Get color in range 0...numwidgets-1."""
            div = max(numwidgets-1, 1)

            H = i * (H2-H1) / div + H1
            S = i * (S2-S1) / div + S1
            V = i * (V2-V1) / div + V1
            return str(qt4.QColor.fromHsv(H, S, V).name())

        idx = 0
        for node in fromwidget.WalkWidgets():
            t = node.widgettype
            if fields['randxy'] and t == 'xy':
                col = colatidx(idx)
                idx += 1
                node.PlotLine.color = col

            if fields['randfunc'] and t == 'function':
                node.Line.color.val = colatidx(idx)
                idx += 1

class ColorsReplace(ToolsPlugin):
    """Replace one color by another."""

    menu = (_('Colors'), _('Replace'))
    name = 'Replace colors'
    description_short = _('Search and replace colors')
    description_full = _('Searches for a color and replaces it with a different color')

    def __init__(self):
        """Construct plugin."""
        self.fields = [
            field.FieldWidget("widget", descr=_("Start from widget"),
                              default="/"),
            field.FieldBool("follow", descr=_("Change references and defaults"),
                            default=True),
            field.FieldColor('color1', descr=_("Color to change"),
                             default='black'),
            field.FieldColor('color2', descr=_("Replacement color"),
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

class ColorsSwap(ToolsPlugin):
    """Swap colors used in plotting."""

    menu = (_('Colors'), _('Swap'))
    name = 'Swap colors'
    description_short = _('Swap two colors')
    description_full = _('Swaps two colors in the plot')

    def __init__(self):
        """Construct plugin."""
        self.fields = [
            field.FieldWidget("widget", descr=_("Start from widget"),
                              default="/"),
            field.FieldBool("follow", descr=_("Change references and defaults"),
                            default=True),
            field.FieldColor('color1', descr=_("First color"),
                             default='black'),
            field.FieldColor('color2', descr=_("Second color"),
                             default='red'),
            ]

    def apply(self, ifc, fields):
        """Do the color search and replace."""

        col1 = qt4.QColor(fields['color1'])
        col2 = qt4.QColor(fields['color2'])

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
                if qt4.QColor(node.val) == col1:
                    node.val = fields['color2']
                elif qt4.QColor(node.val) == col2:
                    node.val = fields['color1']
            else:
                for c in node.children:
                    walkNodes(c)

        fromwidget = ifc.Root.fromPath(fields['widget'])
        walkNodes(fromwidget)

class TextReplace(ToolsPlugin):
    """Randomize the colors used in plotting."""

    menu = (_('General'), _('Replace text'))
    name = 'Replace text'
    description_short = _('Search and replace text in settings')
    description_full = _('Searches for text in a setting and replaces it')

    def __init__(self):
        """Construct plugin."""
        self.fields = [
            field.FieldWidget("widget", descr=_("Start from widget"),
                              default="/"),
            field.FieldBool("follow", descr=_("Change references and defaults"),
                            default=True),
            field.FieldBool("onlystr", descr=_("Change only textual data"),
                            default=False),
            field.FieldText('text1', descr=_("Text to change"),
                            default=''),
            field.FieldText('text2', descr=_("Replacement text"),
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
                if isinstance(val, cbasestr) and (not fields['onlystr'] or
                                                  node.settingtype == 'str'):
                    # update text if it changes
                    val2 = val.replace(fields['text1'], fields['text2'])
                    if val != val2:
                        try:
                            node.val = val2
                        except utils.InvalidType:
                            pass
            else:
                for c in node.children:
                    walkNodes(c)

        fromwidget = ifc.Root.fromPath(fields['widget'])
        walkNodes(fromwidget)

class WidgetsClone(ToolsPlugin):
    """Take a widget and children and clone them."""

    menu = (_('Widgets'), _('Clone for datasets'))
    name = 'Clone widgets for datasets'
    description_short = _('Clones a widget and its children for datasets')
    description_full = _('Take a widget and its children and clone it, plotting different sets of data in each clone.\nHint: Use a "*" in the name of a replacement dataset to match multiple datasets, e.g. x_*')

    def __init__(self):
        """Construct plugin."""
        self.fields = [
            field.FieldWidget("widget", descr=_("Clone widget"),
                              default=""),
            field.FieldDataset('ds1', descr=_("Dataset 1 to change"),
                               default=''),
            field.FieldDatasetMulti('ds1repl',
                                    descr=_("Replacement(s) for dataset 1")),
            field.FieldDataset('ds2', descr=_("Dataset 2 to change (optional)"),
                               default=''),
            field.FieldDatasetMulti('ds2repl',
                                    descr=_("Replacement(s) for dataset 2")),
            field.FieldBool("names", descr=_("Build new names from datasets"),
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
                        'dataset', 'dataset-extended',
                        'dataset-or-floatlist', 'dataset-or-str'):
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
                if ds1r:
                    # / cannot be in dataset name
                    flt1 = ds1r.replace('/', '_')
                    newname += ' ' + flt1
                if ds2r:
                    flt2 = ds2r.replace('/', '_')
                    newname += ' ' + flt2

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
            field.FieldWidget("widget", descr=_("Start from widget"),
                              default="/"),
            field.FieldBool("follow", descr=_("Change references and defaults"),
                            default=True),
            field.FieldFloat("delta", descr=_("Change by value"),
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
    menu = (_('General'), _('Increase font sizes'))
    name = 'Increase font sizes'
    description_short = _('Increase font sizes')
    description_full = _('Increase font sizes by number of points given')

    def __init__(self):
        FontSize.__init__(self, 1)

class FontSizeDecrease(FontSize):
    menu = (_('General'), _('Decrease font sizes'))
    name = 'Decrease font sizes'
    description_short = _('Decrease font sizes')
    description_full = _('Decrease font sizes by number of points given')

    def __init__(self):
        FontSize.__init__(self, -1)

toolspluginregistry += [
    ColorsRandomize,
    ColorsSequence,
    ColorsSequenceCMap,
    ColorsReplace,
    ColorsSwap,
    TextReplace,
    WidgetsClone,
    FontSizeIncrease,
    FontSizeDecrease,
    ]
