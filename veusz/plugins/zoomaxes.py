#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Zoom by changing axes scales."""
from .operation_wrapper import OperationWrapper
from . import toolsplugin
from . import field
from ..utils import searchFirstOccurrence

class ZoomAxesPlugin(OperationWrapper, toolsplugin.ToolsPlugin):

    """Re-scale all axes in order to zoom in/out from the graph"""
    # a tuple of strings building up menu to place plugin on
    menu = ('General', 'Zoom axes')
    # unique name for plugin
    name = 'Zoom by axes'
    # name to appear on status tool bar
    description_short = 'Zoom by rescaling axes'
    # text to appear in dialog box
    description_full = 'Re-scale axes in order to zoom in/out from the graph.'

    def __init__(self):
        """Make list of fields."""
        toolsplugin.ToolsPlugin.__init__(self)
        self.fields = [
            field.FieldFloat(
                "zoom", descr="Zoom factor (%). \n>0: zoom in. \n<0: zoom out. \n=0: restores autoranges.", default=-10, minval=-300, maxval=300),
            field.FieldBool(
                "x", descr="Scale also horizontal (X) axes", default=False),
        ]

    def apply(self, cmd, fields):
        """Do the work of the plugin.
        cmd: veusz command line interface object (exporting commands)
        fields: dict mapping field names to values
        """
        self.ops = []
        doc = cmd.document
        self.doc = doc
        g = doc.resolveFullWidgetPath(fields['currentwidget'])
        g = searchFirstOccurrence(g, 'graph')
        if g is None:
            raise toolsplugin.ToolsPluginException(
                'You should use this tool on a graph object.')
        z = -fields['zoom']
        x = fields['x']
        # Apply zoom on all axes:
        for ax in g.children:
            if ax.typename not in ('axis', 'axis-function'):
                continue
            if not x and ax.settings.direction == 'horizontal':
                continue
            if z == 0:
                self.toset(ax, 'min', 'Auto')
                self.toset(ax, 'max', 'Auto')
                continue
            m, M = ax.plottedrange
            # Delta
            d = z * (M - m) / 200.
            self.toset(ax, 'min', float(m - d))
            self.toset(ax, 'max', float(M + d))

        self.apply_ops('ZoomAxes %.1f%%' % z)


toolsplugin.toolspluginregistry.append(ZoomAxesPlugin)
