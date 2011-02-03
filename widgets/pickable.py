#    pickable.py
#    stuff related to the Picker (aka Read Data) tool

#    Copyright (C) 2011 Benjamin K. Stuhl
#    Email: Benjamin K. Stuhl <bks24@cornell.edu>
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
###############################################################################

# $Id: $

import numpy as N

import veusz.document as document

class PickInfo:
    """Encapsulates the results of a Pick operation"""
    def __init__(self, widget=None, screenpos=None, labels=None, coords=None, index=None):
        self.widget = widget
        self.screenpos = screenpos
        self.labels = labels
        self.coords = coords
        self.index = index
        self.distance = float('inf')

    def __nonzero__(self):
        if self.widget and self.screenpos and self.labels and self.coords:
            return True
        return False
