# preftypes.ps
# Various kinds of classes for holding preferences of various types

#    Copyright (C) 2003 Jeremy S. Sanders
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

import qt

import pref
import points
import utils

#######################################################################
# this is the main class everything is based on here.
# override _addPrefs to define preferences

class GenericPrefType:
    """ Class holds list of preferences """

    def __init__(self, name):
        """ Deault initialisation """
        self.prefs = pref.Preferences( name, self )
        self._addPrefs()
        self.prefs.read()

    def _addPrefs(self):
        """ Add a list of preferences to the object"""
        pass

    def hasPref(self, name):
        """Does the preferences list have the preference?"""
        return self.prefs.hasPref(name)

    def getPref(self, name):
        """Get the preference."""
        return self.prefs.getPref(name)
    
    def setPref(self, name, val):
        """Set the preference."""
        self.prefs.setPref(name, val)

    def makeDefault(self):
        """ Make these values the saved preferences """
        self.prefs.write()

########################################################################

class PreferencesLine(GenericPrefType):
    """ Class holds list of line preferences """

    def _addPrefs(self):
        """ Initialise preferences """
        self.prefs.addPref( 'color', 'string', '#000000' )

        # line width in points
        self.prefs.addPref( 'width', 'double', 1. )

        self.prefs.addPref( 'style', 'int', qt.Qt.SolidLine )        
        self.prefs.addPref( 'hide', 'int', 0 )

    def notHidden(self):
        return not self.hide

    def makeQPen(self, painter):
        """ Set the QPen from the one here. """
        # calculate thickness of line
        width = int( utils.getPixelsPerPoint(painter) * self.width )
        return qt.QPen( qt.QColor(self.color), width, self.style )

class PreferencesPlotLine(PreferencesLine):
    """ Class to hold preferences for lines actually on plots."""

    def __init__(self, name):
        PreferencesLine.__init__(self, name)
        if self.color == 'auto':
            # lookup next colour
            self.color = points.getAutoColor()

    def _addPrefs(self):
        PreferencesLine._addPrefs(self)
        self.prefs.setDefault( 'color', 'auto' )

class PreferencesPoint(PreferencesLine):
    """ Class to hold preferences for plotting points."""

    def __init__(self, name):
        PreferencesLine.__init__(self, name)
        if self.prefs.color == 'auto':
            # lookup next colour
            self.prefs.color = points.getAutoColor()

        if self.prefs.symbol == 'auto':
            # lookup next symbol
            self.prefs.symbol = points.getAutoMarker()

    def _addPrefs(self):
        """ Initialise prefs."""
        PreferencesLine._addPrefs(self)
        # marker size in points
        self.prefs.addPref( 'size', 'double', 3. )
        self.prefs.addPref( 'symbol', 'text', 'auto' )
        self.prefs.setDefault( 'color', 'auto' )

######################################################################

class PreferencesBrush(GenericPrefType):
    """ Class to hold filling preferences."""

    def _addPrefs(self):
        """ Initialise preferences."""
        self.prefs.addPref( 'color', 'string', '#000000' )
        self.prefs.addPref( 'style', 'int', qt.Qt.SolidPattern )
        self.prefs.addPref( 'hide', 'int', 0 )

    def notHidden(self):
        return not self.hide

    def makeQBrush(self):
        return qt.QBrush( qt.QColor(self.color), self.style )

class PreferencesPlotFill(PreferencesBrush):
    """ Class to hold preferences for filling on plots."""

    def _addPrefs(self):
        """ Initialise prefs."""
        PreferencesBrush._addPrefs(self)
        self.prefs.setDefault( 'color', 'auto' )
        self.prefs.setDefault( 'hide', 1 )

    def __init__(self, name):
        PreferencesBrush.__init__(self, name)
        if self.color == 'auto':
            # lookup next colour
            self.color = points.getAutoColor()


#####################################################################

class PreferencesMajorTick(PreferencesLine):
    """ Class to hold list of preferences for major tick marks."""

    def _addPrefs(self):
        PreferencesLine._addPrefs(self)
        self.prefs.addPref( 'length', 'double', 6 )

class PreferencesMinorTick(PreferencesLine):
    """ Class to hold list of preferences for minor tick marks."""

    def _addPrefs(self):
        PreferencesLine._addPrefs(self)
        self.prefs.addPref( 'length', 'double', 3 )

class PreferencesGridLine(PreferencesLine):
    """ Class to hold list of preferences for grid lines.

    Grid lines aren't shown by default
    """

    def _addPrefs(self):
        PreferencesLine._addPrefs(self)
        self.prefs.setDefault( 'color', '#808080' )
        self.prefs.setDefault( 'hide', 1 )
        self.prefs.setDefault( 'style', qt.Qt.DotLine )

#####################################################################

class PreferencesText(GenericPrefType):
    """ Class holds list of text preferences """

    def _addPrefs(self):
        """ Initialise list of preferences """
        self.prefs.addPref( 'font', 'string', 'Times-Roman' )
        self.prefs.addPref( 'size', 'int', 12 )
        self.prefs.addPref( 'italic', 'int', 0 )
        self.prefs.addPref( 'bold', 'int', 0 )
        self.prefs.addPref( 'underline', 'int', 0 )
        self.prefs.addPref( 'valign', 'int', 0 )
        self.prefs.addPref( 'halign', 'int', 0 )
        self.prefs.addPref( 'color', 'string', '#000000' )
        self.prefs.addPref( 'hide', 'int', 0 )
        
    def makeQFont(self):
        """ Return a qt.QFont object corresponding to the prefs """
        weight = qt.QFont.Normal
        if self.bold: weight = qt.QFont.Bold
        f = qt.QFont(self.font, self.size,  weight, self.italic)
        if self.underline: f.setUnderline(1)
        f.setStyleHint( qt.QFont.Times, qt.QFont.PreferDevice )
        return f

    def notHidden(self):
        return not self.hide

    def makeQPen(self):
        """ Return a qt.QPen object for the font pen """
        return qt.QPen(qt.QColor(self.color))

class PreferencesAxisLabel(PreferencesText):
    """ Hold preferences for axis labels."""

    def _addPrefs(self):
        PreferencesText._addPrefs(self)
        self.prefs.addPref( 'label', 'string', '' )
        self.prefs.addPref( 'rotate', 'int', 0 )

class PreferencesTickLabel(PreferencesText):
    """ Hold the preferences for tick (or axis) label text."""

    def _addPrefs(self):
        PreferencesText._addPrefs(self)
        self.prefs.addPref( 'rotate', 'int', 0 )
        self.prefs.addPref( 'format', 'string', 'g*' )

    def formatNumber(self, num):
        """ Used the stored format preference to format a number."""
        return utils.formatNumber(num, self.format)
    
