#    Copyright (C) 2005 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This file is part of Veusz.
#
#    Veusz is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    Veusz is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Veusz. If not, see <https://www.gnu.org/licenses/>.
#
##############################################################################

"""A database for default values of settings."""

import sys
import datetime

import numpy as N
from .. import qtall as qt

def _(text, disambiguation=None, context="Preferences"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

# default values to some settings in case the user does not have these
defaultValues = {
    # export options
    'export_DPI': 100,
    'export_DPI_PDF': 150,
    'export_DPI_SVG': 96,
    'export_color': True,
    'export_antialias': True,
    'export_quality': 85,
    'export_background': '#ffffff00',
    'export_SVG_text_as_text': False,

    # plot options
    'plot_updatepolicy': -1, # update on document changed
    'plot_antialias': True,
    'plot_numthreads': 2,

    # recent files list
    'main_recentfiles': [],

    # default stylesheet
    'stylesheet_default': '',
    # default custom definitons
    'custom_default': '',

    # colors (isdefault, 'notdefaultcolor')
    'color_scheme': 'default',
    'color_page': (True, 'white'),
    'color_error': (True, 'red'),
    'color_command': (True, 'blue'),
    'color_cntrlline': (True, 'blue'),
    'color_cntrlcorner': (True, 'black'),

    # document theme
    'colortheme_default': 'default-latest',

    # further ui options
    'toolbar_size': 24,

    # if set to true, do UI formatting in US/English
    'ui_english': False,

    # use cwd as starting directory
    'dirname_usecwd': False,

    # use document directory for export
    'dirname_export_location': 'doc',

    # export templates
    'export_template_single': '%DOCNAME%',
    'export_template_multi': '%DOCNAME%_%PAGE00%',

    # add import paths
    'docfile_addimportpaths': True,

    # ask tutorial before?
    'ask_tutorial': False,

    # log picked points to clipboard or to console
    'picker_to_clipboard': False,
    'picker_to_console': True,
    'picker_sig_figs': 5,

    # add these directories to the python path (colon-separated)
    'external_pythonpath': '',

    # location of ghostscript (or empty to search)
    'external_ghostscript': '',

    # translation file to load
    'translation_file': '',

    # user has disabled version update checks
    # (packagers: please don't disable here, see disableVersionChecks in
    #  veusz/utils/version.py)
    'vercheck_disabled': False,
    'vercheck_asked_user': False,
    'vercheck_last_done': (2000,1,1),
    'vercheck_latest': '',

    # whether to send feedback about usage
    # (packagers, please don't disable here but in veusz/setting/feedback.py)
    'feedback_disabled': False,
    'feedback_asked_user': False,

    # locations considered secure to load
    'secure_dirs': [],
    'secure_unsaved': True,
}

class _SettingDB:
    """A class which provides access to a persistant settings database.

    Items are accesses as a dict, with items as key=value
    """

    # list of colors
    colors = ('page', 'error', 'command', 'cntrlline', 'cntrlcorner')
    # default colors if isdefault is set in the setting
    color_defaults = {
        'page': 'LightBase',
        'error': 'red',
        'command': 'blue',
        'cntrlline': 'blue',
        'cntrlcorner': 'black',
    }

    def __init__(self):
        """Initialise the object, reading the settings."""

        # This domain name is fictional!
        self.domain = 'veusz.org'
        self.product = 'veusz'
        self.database = {}
        self.sepchars = "%%%"

        # read settings using QSettings
        self.readSettings()

    def color(self, name):
        """Get a color setting as a QColor."""

        val = self.database['color_' + name]
        if val[0]:
            default = self.color_defaults[name]
            if default == 'LightBase':
                base = qt.QGuiApplication.palette().color(
                    qt.QPalette.ColorRole.Base)
                if base.value() < 127:
                    base = qt.QColor(qt.Qt.GlobalColor.white)
                return base

            return qt.QColor(default)
        else:
            return qt.QColor(val[1])

    def readSettings(self):
        """Read the settings using QSettings.

        Entries have / replaced with set of characters self.sepchars
        This is because it greatly simplifies the logic as QSettings
        has special meaning for /

        The only issues are that the key may be larger than 255 characters
        We should probably check for this
        """

        s = qt.QSettings(self.domain, self.product)

        for key in s.childKeys():
            val = s.value(key)
            realkey = key.replace(self.sepchars, '/')

            try:
                self.database[realkey] = eval(val)
            except:
                print(
                    'Error interpreting item "%s" in '
                    'settings file' % realkey, file=sys.stderr)

        # set any defaults which haven't been set
        for key in defaultValues:
            if key not in self.database:
                self.database[key] = defaultValues[key]

        # keep install date for reminders, etc
        if 'install_date' not in self.database:
            today = datetime.date.today()
            self.database['install_date'] = (today.year, today.month, today.day)

    def writeSettings(self):
        """Write the settings using QSettings.

        This is called by the mainwindow on close
        """

        s = qt.QSettings(self.domain, self.product)

        # write each entry, keeping track of which ones haven't been written
        cleankeys = []
        for key in self.database:
            cleankey = key.replace('/', self.sepchars)
            cleankeys.append(cleankey)

            s.setValue(cleankey, repr(self.database[key]))

        # now remove all the values which have been removed
        for key in list(s.childKeys()):
            if key not in cleankeys:
                s.remove(key)

    def get(self, key, defaultval=None):
        """Return key if it is in database, else defaultval."""
        return self.database.get(key, defaultval)

    def __getitem__(self, key):
        """Get the item from the database."""
        return self.database[key]

    def __setitem__(self, key, value):
        """Set the value in the database."""
        self.database[key] = value

    def __delitem__(self, key):
        """Remove the key from the database."""
        del self.database[key]

    def __contains__(self, key):
        """Is the key in the database."""
        return key in self.database

# create the SettingDB singleton
settingdb = _SettingDB()

# a normal dict for non-persistent settings
transient_settings = {
    # disable safety checks on evaluated code
    'unsafe_mode': False,
}

def updateUILocale():
    """Update locale to one given in preferences."""
    global uilocale

    if settingdb['ui_english']:
        uilocale = qt.QLocale.c()
    else:
        uilocale = qt.QLocale.system()
    uilocale.setNumberOptions(qt.QLocale.NumberOption.OmitGroupSeparator)

    qt.QLocale.setDefault(uilocale)

def ui_floattostring(f, maxdp=14):
    """Convert float to string with more precision."""
    if not N.isfinite(f):
        if N.isnan(f):
            return 'nan'
        if f < 0:
            return '-inf'
        return 'inf'
    elif 1e-4 <= abs(f) <= 1e5 or f == 0:
        s = ('%.'+str(maxdp)+'g') % f
        # strip excess zeros to right
        if s.find('.') >= 0:
            s = s.rstrip('0').rstrip('.')
    else:
        s = ('%.'+str(maxdp)+'e') % f
        # split into mantissa/exponent and strip extra zeros, etc
        mant, expon = s.split('e')
        mant = mant.rstrip('0').rstrip('.')
        expon = int(expon)
        s = '%se%i' % (mant, expon)
    # make decimal point correct for local
    s = s.replace('.', uilocale.decimalPoint())
    return s

def ui_stringtofloat(s):
    """Convert string to float, allowing for decimal point in different
    locale."""
    s = s.replace(uilocale.decimalPoint(), '.')
    return float(s)

updateUILocale()
