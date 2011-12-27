#    Copyright (C) 2011 Jeremy S. Sanders
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

"""Parameters for import routines."""

class ImportParamsBase(object):
    """Import parameters for the various imports.

    Parameters:
     filename: filename to import from
     linked: whether to link to file
     encoding: encoding for file
     prefix: prefix for output dataset names
     suffix: suffix for output dataset names
     tags: list of tags to apply to output datasets
    """

    defaults = {
        'filename': None,
        'linked': False,
        'encoding': 'utf_8',
        'prefix': '',
        'suffix': '',
        'tags': None,
        }

    def __init__(self, **argsv):
        """Initialise the reader to import data from filename.
        """

        #  set defaults
        for k, v in self.defaults.iteritems():
            setattr(self, k, v)

        # set parameters
        for k, v in argsv.iteritems():
            if k not in self.defaults:
                raise ValueError, "Invalid parameter %s" % k
            setattr(self, k, v)

        # extra parameters to copy besides defaults
        self._extras = []

    def copy(self):
        """Make a copy of the parameters object."""

        newp = {}
        for k in self.defaults.keys() + self._extras:
            newp[k] = getattr(self, k)
        return self.__class__(**newp)

class ImportParamsSimple(ImportParamsBase):
    """simpleread import parameters.

    additional parameters:
     descriptor: data descriptor
     useblocks: read datasets as blocks
     datastr: text to read from instead of file
     ignoretext: whether to ignore lines of text
    """

    defaults = {
        'descriptor': '',
        'useblocks': False,
        'datastr': None,
        'ignoretext': False,
        }
    defaults.update(ImportParamsBase.defaults)

class ImportParamsCSV(ImportParamsBase):
    """CSV import parameters.

    additional parameters:
     readrows: readdata in rows
     delimiter: CSV delimiter
     textdelimiter: delimiter for text
     headerignore: number of lines to ignore after headers
     rowsignore: number of lines to ignore at top fo file
     blanksaredata: treat blank entries as nans
     numericlocale: name of local for numbers
     dateformat: date format string
     headermode: 'multi', '1st' or 'none'
    """

    defaults = {
        'readrows': False,
        'delimiter': ',',
        'textdelimiter': '"',
        'headerignore': 0,
        'rowsignore': 0,
        'blanksaredata': False,
        'numericlocale': 'en_US',
        'dateformat': 'YYYY-MM-DD|T|hh:mm:ss',
        'headermode': 'multi',
        }
    defaults.update(ImportParamsBase.defaults)

    def __init__(self, **argsv):
        ImportParamsBase.__init__(self, **argsv)
        if self.headermode not in ('multi', '1st', 'none'):
            raise ValueError, "Invalid headermode"

class ImportParams2D(ImportParamsBase):
    """2D import parameters.

    additional parameters:
     datastr: text to read from instead of file
     xrange: tuple with range of x data coordinates
     yrange: tuple with range of y data coordinates
     invertrows: invert rows when reading
     invertcols: invert columns when reading
     transpose: swap rows and columns
    """

    defaults = {
        'datasetnames': None,
        'datastr': None,
        'xrange': None,
        'yrange': None,
        'invertrows': False,
        'invertcols': False,
        'transpose': False,
        }
    defaults.update(ImportParamsBase.defaults)

class ImportParamsFITS(ImportParamsBase):
    """FITS file import parameters.

    Additional parameters:
     dsname: name of dataset
     hdu: name/number of hdu
     datacol: name of column
     symerrcol: symmetric error column
     poserrcol: positive error column
     negerrcol: negative error column
    """

    defaults = {
        'dsname': None,
        'hdu': None,
        'datacol': None,
        'symerrcol': None,
        'poserrcol': None,
        'negerrcol': None,
        }
    defaults.update(ImportParamsBase.defaults)

class ImportParamsPlugin(ImportParamsBase):
    """Parameters for import plugins.

    Additional parameter:
     plugin: name of plugin

    Plugins have their own parameters."""

    defaults = {
        'plugin': None,
        }
    defaults.update(ImportParamsBase.defaults)

    def __init__(self, **argsv):
        """Initialise plugin parameters, splitting up default parameters
        and plugin parameters."""

        pluginpars = {}
        upvars = {}
        for n, v in argsv.iteritems():
            if n in self.defaults:
                upvars[n] = v
            else:
                pluginpars[n] = v

        ImportParamsBase.__init__(self, **upvars)
        self.pluginpars = pluginpars
        self._extras.append('pluginpars')
