#    Copyright (C) 2016 Jeremy S. Sanders
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
###############################################################################

from ..compat import cstr, citems

from .oned import Dataset1DBase, Dataset
from .twod import Dataset2DBase, Dataset2D
from .nd import DatasetNDBase, DatasetND
from .text import DatasetText
from .date import DatasetDateTimeBase, DatasetDateTime

class _DatasetPlugin(object):
    """Shared methods for dataset plugins."""

    def __init__(self, manager, ds):
        self.pluginmanager = manager
        self.pluginds = ds

    def getPluginData(self, attr):
        self.pluginmanager.update()
        return getattr(self.pluginds, attr)

    def linkedInformation(self):
        """Return information about how this dataset was created."""

        fields = []
        for name, val in citems(self.pluginmanager.fields):
            fields.append('%s: %s' % (cstr(name), cstr(val)))

        try:
            shape = [str(x) for x in self.data.shape]
        except AttributeError:
            shape = [str(len(self.data))]
        shape = u'\u00d7'.join(shape)

        return '%s plugin dataset (fields %s), size %s' % (
            self.pluginmanager.plugin.name,
            ', '.join(fields),
            shape)

    def canUnlink(self):
        """Can relationship be unlinked?"""
        return True

    def deleteRows(self, row, numrows):
        pass

    def insertRows(self, row, numrows, rowdata):
        pass

    def saveDataRelationToText(self, fileobj, name):
        """Save plugin to file, if this is the first one."""

        # only try to save if this is the 1st dataset of this plugin
        # manager in the document, so that we don't save more than once
        docdatasets = set( self.document.data.values() )

        for ds in self.pluginmanager.veuszdatasets:
            if ds in docdatasets:
                if ds is self:
                    # is 1st dataset
                    self.pluginmanager.saveToFile(fileobj)
                return

    def saveDataDumpToText(self, fileobj, name):
        """Save data to text: not used."""

    def saveDataDumpToHDF5(self, group, name):
        """Save data to HDF5: not used."""

    @property
    def dstype(self):
        """Return type of plugin."""
        return self.pluginmanager.plugin.name

class Dataset1DPlugin(_DatasetPlugin, Dataset1DBase):
    """Return 1D dataset from a plugin."""

    def __init__(self, manager, ds):
        _DatasetPlugin.__init__(self, manager, ds)
        Dataset1DBase.__init__(self)

    def userSize(self):
        """Size of dataset."""
        return str( self.data.shape[0] )

    def __getitem__(self, key):
        """Return a dataset based on this dataset

        We override this from DatasetConcreteBase as it would return a
        DatsetExpression otherwise, not chopped sets of data.
        """
        return Dataset(**self._getItemHelper(key))

    # parent class sets these attributes, so override setattr to do nothing
    data = property( lambda self: self.getPluginData('data'),
                     lambda self, val: None )
    serr = property( lambda self: self.getPluginData('serr'),
                     lambda self, val: None )
    nerr = property( lambda self: self.getPluginData('nerr'),
                     lambda self, val: None )
    perr = property( lambda self: self.getPluginData('perr'),
                     lambda self, val: None )

class Dataset2DPlugin(_DatasetPlugin, Dataset2DBase):
    """Return 2D dataset from a plugin."""

    def __init__(self, manager, ds):
        _DatasetPlugin.__init__(self, manager, ds)
        Dataset2DBase.__init__(self)

    def __getitem__(self, key):
        return Dataset2D(self.data[key], xrange=self.xrange, yrange=self.yrange,
                         xedge=self.xedge, yedge=self.yedge,
                         xcent=self.xcent, ycent=self.ycent)

    data   = property( lambda self: self.getPluginData('data'),
                       lambda self, val: None )
    xrange = property( lambda self: self.getPluginData('rangex'),
                       lambda self, val: None )
    yrange = property( lambda self: self.getPluginData('rangey'),
                       lambda self, val: None )
    xedge  = property( lambda self: self.getPluginData('xedge'),
                       lambda self, val: None )
    yedge  = property( lambda self: self.getPluginData('yedge'),
                       lambda self, val: None )
    xcent  = property( lambda self: self.getPluginData('xcent'),
                       lambda self, val: None )
    ycent  = property( lambda self: self.getPluginData('ycent'),
                       lambda self, val: None )

class DatasetNDPlugin(_DatasetPlugin, DatasetNDBase):
    """Return N-dimensional dataset from plugin."""

    def __init__(self, manager, ds):
        _DatasetPlugin.__init__(self, manager, ds)
        DatasetNDBase.__init__(self)

    def __getitem__(self, key):
        return DatasetND(self.data[key])

    data = property( lambda self: self.getPluginData('data'),
                     lambda self, val: None )

class DatasetTextPlugin(_DatasetPlugin, DatasetText):
    """Return text dataset from a plugin."""

    def __init__(self, manager, ds):
        _DatasetPlugin.__init__(self, manager, ds)
        DatasetText.__init__(self, [])

    def __getitem__(self, key):
        return DatasetText(self.data[key])

    data = property( lambda self: self.getPluginData('data'),
                     lambda self, val: None )

class DatasetDateTimePlugin(_DatasetPlugin, DatasetDateTimeBase):
    """Return date dataset from plugin."""

    def __init__(self, manager, ds):
        _DatasetPlugin.__init__(self, manager, ds)
        DatasetDateTimeBase.__init__(self)
        self.serr = self.perr = self.nerr = None

    def __getitem__(self, key):
        return DatasetDateTime(self.data[key])

    data = property( lambda self: self.getPluginData('data'),
                     lambda self, val: None )
