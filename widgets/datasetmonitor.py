#    datasetmonitor.py
#    utility class to keep track of whether a widget's dataset(s) have changed

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

class DatasetMonitor(object):
    def __init__(self, document, *dssettings):
        self.document = document
        self.datachangeset = document.datachangeset
        self.changesets = self.findChangesets(dssettings)

    def findChangesets(self, dssettings):
        """Look up the changeset numbers for the datasets referenced by
        all the settings in dssettings"""
        
        doc = self.document
        changesets = dict()
        
        for s in dssettings:
            if s.isDataset(doc):
                val = s.val
                if not isinstance(val, tuple):
                    val = (val, )
                
                for name in val:
                    changesets[name] = doc.datachangesets.get(name, -1)
        
        return changesets

    def hasChanged(self, *dssettings):
        """Have any of the monitored datasets changed?"""
        changed = False

        oldchangesets = self.changesets
        self.changesets = self.findChangesets(dssettings)

        for name in self.changesets.iterkeys():
            # either the setting has changed or the dataset has changed
            if oldchangesets.get(name, -2) != self.changesets[name]:
                return True
            
            # if the dataset exists, see if it's changeset number is reliable
            try:
                if (not self.document.getData(name).isstable and 
                        self.datachangeset != self.document.datachangeset):
                    self.datachangeset = self.document.datachangeset
                    return True
            except KeyError:
                pass

        return False