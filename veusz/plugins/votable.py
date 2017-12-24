#    Copyright (C) 2012 Science and Technology Facilities Council.
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

from __future__ import division, print_function

from ..compat import CStringIO, CBytesIO, curlrequest
from .importplugin import ImportPlugin, importpluginregistry
from .datasetplugin import Dataset1D, DatasetText

try:
    from astropy.io.votable.table import parse
except ImportError:
    parse = None
    print('VO table import: astropy module not available')

class ImportPluginVoTable(ImportPlugin):
    name = 'VO table import'
    author = 'Graham Bell'
    description = 'Reads datasets from VO tables'

    def _load_votable(self, params):
        if 'url' in params.field_results:
            try:
                buff = CStringIO(curlrequest.urlopen(
                        params.field_results['url']).read())
            except TypeError:
                buff = CBytesIO(curlrequest.urlopen(
                        params.field_results['url']).read())
            return parse(buff, filename=params.filename)
        else:
            return parse(params.filename)

    def doImport(self, params):
        result = []
        votable = self._load_votable(params)

        for table in votable.iter_tables():
            for field in table.fields:
                fieldname = field.name

                if field.datatype in ['float', 'double', 'short',
                                      'int', 'unsignedByte']:
                    result.append(Dataset1D(fieldname,
                                            table.array[fieldname]))

                elif field.datatype in ['char', 'string', 'unicodeChar']:
                    result.append(DatasetText(fieldname,
                                             table.array[fieldname]))

                elif field.datatype in ['floatComplex', 'doubleComplex']:
                    print(('VO table import: skipping complex field ' +
                          fieldname))

                elif field.datatype in ['boolean', 'bit']:
                    print(('VO table import: skipping boolean field ' +
                          fieldname))

                else:
                    print(('VO table import: unknown data type ' +
                          field.datatype + ' for field ' + fieldname))

        return result

    def getPreview(self, params):
        try:
            votable = self._load_votable(params)

        except:
            return ('', False)

        summary = []

        for table in votable.iter_tables():
            summary.append(table.name + ':')
            for field in table.fields:
                summary.append('    ' + field.name +
                               ' (' + field.datatype +')')

        return ('\n'.join(summary), True)

if parse is not None:
    importpluginregistry += [ImportPluginVoTable]
