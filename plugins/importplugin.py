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

"""Import plugin base class and helpers."""

import numpy as N
import veusz.utils as utils

from field import Field as ImportField
from field import FieldBool as ImportFieldCheck
from field import FieldText as ImportFieldText
from field import FieldFloat as ImportFieldFloat
from field import FieldInt as ImportFieldInt
from field import FieldCombo as ImportFieldCombo
import field

from datasetplugin import Dataset1D as ImportDataset1D
from datasetplugin import Dataset2D as ImportDataset2D
from datasetplugin import DatasetText as ImportDatasetText

# add an instance of your class to this list to get it registered
importpluginregistry = []

class ImportPluginParams(object):
    """Parameters to plugin are passed in this object."""
    def __init__(self, filename, encoding, field_results):
        self.filename = filename
        self.encoding = encoding
        self.field_results = field_results

    def openFileWithEncoding(self):
        """Helper to open filename but respecting encoding."""
        return utils.openEncoding(self.filename, self.encoding)

class ImportPluginException(RuntimeError):
    """An exception to return errors about importing or previewing data."""

class ImportPlugin(object):
    """Define a plugin to read data in a particular format.
    
    override doImport and optionally getPreview to define a new plugin
    register the class by adding to the importpluginregistry list
    if promote_tab is set to some text, put the plugin on its own tab
     in the import dialog using that text as the tab name
    """

    name = 'Import plugin'
    author = ''
    description = ''

    # if set to some text, use this plugin on its own tab
    promote_tab = None

    # set these to get focus if a file is selected with these extensions
    # include the dot in the extension names
    file_extensions = set()

    def __init__(self):
        """Override this to declare a list of input fields if required."""
        # a list of ImportField objects to display
        self.fields = []

    def getPreview(self, params):
        """Get data to show in a text box to show a preview.
        params is a ImportPluginParams object.
        Returns (text, okaytoimport)
        """
        f = params.openFileWithEncoding()
        return f.read(4096), True

    def doImport(self, params):
        """Actually import data
        params is a ImportPluginParams object.
        Return a list of ImportDataset1D, ImportDataset2D objects
        """
        return []

#################################################################

class ImportPluginExample(ImportPlugin):
    """An example plugin for reading a set of unformatted numbers
    from a file."""

    name = "Example plugin"
    author = "Jeremy Sanders"
    description = "Reads a list of numbers in a text file"

    def __init__(self):
        self.fields = [
            ImportFieldText("name", descr="Dataset name", default="name"),
            ImportFieldCheck("invert", descr="invert values"),
            ImportFieldFloat("mult", descr="Multiplication factor", default=1),
            ImportFieldInt("skip", descr="Skip N lines",
                           default=0, minval=0),
            ImportFieldCombo("subtract", items=("0", "1", "2"),
                             editable=False, default="0")
            ]

    def doImport(self, params):
        """Actually import data
        params is a ImportPluginParams object.
        Return a list of ImportDataset1D, ImportDataset2D objects
        """
        f = params.openFileWithEncoding()
        data = []
        mult = params.field_results["mult"]
        sub = float(params.field_results["subtract"])
        if params.field_results["invert"]:
            mult *= -1
        for i in xrange(params.field_results["skip"]):
            f.readline()
        for line in f:
            data += [float(x)*mult-sub for x in line.split()]

        return [ImportDataset1D(params.field_results["name"], data)]

class QdpFile(object):
    """Handle reading of a Qdp file."""

    def __init__(self, colnames):
        self.colmodes = {}
        self.skipmode = 'none'
        self.retndata = []
        self.data = []
        self.dataindex = 1
        self.colnames = colnames

    def handleRead(self, p):
        """Handle read command."""
        try:
            mode = {'t': 'terr', 's': 'serr'}[p[1][:1]]
        except (IndexError, KeyError):
            raise ImportPluginException("read command takes terr/serr")
        try:
            cols = [int(x) for x in p[2:]]
        except ValueError:
            raise ImportPluginException("read command takes list of columns separated by spaces")
        for c in cols:
            self.colmodes[c] = mode

    def handleSkip(self, p):
        """Handle skip command."""
        try:
            self.skipmode = {'o': 'off', 's': 'single', 'd': 'double'}[p[1][:1]]
        except (IndexError, KeyError):
            raise ImportPluginException("skip command takes single/double/off")

    def handleNO(self, p, lastp):
        """Handle no command, meaning no data."""
        if self.skipmode == 'none':
            self.addNans( len(p) )
        elif self.skipmode == 'single':
            self.pushData()
            self.dataindex += 1
        elif self.skipmode == 'double':
            if lastp[0] == 'no':
                self.pushData()
                self.dataindex += 1
            else:
                self.addNans( len(p) )

    def addNans(self, num):
        """Add a blank set of data to output."""
        col = 0
        ds = 0
        while col < num or ds < len(self.data):
            if ds >= len(self.data):
                self.data.append([])
            m = self.colmodes.get(ds+1)
            if m == 'serr':
                self.data[ds].append( (N.nan, N.nan) )
                col += 2
            elif m == 'terr':
                self.data[ds].append( (N.nan, N.nan, N.nan) )
                col += 3
            else:
                self.data[ds].append( N.nan )
                col += 1
            ds += 1

    def pushData(self):
        """Add data to output array."""

        for i in xrange(len(self.data)):
            # get dataset name
            if i < len(self.colnames):
                name = self.colnames[i]
            else:
                name = 'vec%i' % (i+1)
            if self.skipmode == 'single' or self.skipmode == 'double':
                name = name + '_' + str(self.dataindex)

            # convert data
            a = N.array(self.data[i])
            if len(a.shape) == 1:
                # no error bars
                ds = ImportDataset1D(name, data=a)
            elif a.shape[1] == 2:
                # serr
                ds = ImportDataset1D(name, data=a[:,0], serr=a[:,1])
            elif a.shape[1] == 3:
                # perr/nerr
                p = N.where(a[:,1] < a[:,2], a[:,2], a[:,1])
                n = N.where(a[:,1] < a[:,2], a[:,1], a[:,2])

                ds = ImportDataset1D(name, data=a[:,0], perr=p, nerr=n)
            else:
                raise RuntimeError

            self.retndata.append(ds)
        self.data = []

    def handleNum(self, p):
        """Handle set of numbers."""
        
        try:
            nums = [float(x) for x in p]
        except ValueError:
            raise ImportPluginException("Cannot convert '%s' to numbers" %
                                        (' '.join(p)))
        col = 0
        ds = 0
        while col < len(nums):
            if ds >= len(self.data):
                self.data.append([])
            m = self.colmodes.get(ds+1)
            if m == 'serr':
                self.data[ds].append( (nums[col], nums[col+1]) )
                col += 2
            elif m == 'terr':
                self.data[ds].append( (nums[col], nums[col+1], nums[col+2]) )
                col += 3
            else:
                self.data[ds].append( nums[col] )
                col += 1

            ds += 1

    def importFile(self, fileobj):
        """Read data from file object."""

        contline = None
        lastp = []
        for line in fileobj:
            # strip comments
            if line.find("!") >= 0:
                line = line[:line.find("!")]
            if line[:1] == '@':
                # read another file - we don't do this
                continue
            
            p = [x.lower() for x in line.split()]

            if contline:
                # add on previous continuation if existed
                p = contline + p
                contline = None

            if len(p) > 0 and p[-1][-1] == '-':
                # continuation
                p[-1] = p[-1][:-1]
                contline = p
                continue

            if len(p) == 0:
                # nothing
                continue

            if p[0] == 'read':
                self.handleRead(p)
            elif p[0][:2] == 'sk':
                self.handleSkip(p)
            elif p[0] == 'no':
                self.handleNO(p, lastp)
            elif p[0][0] in '0123456789-.':
                self.handleNum(p)
            else:
                # skip everything else (for now)
                pass

            lastp = p

        self.pushData()

class ImportPluginQdp(ImportPlugin):
    """An example plugin for reading data from QDP files."""

    name = "QDP import"
    author = "Jeremy Sanders"
    description = "Reads datasets from QDP files"
    file_extensions = set(['.qdp'])

    def __init__(self):
        self.fields = [
            field.FieldTextMulti("names", descr="Vector name list ",
                                 default=['']),
            ]

    def doImport(self, params):
        """Actually import data
        params is a ImportPluginParams object.
        Return a list of ImportDataset1D, ImportDataset2D objects
        """
        names = [x.strip() for x in params.field_results["names"]
                 if x.strip()]

        f = params.openFileWithEncoding()
        rqdp = QdpFile(names)
        rqdp.importFile(f)
        f.close()

        return rqdp.retndata

def cnvtImportNumpyArray(name, val, errorsin2d=True):
    """Convert a numpy array to plugin returns."""

    try:
        val.shape
    except AttributeError:
        raise ImportPluginException("Not the correct format file")
    try:
        val + 0.
        val = val.astype(N.float64)
    except TypeError:
        raise ImportPluginException("Unsupported array type")

    if val.ndim == 1:
        return ImportDataset1D(name, val)
    elif val.ndim == 2:
        if errorsin2d and val.shape[1] in (2, 3):
            # return 1d array
            if val.shape[1] == 2:
                # use as symmetric errors
                return ImportDataset1D(name, val[:,0], serr=val[:,1])
            else:
                # asymmetric errors
                # unclear on ordering here...
                return ImportDataset1D(name, val[:,0], perr=val[:,1],
                                       nerr=val[:,2])
        else:
            return ImportDataset2D(name, val)
    else:
        raise ImportPluginException("Unsupported dataset shape")

class ImportPluginNpy(ImportPlugin):
    """For reading single datasets from NPY numpy saved files."""

    name = "Numpy NPY import"
    author = "Jeremy Sanders"
    description = "Reads a 1D/2D numeric dataset from a Numpy NPY file"
    file_extensions = set(['.npy'])

    def __init__(self):
        self.fields = [
            field.FieldText("name", descr="Dataset name",
                            default=''),
            field.FieldBool("errorsin2d",
                            descr="Treat 2 and 3 column 2D arrays as\n"
                            "data with error bars",
                            default=True),
            ]

    def getPreview(self, params):
        """Get data to show in a text box to show a preview.
        params is a ImportPluginParams object.
        Returns (text, okaytoimport)
        """
        try:
            retn = N.load(params.filename)
        except Exception, e:
            return "Cannot read file", False

        try:
            text = 'Array shape: %s\n' % str(retn.shape)
            text += 'Array datatype: %s (%s)\n' % (retn.dtype.str,
                                                   str(retn.dtype))
            text += str(retn)
            return text, True
        except AttributeError:
            return "Not an NPY file", False

    def doImport(self, params):
        """Actually import data.
        """

        name = params.field_results["name"].strip()
        if not name:
            raise ImportPluginException("Please provide a name for the dataset")

        try:
            retn = N.load(params.filename)
        except Exception, e:
            raise ImportPluginException("Error while reading file: %s" %
                                        unicode(e))

        return [ cnvtImportNumpyArray(
                name, retn, errorsin2d=params.field_results["errorsin2d"]) ]

class ImportPluginNpz(ImportPlugin):
    """For reading single datasets from NPY numpy saved files."""

    name = "Numpy NPZ import"
    author = "Jeremy Sanders"
    description = "Reads datasets from a Numpy NPZ file."
    file_extensions = set(['.npz'])

    def __init__(self):
        self.fields = [
            field.FieldBool("errorsin2d",
                            descr="Treat 2 and 3 column 2D arrays as\n"
                            "data with error bars",
                            default=True),
            ]

    def getPreview(self, params):
        """Get data to show in a text box to show a preview.
        params is a ImportPluginParams object.
        Returns (text, okaytoimport)
        """
        try:
            retn = N.load(params.filename)
        except Exception, e:
            return "Cannot read file", False

        # npz files should define this attribute
        try:
            retn.files
        except AttributeError:
            return "Not an NPZ file", False

        text = []
        for f in sorted(retn.files):
            a = retn[f]
            text.append('Name: %s' % f)
            text.append(' Shape: %s' % str(a.shape))
            text.append(' Datatype: %s (%s)' % (a.dtype.str, str(a.dtype)))
            text.append('')
        return '\n'.join(text), True

    def doImport(self, params):
        """Actually import data.
        """

        try:
            retn = N.load(params.filename)
        except Exception, e:
            raise ImportPluginException("Error while reading file: %s" %
                                        unicode(e))

        try:
            retn.files
        except AttributeError:
            raise ImportPluginException("File is not in NPZ format")

        # convert each of the imported arrays
        out = []
        for f in sorted(retn.files):
            out.append( cnvtImportNumpyArray(
                    f, retn[f], errorsin2d=params.field_results["errorsin2d"]) )

        return out

class ImportPluginBinary(ImportPlugin):

    name = "Binary import"
    author = "Jeremy Sanders"
    description = "Reads numerical binary files."
    file_extensions = set(['.bin'])

    def __init__(self):
        self.fields = [
            field.FieldText("name", descr="Dataset name",
                            default=""),
            field.FieldCombo("datatype", descr="Data type",
                             items = ("float32", "float64",
                                      "int8", "int16", "int32", "int64",
                                      "uint8", "uint16", "uint32", "uint64"),
                             default="float64", editable=False),
            field.FieldCombo("endian", descr="Endian (byte order)",
                             items = ("little", "big"), editable=False),
            field.FieldInt("offset", descr="Offset (bytes)", default=0, minval=0),
            field.FieldInt("length", descr="Length (values)", default=-1)
            ]

    def getNumpyDataType(self, params):
        """Convert params to numpy datatype."""
        t = N.dtype(str(params.field_results["datatype"]))
        return t.newbyteorder( {"little": "<", "big": ">"} [
                params.field_results["endian"]] )

    def getPreview(self, params):
        """Preview of data files."""
        try:
            f = open(params.filename, "rb")
            data = f.read()
            f.close()
        except IOError:
            return "Cannot read file", False

        text = ['File length: %i bytes' % len(data)]

        def filtchr(c):
            """Filtered character to ascii range."""
            if ord(c) <= 32 or ord(c) > 127:
                return '.'
            else:
                return c

        # do a hex dump (like in CP/M)
        for i in xrange(0, min(65536, len(data)), 16):
            hdr = '%04X  ' % i
            subset = data[i:i+16]
            hexdata = ('%02X '*len(subset)) % tuple([ord(x) for x in subset])
            chrdata = ''.join([filtchr(c) for c in subset])

            text.append(hdr+hexdata + '  ' + chrdata)

        return '\n'.join(text), True

    def doImport(self, params):
        """Import the data."""

        name = params.field_results["name"].strip()
        if not name:
            raise ImportPluginException("Please provide a name for the dataset")

        try:
            f = open(params.filename, "rb")
            f.seek( params.field_results["offset"] )
            retn = f.read()
            f.close()
        except IOError, e:
            raise ImportPluginException("Error while reading file: %s" %
                                        unicode(e))

        data = N.fromstring(retn, dtype=self.getNumpyDataType(params),
                            count=params.field_results["length"])
        data = data.astype(N.float64)
        return [ ImportDataset1D(name, data) ]

importpluginregistry += [
    ImportPluginNpy(),
    ImportPluginNpz(),
    ImportPluginQdp(),
    ImportPluginBinary(),
    ImportPluginExample(),
    ]
