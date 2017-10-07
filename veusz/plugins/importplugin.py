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

from __future__ import division
import os.path
import numpy as N

from ..compat import crange, cstr, cstrerror
from .. import utils
from .. import qtall as qt4

from . import field
from . import datasetplugin

def _(text, disambiguation=None, context='ImportPlugin'):
    """Translate text."""
    return qt4.QCoreApplication.translate(context, text, disambiguation)

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

    Override doImport and optionally getPreview to define a new plugin.
    Register the class by adding it to the importpluginregistry list.
    Of promote_tab is set to some text, put the plugin on its own tab
     in the import dialog using that text as the tab name.
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
        # a list of Field objects to display
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
        Return a list of datasetplugin.Dataset1D, datasetplugin.Dataset2D objects
        """
        return []

#################################################################

class ImportPluginExample(ImportPlugin):
    """An example plugin for reading a set of unformatted numbers
    from a file."""

    name = "Example plugin"
    author = "Jeremy Sanders"
    description = _("Reads a list of numbers in a text file")

    def __init__(self):
        self.fields = [
            field.FieldText("name", descr=_("Dataset name"), default="name"),
            field.FieldBool("invert", descr=_("invert values")),
            field.FieldFloat("mult", descr=_("Multiplication factor"), default=1),
            field.FieldInt("skip", descr=_("Skip N lines"),
                           default=0, minval=0),
            field.FieldCombo("subtract", items=("0", "1", "2"),
                             editable=False, default="0")
            ]

    def doImport(self, params):
        """Actually import data
        params is a ImportPluginParams object.
        Return a list of datasetplugin.Dataset1D, datasetplugin.Dataset2D objects
        """
        try:
            f = params.openFileWithEncoding()
            data = []
            mult = params.field_results["mult"]
            sub = float(params.field_results["subtract"])
            if params.field_results["invert"]:
                mult *= -1
            for i in crange(params.field_results["skip"]):
                f.readline()
            for line in f:
                data += [float(x)*mult-sub for x in line.split()]

            return [datasetplugin.Dataset1D(params.field_results["name"], data),
                    datasetplugin.Constant("testconst", "42"),
                    datasetplugin.Function("testfunc(x)", "testconst*x**2")]
        except IOError as e:
            raise e
        except Exception as e:
            raise ImportPluginException(cstr(e))

class ImportPluginDateTime(ImportPlugin):
    """An example plugin for reading a set of iso date-times from a
    file."""

    name = "Example plugin for date/times"
    author = "Jeremy Sanders"
    description = _("Reads a list of ISO date times in a text file")

    def __init__(self):
        self.fields = [
            field.FieldText("name", descr=_("Dataset name"), default="name"),
            ]

    def doImport(self, params):
        """Actually import data
        params is a ImportPluginParams object.
        Return a list of datasetplugin.Dataset1D, datasetplugin.Dataset2D objects
        """
        f = params.openFileWithEncoding()
        data = []
        for line in f:
            data.append( datasetplugin.DatasetDateTime.
                         dateStringToFloat(line.strip()) )
        return [ datasetplugin.DatasetDateTime(params.field_results["name"],
                                               data) ]
#importpluginregistry.append( ImportPluginDateTime )

class QdpFile(object):
    """Handle reading of a Qdp file."""

    def __init__(self, colnames):
        self.colmodes = {}
        self.skipmode = 'none'
        self.retndata = []

        # store read in data here
        self.data = []
        # index of max vector
        self.dataindex = 1
        self.colnames = colnames

        # list of data groups for 2d objects
        self.datagroup2d = []
        # axis ranges for 2d objects
        self.axis2d = [None, None]

    def handleRead(self, p):
        """Handle read command."""
        try:
            mode = {'t': 'terr', 's': 'serr'}[p[1][:1]]
        except (IndexError, KeyError):
            raise ImportPluginException(_("read command takes terr/serr"))
        try:
            cols = [int(x) for x in p[2:]]
        except ValueError:
            raise ImportPluginException(_("read command takes list of columns separated by spaces"))
        for c in cols:
            self.colmodes[c] = mode

    def handleSkip(self, p):
        """Handle skip command."""
        try:
            self.skipmode = {'o': 'off', 's': 'single', 'd': 'double'}[p[1][:1]]
        except (IndexError, KeyError):
            raise ImportPluginException(_("skip command takes single/double/off"))

    def handleNO(self, p, lastp):
        """Handle no command, meaning no data."""
        if self.skipmode == 'none':
            self.addNans( len(p) )
        elif self.skipmode == 'single':
            self.pushData()
            del self.data[:]
            self.dataindex += 1
        elif self.skipmode == 'double':
            if lastp[0] == 'no':
                self.pushData()
                del self.data[:]
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

    def pushData2D(self):
        """Handle 2D data groups."""

        for num, r1, c1, r2, c2 in self.datagroup2d:
            arr = []
            for c in crange(c1-1,c2-1+1):
                arr.append( self.data[c][r1-1:r2-1+1] )
                # make data as "used"
                self.data[c] = None
            arr = N.array(arr)
            if num-1 < len(self.colnames):
                name = self.colnames[num-1]
            else:
                name = 'vec2d%i' % num

            rangex = rangey = None
            if self.axis2d[0] is not None:
                minval, pixsize = self.axis2d[0]
                rangex = (minval - pixsize*0.5,
                          minval+(arr.shape[1]-0.5)*pixsize )
            if self.axis2d[1] is not None:
                minval, pixsize = self.axis2d[1]
                rangey = (minval - pixsize*0.5,
                          minval+(arr.shape[0]-0.5)*pixsize )

            ds = datasetplugin.Dataset2D(name, data=arr,
                                         rangex=rangex, rangey=rangey)
            self.retndata.append(ds)

    def pushData(self):
        """Add data to output array.
        """

        for i in crange(len(self.data)):
            if self.data[i] is None:
                continue

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
                ds = datasetplugin.Dataset1D(name, data=a)
            elif a.shape[1] == 2:
                # serr
                ds = datasetplugin.Dataset1D(name, data=a[:,0], serr=a[:,1])
            elif a.shape[1] == 3:
                # perr/nerr
                p = N.where(a[:,1] < a[:,2], a[:,2], a[:,1])
                n = N.where(a[:,1] < a[:,2], a[:,1], a[:,2])

                ds = datasetplugin.Dataset1D(name, data=a[:,0], perr=p, nerr=n)
            else:
                raise RuntimeError

            self.retndata.append(ds)

    def handleDataGroup(self, p):
        """Handle data groups."""

        if len(p) == 3:
            # we don't support the renaming thing
            pass
        elif len(p) == 6:
            # 2d data
            try:
                pint = [int(x) for x in p[1:]]
            except ValueError:
                raise ImportPluginException(_("invalid 2d datagroup command"))

            self.datagroup2d.append(pint)

    def handleAxis(self, p):
        """Axis command gives range of axes (used for 2d)."""

        try:
            minval, maxval = float(p[2]), float(p[3])
        except ValueError:
            raise ImportPluginException(_("invalid axis range"))
        self.axis2d[ p[0][0] == 'y' ] = (minval, maxval)

    def handleNum(self, p):
        """Handle set of numbers."""

        nums = []
        try:
            for n in p:
                if n.lower() == 'no':
                    nums.append(N.nan)
                else:
                    nums.append(float(n))
        except ValueError:
            raise ImportPluginException(_("Cannot convert '%s' to numbers") %
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

    def importFile(self, fileobj, dirname):
        """Read data from file object.
        dirname is the directory in which the file is located
        """

        contline = None
        lastp = []
        for line in fileobj:
            # strip comments
            if line.find("!") >= 0:
                line = line[:line.find("!")]
            if line[:1] == '@':
                # read another file
                fname = os.path.join(dirname, line[1:].strip())
                try:
                    newf = open(fname)
                    self.importFile(newf, dirname)
                except EnvironmentError:
                    pass
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

            v0 = p[0]
            if v0[0] in '0123456789-.':
                self.handleNum(p)
            elif v0 == 'no':
                self.handleNO(p, lastp)
            elif v0 == 'read':
                self.handleRead(p)
            elif v0[:2] == 'sk':
                self.handleSkip(p)
            elif v0[:2] == 'dg':
                self.handleDataGroup(p)
            elif v0[:1] == 'x' or v0[:2] == 'ya':
                self.handleAxis(p)
            else:
                # skip everything else (for now)
                pass

            lastp = p

class ImportPluginQdp(ImportPlugin):
    """An example plugin for reading data from QDP files."""

    name = "QDP import"
    author = "Jeremy Sanders"
    description = _("Reads datasets from QDP files")
    file_extensions = set(['.qdp'])

    def __init__(self):
        self.fields = [
            field.FieldTextMulti("names", descr=_("Vector name list "),
                                 default=['']),
            ]

    def doImport(self, params):
        """Actually import data
        params is a ImportPluginParams object.
        Return a list of datasetplugin.Dataset1D, datasetplugin.Dataset2D objects
        """
        names = [x.strip() for x in params.field_results["names"]
                 if x.strip()]

        f = params.openFileWithEncoding()
        rqdp = QdpFile(names)
        rqdp.importFile(f, os.path.dirname(params.filename))
        rqdp.pushData2D()
        rqdp.pushData()
        f.close()

        return rqdp.retndata

def cnvtImportNumpyArray(name, val, errorsin2d=True):
    """Convert a numpy array to plugin returns."""

    try:
        val.shape
    except AttributeError:
        raise ImportPluginException(_("Not the correct format file"))
    try:
        val + 0.
        val = val.astype(N.float64)
    except TypeError:
        raise ImportPluginException(_("Unsupported array type"))

    if val.ndim == 1:
        return datasetplugin.Dataset1D(name, val)
    elif val.ndim == 2:
        if errorsin2d and val.shape[1] in (2, 3):
            # return 1d array
            if val.shape[1] == 2:
                # use as symmetric errors
                return datasetplugin.Dataset1D(name, val[:,0], serr=val[:,1])
            else:
                # asymmetric errors
                # unclear on ordering here...
                return datasetplugin.Dataset1D(name, val[:,0], perr=val[:,1],
                                               nerr=val[:,2])
        else:
            return datasetplugin.Dataset2D(name, val)
    else:
        return datasetplugin.DatasetND(name, val)

class ImportPluginNpy(ImportPlugin):
    """For reading single datasets from NPY numpy saved files."""

    name = "Numpy NPY import"
    author = "Jeremy Sanders"
    description = _("Reads a 1D/2D numeric dataset from a Numpy NPY file")
    file_extensions = set(['.npy'])

    def __init__(self):
        self.fields = [
            field.FieldText("name", descr=_("Dataset name"),
                            default=''),
            field.FieldBool("errorsin2d",
                            descr=_("Treat 2 and 3 column 2D arrays as\n"
                                    "data with error bars"),
                            default=True),
            ]

    def getPreview(self, params):
        """Get data to show in a text box to show a preview.
        params is a ImportPluginParams object.
        Returns (text, okaytoimport)
        """
        try:
            retn = N.load(params.filename)
        except Exception:
            return _("Cannot read file"), False

        try:
            text = _('Array shape: %s\n') % str(retn.shape)
            text += _('Array datatype: %s (%s)\n') % (retn.dtype.str,
                                                      str(retn.dtype))
            text += str(retn)
            return text, True
        except AttributeError:
            return _("Not an NPY file"), False

    def doImport(self, params):
        """Actually import data.
        """

        name = params.field_results["name"].strip()
        if not name:
            raise ImportPluginException(_("Please provide a name for the dataset"))

        try:
            retn = N.load(params.filename)
        except IOError as e:
            raise e
        except Exception as e:
            raise ImportPluginException(_("Error while reading file: %s") %
                                        cstr(e))

        return [ cnvtImportNumpyArray(
                name, retn, errorsin2d=params.field_results["errorsin2d"]) ]

class ImportPluginNpz(ImportPlugin):
    """For reading single datasets from NPY numpy saved files."""

    name = "Numpy NPZ import"
    author = "Jeremy Sanders"
    description = _("Reads datasets from a Numpy NPZ file.")
    file_extensions = set(['.npz'])

    def __init__(self):
        self.fields = [
            field.FieldBool("errorsin2d",
                            descr=_("Treat 2 and 3 column 2D arrays as\n"
                                    "data with error bars"),
                            default=True),
            ]

    def getPreview(self, params):
        """Get data to show in a text box to show a preview.
        params is a ImportPluginParams object.
        Returns (text, okaytoimport)
        """
        try:
            retn = N.load(params.filename)
        except Exception:
            return _("Cannot read file"), False

        # npz files should define this attribute
        try:
            retn.files
        except AttributeError:
            return _("Not an NPZ file"), False

        text = []
        for f in sorted(retn.files):
            a = retn[f]
            text.append(_('Name: %s') % f)
            text.append(_(' Shape: %s') % str(a.shape))
            text.append(_(' Datatype: %s (%s)') % (a.dtype.str, str(a.dtype)))
            text.append('')
        return '\n'.join(text), True

    def doImport(self, params):
        """Actually import data.
        """

        try:
            retn = N.load(params.filename)
        except IOError as e:
            raise e
        except Exception as e:
            raise ImportPluginException(_("Error while reading file: %s") %
                                        cstr(e))

        try:
            retn.files
        except AttributeError:
            raise ImportPluginException(_("File is not in NPZ format"))

        # convert each of the imported arrays
        out = []
        for f in sorted(retn.files):
            out.append( cnvtImportNumpyArray(
                    f, retn[f], errorsin2d=params.field_results["errorsin2d"]) )

        return out

class ImportPluginBinary(ImportPlugin):

    name = "Binary import"
    author = "Jeremy Sanders"
    description = _("Reads numerical binary files.")
    file_extensions = set(['.bin'])

    def __init__(self):
        self.fields = [
            field.FieldText("name", descr=_("Dataset name"),
                            default=""),
            field.FieldCombo("datatype", descr=_("Data type"),
                             items = ("float32", "float64",
                                      "int8", "int16", "int32", "int64",
                                      "uint8", "uint16", "uint32", "uint64"),
                             default="float64", editable=False),
            field.FieldCombo("endian", descr=_("Endian (byte order)"),
                             items = ("little", "big"), editable=False),
            field.FieldInt("offset", descr=_("Offset (bytes)"), default=0, minval=0),
            field.FieldInt("length", descr=_("Length (values)"), default=-1)
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
        except EnvironmentError as e:
            return _("Cannot read file (%s)") % cstrerror(e), False

        text = [_('File length: %i bytes') % len(data)]

        def filtchr(c):
            """Filtered character to ascii range."""
            if ord(c) <= 32 or ord(c) > 127:
                return '.'
            else:
                return c

        # do a hex dump (like in CP/M)
        for i in crange(0, min(65536, len(data)), 16):
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
            raise ImportPluginException(_("Please provide a name for the dataset"))

        try:
            f = open(params.filename, "rb")
            f.seek( params.field_results["offset"] )
            retn = f.read()
            f.close()
        except EnvironmentError as e:
            raise ImportPluginException(_("Error while reading file '%s'\n\n%s") %
                                        (params.filename, cstrerror(e)))

        try:
            data = N.fromstring(retn, dtype=self.getNumpyDataType(params),
                                count=params.field_results["length"])
        except ValueError as e:
            raise ImportPluginException(_("Error converting data for file '%s'\n\n%s") %
                                        (params.filename, cstr(e)))

        data = data.astype(N.float64)
        return [ datasetplugin.Dataset1D(name, data) ]

class ImportPluginGnuplot2D(ImportPlugin):
    """A Veusz plugin for reading data in Gnuplot 2D data format from a file."""

    name = "Gnuplot 2D data import plugin"
    author = "Joerg Meyer, j.meyer@chem.leidenuniv.nl"
    description = "Reads data in Gnuplot 2D format from a text file."

    file_extensions = set(['.data','.elbow'])

    def __init__(self):
        ImportPlugin.__init__(self)
        self.fields = [
            field.FieldText(
                "name", descr="Dataset name", default="name"),
            field.FieldFloat(
                "subtract", descr="Offset to subtract", default=0.0),
            field.FieldFloat(
                "mult", descr="Multiplication factor", default=1),
            ]

    def doImport(self, params):
        """Actually import data
        params is a ImportPluginParams object.
        Return a list of ImportDataset1D, ImportDataset2D objects
        """

        sub = float(params.field_results["subtract"])
        mult = params.field_results["mult"]
        f = params.openFileWithEncoding()
        data_gp = []
        data_gp_block = []
        for line in f:
            fields = line.split()
            if not fields:
                if data_gp_block:
                    data_gp.append( data_gp_block )
                    data_gp_block = []
                else:                        # ignore initial blank lines
                    continue
            elif '#' in fields[0]:            # ignore comment lines
                continue
            else:
                if len(fields) < 3:
                    raise ImportPluginException(_("Too few columns in file"))
                try:
                    x,y,z = map(float, fields[0:3])
                except ValueError:
                    raise ImportPluginException(_("Non-numeric data in file"))
                data_gp_block.append( [x,y,(z-sub)*mult] )

        if data_gp_block:                    # append last block if necessary
            data_gp.append( data_gp_block )
            data_gp_block = []

        data = N.array(data_gp)
        S = data.shape
        data_for_sorting = data.reshape((S[0]*S[1],S[2]))
        ind = N.lexsort( [data_for_sorting[:,0], data_for_sorting[:,1]] )
        data_veusz = data_for_sorting[ind].reshape(S)[:,:,2]
        rangex = (data[:,:,0].min(),data[:,:,0].max())
        rangey = (data[:,:,1].min(),data[:,:,1].max())

        return [
            datasetplugin.Dataset2D(
                params.field_results["name"],
                data=data_veusz, rangex=rangex, rangey=rangey)
        ]

importpluginregistry += [
    ImportPluginNpy,
    ImportPluginNpz,
    ImportPluginQdp,
    ImportPluginBinary,
    ImportPluginExample,
    ImportPluginGnuplot2D,
    ]
