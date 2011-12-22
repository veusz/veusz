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

"""DBus interface to Veusz document."""

import veusz.utils.vzdbus as vzdbus
import commandinterpreter

class DBusInterface(vzdbus.Object):
    """DBus interface to Veusz document command interface."""

    _ctr = 1
    interface = 'org.veusz.document'

    def __init__(self, doc):
        root = '/Windows/%i/Document' % DBusInterface._ctr
        vzdbus.Object.__init__(self, vzdbus.sessionbus, root)
        self.index = DBusInterface._ctr
        DBusInterface._ctr += 1
        self.cmdinter = commandinterpreter.CommandInterpreter(doc)
        self.ci = self.cmdinter.interface

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s')
    def RunPython(self, cmdstr):
        return self.cmdinter.run(cmdstr)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='sa{sv}')
    def Action(self, action, optargs):
        return self.ci.Action(action, **optargs)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='sa{sv}',
                   out_signature='s')
    def Add(self, wtype, optargs):
        return self.ci.Add(wtype, **optargs)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='sssa{sv}')
    def AddCustom(self, thetype, name, val, argsv):
        self.ci.AddCustom(thetype, name, val, **argsv)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s')
    def AddImportPath(self, dirname):
        self.ci.AddImportPath(unicode(dirname))

    @vzdbus.method(dbus_interface=interface,
                   in_signature='ssa{sv}', out_signature='s')
    def CloneWidget(self, widget, newparent, optargs):
        return self.ci.CloneWidget(unicode(widget), unicode(newparent),
                                   **optargs)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='sssa{sv}')
    def CreateHistogram(self, inexpr, outbinsds, outvalsds, optargs):
        self.ci.CreateHistogram(unicode(inexpr), unicode(outbinsds),
                                unicode(outvalsds), **optargs)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='sa{sv}a{sv}')
    def DatasetPlugin(self, pluginname, fields, datasetnames):
        self.ci.DatasetPlugin(unicode(pluginname), fields, datasetnames)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='sa{sv}')
    def Export(self, filename, optargs):
        self.ci.Export(filename, **optargs)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s', out_signature='v')
    def Get(self, val):
        return self.ci.Get(val)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s', out_signature='as')
    def GetChildren(self, where):
        return self.ci.GetChildren(where=where)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s', out_signature='adadadad')
    def GetData1D(self, datasetname):
        """Get a numeric dataset. Returns lists of numeric values
        for data, symmetric error, negative error and positive error."""
        def lornull(l):
            """Get blank list if None or convert to list otherwise."""
            if l is None: return []
            return list(l)
        data, serr, nerr, perr = self.ci.GetData(unicode(datasetname))
        return lornull(data), lornull(serr), lornull(nerr), lornull(perr)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s', out_signature='iiddddad')
    def GetData2D(self, datasetname):
        """Get a 2D dataset. Returns
        (X dim, Y dim, rangex min, rangex max,
         rangey min, rangey max,
         data (as 1d numeric array))
        """
        data = self.ci.GetData(unicode(datasetname))
        return ( data[0].shape[1], data[0].shape[0],
                 data[1][0], data[1][1], data[2][0], data[2][1],
                 list(data[0].flat) )

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s', out_signature='as')
    def GetDataText(self, datasetname):
        """Get a text dataset as an array of strings."""
        return self.ci.GetData(unicode(datasetname))

    @vzdbus.method(dbus_interface=interface,
                   out_signature='as')
    def GetDatasets(self):
        return self.ci.GetDatasets()

    @vzdbus.method(dbus_interface=interface,
                   in_signature='ssa{sv}')
    def ImportFile(self, filename, descriptor, optargs):
        self.ci.ImportFile(filename, descriptor, **optargs)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='sasa{sv}')
    def ImportFile2D(self, filename, datasetnames, optargs):
        self.ci.ImportFile2D(filename, datasetnames, **optargs)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='sa{sv}')
    def ImportFileCSV(self, filename, optargs):
        self.ci.ImportFileCSV(filename, **optargs)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='sssa{sv}')
    def ImportFITSFile(self, dsname, filename, hdu, optargs):
        self.ci.ImportFITSFile(filename, **optargs)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='ssa{sv}')
    def ImportFileCSV(self, plugin, filename, optargs):
        self.ci.ImportFilePlugin(plugin, filename, **optargs)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='ssa{sv}')
    def ImportString(self, descriptor, string, optargs):
        self.ci.ImportString(unicode(descriptor), unicode(string),
                             **optargs)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s')
    def Load(self, filename):
        self.cmdinter.Load(filename)

    @vzdbus.method(dbus_interface=interface)
    def Print(self):
        self.ci.Print()

    @vzdbus.method(dbus_interface=interface)
    def ReloadData(self):
        self.ci.ReloadData()

    @vzdbus.method(dbus_interface=interface,
                   in_signature='ss')
    def Rename(self, widget, newname):
        self.ci.Rename( unicode(widget), unicode(newname) )

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s')
    def Remove(self, name):
        self.ci.Remove(unicode(name))

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s')
    def RemoveCustom(self, name):
        self.ci.RemoveCustom(unicode(name))

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s', out_signature='s')
    def ResolveReference(self, name):
        return self.ci.ResolveReference(unicode(name))

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s')
    def Save(self, filename):
        self.ci.Save(unicode(filename))

    @vzdbus.method(dbus_interface=interface,
                   in_signature='sv')
    def Set(self, name, val):
        return self.ci.Set(unicode(name), val)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='ss')
    def SetToReference(self, name, val):
        return self.ci.SetToReference(unicode(name), unicode(val))

    @vzdbus.method(dbus_interface=interface,
                   in_signature='sadadadad')
    def SetData(self, name, data, symerr, negerr, poserr):
        if not symerr: symerr = None
        if not negerr: negerr = None
        if not poserr: poserr = None
        self.ci.SetData(unicode(name), data, symerr, negerr, poserr)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='ssssa{sv}')
    def SetData2DExpressionXYZ(self, name, xexpr, yexpr, zexpr, optargs):
        self.ci.SetData2DExpressionXYZ(unicode(name), xexpr, yexpr, zexpr,
                                       **optargs)

    @vzdbus.method(dbus_interface=interface,
                           in_signature='s(ddd)(ddd)sa{sv}')
    def SetData2DXYFunc(self, name, xstep, ystep, expr, optargs):
        self.ci.SetData2DXYFunc(unicode(name), xstep, ystep, expr, **optargs)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='sadii(dd)(dd)')
    def SetData2D(self, name, data, nx, ny, xrange, yrange):
        data = N.array(data).reshape(nx, ny)
        self.ci.SetData2D(unicode(name), data, xrange=xrange, yrange=yrange)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='ssa{sv}')
    def SetDataExpression(self, name, val, optargs):
        self.ci.SetDataExpression(unicode(name), val, **optargs)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='sas')
    def SetDataText(self, name, val):
        val = [unicode(x) for x in val]
        self.ci.SetDataText(unicode(name), val)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='ss')
    def SetToReference(self, var, val):
        self.ci.SetToReference(var, val)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s')
    def To(self, path):
        self.ci.To(path)

    # node interface

    @vzdbus.method(dbus_interface=interface,
                   in_signature='ss', out_signature='as')
    def NodeChildren(self, path, types):
        return self.ci.NodeChildren(path, types)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s', out_signature='s')
    def NodeType(self, path):
        return self.ci.NodeType(path)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s', out_signature='s')
    def SettingType(self, path):
        return self.ci.SettingType(path)

    @vzdbus.method(dbus_interface=interface,
                   in_signature='s', out_signature='s')
    def WidgetType(self, path):
        return self.ci.WidgetType(path)
