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
###############################################################################

from itertools import count
import base64

from io import StringIO
from .. import qtall as qt

from . import doc
from . import operations
from . import widgetfactory

# mime type for copy and paste
widgetmime = 'text/x-vnd.veusz-widget-3'

# dataset mime
datamime = 'text/x-vnd.veusz-data-1'

# svg mime
svgmime = 'image/svg+xml'

def generateWidgetsMime(widgets):
    """Create mime data describing widget and children.
    format is:
     numberofwidgets
     widgettype1
     widgetname1
     widgetpath1
     numberoflines1
     ...
     texttoreproducewidget
    """

    header = [str(len(widgets))]
    savetext = []

    for widget in widgets:
        header.append(widget.typename)
        header.append(repr(widget.name))
        header.append(repr(widget.path))
        save = widget.getSaveText()
        header.append( str(save.count('\n')) )
        savetext.append(save)

    header.append('')
    text = ('\n'.join(header) + ''.join(savetext)).encode('utf-8')

    mimedata = qt.QMimeData()
    mimedata.setData(widgetmime, qt.QByteArray(text))
    return mimedata

def generateDatasetsMime(datasets, document):
    """Generate mime for the list of dataset names given in the document.

    Format is:
    repr of names
    text to recreate dataset 1
    ...
    """

    mimedata = qt.QMimeData()

    # just plain text format
    output = []
    for name in datasets:
        output.append( document.data[name].datasetAsText() )
    text = ('\n'.join(output)).encode('utf-8')
    mimedata.setData('text/plain', qt.QByteArray(text))

    textfile = StringIO()
    for name in datasets:
        # get unlinked copy of dataset
        ds = document.data[name].returnCopy()

        # write into a string file
        ds.saveToFile(textfile, name)

    rawdata = textfile.getvalue().encode('utf-8')
    mimedata.setData(datamime, rawdata)

    return mimedata

def isClipboardDataMime():
    """Returns whether data available on clipboard."""
    mimedata = qt.QApplication.clipboard().mimeData()
    return datamime in mimedata.formats()

def getWidgetMime(mimedata):
    """Given mime data, return decoded python string."""
    formats = mimedata.formats()
    if widgetmime in formats:
        return mimedata.data(widgetmime).data().decode('utf-8')
    elif svgmime in formats:
        ba = mimedata.data(svgmime).data()
        return convertImgtoWidgetMime(ba, svgmime)
    else:
        return None

def getClipboardWidgetMime():
    """Returns widget mime data if mimedata contains correct mimetype or None

    If mimedata is set, use this rather than clipboard directly
    """
    clipboard = qt.QApplication.clipboard()
    clipboardwidgetmime = getWidgetMime(clipboard.mimeData())
    if clipboardwidgetmime is not None:
        return clipboardwidgetmime
    else:
        qimage = clipboard.image()
        if qimage.isNull():
            return None
        else:
            ba = qt.QByteArray()
            buffer = qt.QBuffer(ba)
            buffer.open(qt.QIODevice.WriteOnly)
            qimage.save(buffer, 'png')
            return convertImgtoWidgetMime(ba, 'png')

def getMimeWidgetTypes(data):
    """Get list of widget types in the mime data."""
    lines = data.split('\n')
    try:
        numwidgets = int(lines[0])
    except ValueError:
        return []
    types = lines[1:1+4*numwidgets:4]
    return types

def getMimeWidgetPaths(data):
    """Get list of widget paths in the mime data."""
    lines = data.split('\n')

    numwidgets = int(lines[0])
    paths = [eval(x) for x in lines[3:3+4*numwidgets:4]]
    return paths

def isWidgetMimePastable(parentwidget, mimedata):
    """Is widget mime data suitable to paste at parentwidget?"""

    if mimedata is None:
        return False
    types = getMimeWidgetTypes(mimedata)
    for type in types:
        if doc.getSuitableParent(type, parentwidget) is None:
            return False
    return True

def isMimeDropable(parentwidget, mimedata):
    """Can parent have this data pasted directly inside?"""
    if mimedata is None or parentwidget is None:
        return False
    types = getMimeWidgetTypes(mimedata)
    for type in types:
        wc = widgetfactory.thefactory.getWidgetClass(type)
        if not wc.willAllowParent(parentwidget):
            return False
    return True

def getMimeWidgetCount(mimedata):
    """Get number of widgets in mimedata."""
    return int( mimedata[:mimedata.find('\n')] )

def convertImgtoWidgetMime(ba, mimetype):
    """Given image bite array and mimetype, return decoded python string."""
    if mimetype == svgmime:
        typename = 'svgfile'
        name = 'svgfile1'
        path = '/page1/svgfile1'
        key = 'embeddedSVGData'
    else:
        typename = 'imagefile'
        name = 'imagefile1'
        path = '/page1/imagefile1'
        key = 'embeddedImageData'

    encoded = base64.b64encode(ba).decode('ascii')
    settings = {'filename': "'{embedded}'",
                key: "'{}'".format(encoded),
                }
    head = ['1', typename, "'{}'".format(name), "'{}'".format(path), '2']
    body = ["Set('{}', {})".format(s,v) for (s,v) in settings.items()]

    return '\n'.join(head) + '\n' + '\n'.join(body) + '\n'

class OperationWidgetPaste(operations.OperationMultiple):
    """Paste a widget from mime data."""

    descr= 'paste widget'

    def __init__(self, parent, mimedata, index=-1, newnames=None):
        """Paste widget into parent widget from mimedata.

        newnames is a list of new names for pasting, if given."""

        operations.OperationMultiple.__init__(self, [], descr=None)
        self.parentpath = parent.path
        self.mimedata = mimedata
        self.index = index
        self.newnames = newnames

    def do(self, document):
        """Do the import."""

        from . import commandinterpreter

        index = self.index

        # get document to keep track of changes for undo/redo
        document.batchHistory(self)

        # fire up interpreter to read file
        interpreter = commandinterpreter.CommandInterpreter(document)
        parentwidget = document.resolveWidgetPath(None, self.parentpath)

        lines = self.mimedata.split('\n')
        numwidgets = int(lines[0])

        # get types, names and number of lines for widgets
        types = lines[1:1+4*numwidgets:4]
        names = lines[2:2+4*numwidgets:4]
        names = [eval(name) for name in names]
        if self.newnames is not None:
            names = self.newnames
        # paths = lines[3:3+4*numwidgets:4] (not required here)
        widgetslines = lines[4:4+4*numwidgets:4]
        widgetslines = [int(x) for x in widgetslines]

        newwidgets = []
        widgetline = 1+4*numwidgets
        try:
            for wtype, name, numline in zip(types, names, widgetslines):
                thisparent = doc.getSuitableParent(wtype, parentwidget)

                if thisparent is None:
                    raise RuntimeError("Cannot find suitable parent for pasting")

                # override name if it exists already
                if name in thisparent.childnames:
                    name = None

                # make new widget
                widget = document.applyOperation(
                    operations.OperationWidgetAdd(
                        thisparent, wtype, autoadd=False,
                        name=name, index=index) )
                newwidgets.append(widget)

                # run generating commands
                interpreter.interface.currentwidget = widget
                for line in lines[widgetline:widgetline+numline]:
                    interpreter.run(line)

                if index >= 0:
                    index += 1

                # move to next widget
                widgetline += numline

        except Exception:
            document.batchHistory(None)
            raise

        # stop batching changes
        document.batchHistory(None)
        return newwidgets

class OperationWidgetClone(OperationWidgetPaste):
    """Clone a widget."""

    descr = 'clone widget'

    def __init__(self, widget, newparent, newname):
        mime = generateWidgetsMime([widget])
        mimedec = mime.data(widgetmime).data().decode('utf-8')
        OperationWidgetPaste.__init__(
            self, newparent, mimedec, newnames=[newname])

    def do(self, document):
        """Do the import."""
        widgets = OperationWidgetPaste.do(self, document)
        return widgets[0]

class OperationDataPaste(operations.Operation):
    """Paste dataset from mime data."""

    descr = 'paste data'

    def __init__(self, mimedata):
        """Paste datasets into document."""
        self.data = mimedata.data(datamime).data().decode('utf-8')

    def do(self, thisdoc):
        """Do the data paste."""

        from . import commandinterpreter

        # write data into a temporary document
        tempdoc = doc.Document()
        # interpreter to create datasets
        interpreter = commandinterpreter.CommandInterpreter(tempdoc)
        interpreter.runFile(StringIO(self.data))

        # list of pasted datasets
        self.newds = []

        # now transfer datasets to existing document
        for name, ds in sorted(tempdoc.data.items()):

            # get new name
            if name not in thisdoc.data:
                newname = name
            else:
                for idx in count(2):
                    newname = '%s_%s' % (name, idx)
                    if newname not in thisdoc.data:
                        break

            thisdoc.setData(newname, ds)
            self.newds.append(newname)

    def undo(self, thisdoc):
        """Undo pasting datasets."""
        for n in self.newds:
            thisdoc.deleteData(n)
