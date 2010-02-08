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

# $Id$

from itertools import izip

import veusz.qtall as qt4

import doc
import operations
import commandinterpreter
import widgetfactory

# mime type for copy and paste
widgetmime = 'text/x-vnd.veusz-widget-3'

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
    text = '\n'.join(header) + ''.join(savetext)

    mimedata = qt4.QMimeData()
    mimedata.setData(widgetmime, qt4.QByteArray(text))
    return mimedata

def getClipboardWidgetMime():
    """Returns widget mime data if clipboard contains mime data or None."""
    mimedata = qt4.QApplication.clipboard().mimeData()
    if widgetmime in mimedata.formats():
        return str(mimedata.data(widgetmime))
    else:
        return None

def getMimeWidgetTypes(data):
    """Get list of widget types in the mime data."""
    lines = data.split('\n')

    numwidgets = int(lines[0])
    types = lines[1:1+4*numwidgets:4]
    return types

def getMimeWidgetPaths(data):
    """Get list of widget paths in the mime data."""
    lines = data.split('\n')

    numwidgets = int(lines[0])
    paths = [eval(x) for x in lines[3:3+4*numwidgets:4]]
    return paths

def isMimePastable(parentwidget, mimedata):
    """Is mime data suitable to paste at parentwidget?"""

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

def pasteMime(parentwidget, mimedata, index=-1):
    """Paste mime data at parent widget, starting at index childindex

    Returns list of created widgets
    """

    document = parentwidget.document

    lines = mimedata.split('\n')
    numwidgets = int(lines[0])

    # get types, names and number of lines for widgets
    types = lines[1:1+4*numwidgets:4]
    names = lines[2:2+4*numwidgets:4]
    names = [eval(name) for name in names]
    paths = lines[3:3+4*numwidgets:4] # (not required here)
    widgetslines = lines[4:4+4*numwidgets:4]
    widgetslines = [int(x) for x in widgetslines]

    # start batching changes to document
    op = operations.OperationMultiple([], descr='paste')
    document.applyOperation(op)
    document.batchHistory(op)

    # create an interpreter to put the paste commands into
    interpreter = commandinterpreter.CommandInterpreter(document)

    newwidgets = []
    widgetline = 1+4*numwidgets
    for wtype, name, numline in izip(types, names, widgetslines):
        thisparent = doc.getSuitableParent(wtype, parentwidget)

        # override name if it exists already
        if name in thisparent.childnames:
            name = None

        # make new widget
        widget = document.applyOperation(
            operations.OperationWidgetAdd(thisparent, wtype, autoadd=False,
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

    # stop batching changes
    document.batchHistory(None)
    return newwidgets
