#    Copyright (C) 2005 Jeremy S. Sanders
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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
###############################################################################

# $Id$

"""Module for creating QWidgets for the settings, to enable their values
   to be changed.

    These widgets emit settingChanged(control, setting, val) when the setting is
    changed. The creator should use this to change the setting.
"""

import itertools
import re

import qt
import qttable

import setting
import utils

def _populateCombo(combo, items):
    """Populate the combo with the list of items given.

    This also makes sure the currently entered text persists
    """

    # existing setting
    currenttext = unicode(combo.currentText())

    # get rid of existing items in list (clear doesn't work here)
    for i in range(combo.count()):
        combo.removeItem(0)

    # get index for value, or add value if not set
    try:
        index = items.index(currenttext)
    except ValueError:
        items.append(currenttext)
        index = len(items)-1

    # put in new entries
    combo.insertStrList(items)
    
    # set index to current value
    combo.setCurrentItem(index)

class Edit(qt.QLineEdit):
    """Main control for editing settings which are text."""

    def __init__(self, setting, parent):
        """Initialise the setting widget."""

        qt.QLineEdit.__init__(self, parent)
        self.setting = setting
        self.bgcolor = self.paletteBackgroundColor()

        # set the text of the widget to the 
        self.setText( setting.toText() )

        self.connect(self, qt.SIGNAL('returnPressed()'),
                     self.validateAndSet)
        self.connect(self, qt.SIGNAL('lostFocus()'),
                     self.validateAndSet)

        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.setReadOnly(True)

    def validateAndSet(self):
        """Check the text is a valid setting and update it."""

        text = unicode(self.text())
        try:
            val = self.setting.fromText(text)
            self.setPaletteBackgroundColor(self.bgcolor)

            # value has changed
            if self.setting.val != val:
                self.emit( qt.PYSIGNAL('settingChanged'), (self, self.setting, val) )
                #self.setting.val = val

        except setting.InvalidType:
            self.setPaletteBackgroundColor(qt.QColor('red'))

    def onModified(self, mod):
        """called when the setting is changed remotely"""
        self.setText( self.setting.toText() )

class _EscapeLineEdit(qt.QTextEdit):
    """A special line editor which signals when an escape character is pressed.

    Emits escapePressed()
    """

    def __init__(self, parent):
        qt.QTextEdit.__init__(self, parent)
        self.setTextFormat(qt.Qt.PlainText)
    
    def keyPressEvent(self, event):
        qt.QTextEdit.keyPressEvent(self, event)
        if event.key() == qt.Qt.Key_Escape:
            self.emit( qt.PYSIGNAL('escapePressed'), () )

class _EditBox(qt.QFrame):
    """A popup edit box to support editing long text sections.

    Emits closing(text) when the box closes
    """

    def __init__(self, origtext, readonly, parent):
        """Make a popup, framed widget containing a text editor."""

        qt.QFrame.__init__(self, parent, 'settingeditbox',
                           qt.Qt.WType_Popup)

        self.spacing = self.fontMetrics().height()
        self.layout = qt.QVBoxLayout(self, self.spacing/4)

        self.edit = _EscapeLineEdit(self)
        self.layout.addWidget(self.edit)
        self.connect(self.edit, qt.PYSIGNAL('escapePressed'),
                     self.escapePressed)
        self.connect(self.edit, qt.SIGNAL('returnPressed()'),
                     self.close)

        self.origtext = origtext
        self.edit.setText(origtext)

        if self.style().inherits("QWindowsStyle"):
            fs = qt.QFrame.WinPanel
        else:
            fs = qt.QFrame.Panel
        self.setFrameStyle( fs | qt.QFrame.Raised )            

        if readonly:
            self.edit.setReadOnly(True)

        self.positionSelf(parent)

    def sizeHint(self):
        """A reasonable size for the text editor."""
        return qt.QSize(self.spacing*40, self.spacing*3)

    def positionSelf(self, widget):
        """Open the edit box below the widget."""

        pos = widget.parentWidget().mapToGlobal( widget.pos() )
        desktop = qt.QApplication.desktop()

        # recalculates out position so that size is correct below
        self.adjustSize()

        # is there room to put this widget besides the widget?
        if pos.y() + self.height() + 1 < desktop.height():
            # put below
            y = pos.y() + 1
        else:
            # put above
            y = pos.y() - self.height() - 1
        
        # is there room to the left for us?
        if ( (pos.x() + widget.width() + self.width() < desktop.width()) or
             (pos.x() + widget.width() < desktop.width()/2) ):
            # put left justified with widget
            x = pos.x() + widget.width()
        else:
            # put extending to left
            x = pos.x() - self.width() - 1

        self.move(x, y)
        self.edit.moveCursor(qt.QTextEdit.MoveEnd, False)
        self.edit.setFocus()

    def escapePressed(self):
        """If the user wants to break back out."""
        self.edit.setText(self.origtext)
        self.close()
 
    def closeEvent(self, event):
        """Tell the calling widget that we are closing, and provide
        the new text."""

        text = unicode(self.edit.text())
        text = text.replace('\n', '')
        self.emit( qt.PYSIGNAL('closing'), (text,) )
        event.accept()

class String(qt.QHBox):
    """A line editor which allows editting in a larger popup window."""

    def __init__(self, setting, parent):
        qt.QHBox.__init__(self, parent)

        self.setting = setting
        self.edit = qt.QLineEdit(self)
        b = self.button = qt.QPushButton('..', self)
        b.setMaximumWidth(b.height())
        b.setToggleButton(True)

        self.bgcolor = self.edit.paletteBackgroundColor()
        
        # set the text of the widget to the 
        self.edit.setText( setting.toText() )

        self.connect(self.edit, qt.SIGNAL('returnPressed()'),
                     self.validateAndSet)
        self.connect(self.edit, qt.SIGNAL('lostFocus()'),
                     self.validateAndSet)
        self.connect(b, qt.SIGNAL('toggled(bool)'),
                     self.buttonToggled)

        self.setting.setOnModified(self.onModified)

        self.editwin = None

        if setting.readonly:
            self.edit.setReadOnly(True)

    def buttonToggled(self, on):
        """Button is pressed to bring popup up / down."""

        # if button is down and there's no existing popup, bring up a new one
        if on and self.editwin == None:
            e = _EditBox( unicode(self.edit.text()),
                          self.setting.readonly, self.button)

            # we get notified with text when the popup closes
            self.connect(e, qt.PYSIGNAL('closing'), self.boxClosing)
            e.show()
            self.editwin = e

    def boxClosing(self, text):
        """Called when the popup edit box closes."""

        # update the text if we can
        if not self.setting.readonly:
            self.edit.setText(text)
            self.edit.setFocus()
            self.parentWidget().setFocus()
            self.edit.setFocus()

        # KLUDGE! KLUDGE! KLUDGE!
        # this evily has to check a bit later whether a new window has been
        # created before turing off the toggle button
        # unfortunately clicking on the button to close the popup means
        # a new popup is created if we don't do this
        
        self.editwin = None
        qt.QTimer.singleShot(100, self.timerButtonOff)

    def timerButtonOff(self):
        """Disable button if there's no popup window."""
        if self.editwin == None:
            self.button.setOn(False)

    def validateAndSet(self):
        """Check the text is a valid setting and update it."""

        text = unicode(self.edit.text())
        try:
            val = self.setting.fromText(text)
            self.edit.setPaletteBackgroundColor(self.bgcolor)

            # value has changed
            if self.setting.val != val:
                self.emit( qt.PYSIGNAL('settingChanged'), (self, self.setting, val) )

        except setting.InvalidType:
            self.edit.setPaletteBackgroundColor(qt.QColor('red'))

    def onModified(self, mod):
        """called when the setting is changed remotely"""
        self.edit.setText( self.setting.toText() )
        
class Bool(qt.QCheckBox):
    """A check box for changing a bool setting."""
    
    def __init__(self, setting, parent):
        qt.QCheckBox.__init__(self, parent)

        self.setting = setting
        self.setChecked(setting.val)

        # we get a signal when the button is toggled
        self.connect( self, qt.SIGNAL('toggled(bool)'),
                      self.slotToggled )

        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.setEnabled(False)

    def slotToggled(self, state):
        """Emitted when checkbox toggled."""
        self.emit( qt.PYSIGNAL('settingChanged'), (self, self.setting, state) )
        
    def onModified(self, mod):
        """called when the setting is changed remotely"""
        self.setChecked( self.setting.val )

class Choice(qt.QComboBox):
    """For choosing between a set of values."""

    def __init__(self, setting, iseditable, vallist, parent, pixmaps=None):
        
        qt.QComboBox.__init__(self, parent)
        self.setting = setting
        self.bgcolor = self.paletteBackgroundColor()

        self.setEditable(iseditable)

        if pixmaps == None:
            # add items to list (text only)
            self.insertStrList( list(vallist) )
        else:
            # add pixmaps and text to list
            for pix, txt in itertools.izip(pixmaps, vallist):
                self.insertItem(pix, txt, -1)

        # set the text of the widget to the setting
        self.setCurrentText( setting.toText() )

        # if a different item is selected
        self.connect( self, qt.SIGNAL('activated(const QString&)'),
                      self.slotActivated )

        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.setEnabled(False)

    def focusOutEvent(self, *args):
        """Allows us to check the contents of the widget."""
        qt.QComboBox.focusOutEvent(self, *args)
        self.slotActivated('')

    def slotActivated(self, val):
        """If a different item is chosen."""
        text = unicode(self.currentText())
        try:
            val = self.setting.fromText(text)
            self.setPaletteBackgroundColor(self.bgcolor)
            
            # value has changed
            if self.setting.val != val:
                self.emit( qt.PYSIGNAL('settingChanged'), (self, self.setting, val) )

        except setting.InvalidType:
            self.setPaletteBackgroundColor(qt.QColor('red'))

    def onModified(self, mod):
        """called when the setting is changed remotely"""
        self.setCurrentText( self.setting.toText() )

class MultiLine(qt.QTextEdit):
    """For editting multi-line settings."""

    def __init__(self, setting, parent):
        """Initialise the widget."""

        qt.QTextEdit.__init__(self, parent)
        self.bgcolor = self.paletteBackgroundColor()
        self.setting = setting

        self.setTextFormat(qt.Qt.PlainText)
        self.setWordWrap(qt.QTextEdit.NoWrap)
        
        # set the text of the widget to the 
        self.setText( setting.toText() )

        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.setReadOnly(True)

    def focusOutEvent(self, *args):
        """Allows us to check the contents of the widget."""
        qt.QTextEdit.focusOutEvent(self, *args)

        text = unicode(self.text())
        try:
            val = self.setting.fromText(text)
            self.setPaletteBackgroundColor(self.bgcolor)
            
            # value has changed
            if self.setting.val != val:
                self.emit( qt.PYSIGNAL('settingChanged'), (self, self.setting, val) )

        except setting.InvalidType:
            self.setPaletteBackgroundColor(qt.QColor('red'))

    def onModified(self, mod):
        """called when the setting is changed remotely"""
        self.setText( self.setting.toText() )

class Distance(Choice):
    """For editing distance settings."""

    # used to remove non-numerics from the string
    # we also remove X/ from X/num
    stripnumre = re.compile(r"[0-9]*/|[^0-9.]")

    # remove spaces
    stripspcre = re.compile(r"\s")

    def __init__(self, setting, parent):
        '''Initialise with blank list, then populate with sensible units.'''
        Choice.__init__(self, setting, True, [], parent)
        self.updateComboList()
        
    def updateComboList(self):
        '''Populates combo list with sensible list of other possible units.'''

        # turn off signals, so our modifications don't create more signals
        self.blockSignals(True)

        # get current text
        text = unicode(self.currentText())

        # get rid of non-numeric things from the string
        num = self.stripnumre.sub('', text)

        # here are a list of possible different units the user can choose
        # between. should this be in utils?
        newitems = [ num+'pt', num+'cm', num+'mm',
                     num+'in', num+'%', '1/'+num ]

        # if we're already in this list, we position the current selection
        # to the correct item (up and down keys work properly then)
        # spaces are removed to make sure we get sensible matches
        spcfree = self.stripspcre.sub('', text)
        try:
            index = newitems.index(spcfree)
        except ValueError:
            index = 0
            newitems.insert(0, text)

        # get rid of existing items in list (clear doesn't work here)
        for i in range(self.count()):
            self.removeItem(0)

        # put new items in and select the correct option
        self.insertStrList(newitems)
        self.setCurrentItem(index)

        # must remember to do this!
        self.blockSignals(False)

    def slotActivated(self, val):
        '''Populate the drop down list before activation.'''
        self.updateComboList()
        Choice.slotActivated(self, val)

class Dataset(Choice):
    """Allow the user to choose between the possible datasets."""

    def __init__(self, setting, document, dimensions, parent):
        """Initialise the combobox. The list is populated with datasets.

        dimensions specifies the dimension of the dataset to list

        Changes on the document refresh the list of datasets."""
        
        Choice.__init__(self, setting, True, [], parent)
        self.document = document
        self.dimensions = dimensions
        self._populateEntries()
        self.connect(document, qt.PYSIGNAL('sigModified'),
                     self.slotModified)

    def _populateEntries(self):
        """Put the list of datasets into the combobox."""

        # get datasets of the correct dimension
        datasets = []
        for name, ds in self.document.data.iteritems():
            if ds.dimensions == self.dimensions:
                datasets.append(name)
        datasets.sort()

        _populateCombo(self, datasets)

    def slotModified(self, modified):
        """Update the list of datasets if the document is modified."""
        self._populateEntries()
        
class FillStyle(Choice):
    """For choosing between fill styles."""

    _pixmaps = None
    _fills = None
    _fillcnvt = None

    def __init__(self, setting, parent):
        if self._pixmaps == None:
            self._generatePixmaps()

        Choice.__init__(self, setting, False,
                        self._fills, parent,
                        pixmaps=self._pixmaps)

    def _generatePixmaps(cls):
        """Generate a list of pixmaps for drop down menu."""

        size = 12
        pixmaps = []
        c = qt.QColor('darkgrey')
        for f in cls._fills:
            pix = qt.QPixmap(size, size)
            pix.fill()
            painter = qt.QPainter(pix)
            brush = qt.QBrush(c, cls._fillcnvt[f])
            painter.fillRect(0, 0, size, size, brush)
            pixmaps.append(pix)

        cls._pixmaps = pixmaps
    _generatePixmaps = classmethod(_generatePixmaps)

class Marker(Choice):
    """A control to let the user choose a marker."""

    _pixmaps = None

    def __init__(self, setting, parent):
        if self._pixmaps == None:
            self._generatePixmaps()

        Choice.__init__(self, setting, False,
                        utils.MarkerCodes, parent,
                        pixmaps=self._pixmaps)

    def _generatePixmaps(cls):
        size = 16
        pixmaps = []
        c = qt.QColor('darkgrey')
        for marker in utils.MarkerCodes:
            pix = qt.QPixmap(size, size)
            pix.fill()
            painter = qt.QPainter(pix)
            painter.setBrush(c)
            utils.plotMarker(painter, size/2, size/2, marker, int(size*0.33))
            pixmaps.append(pix)

        cls._pixmaps = pixmaps
    _generatePixmaps = classmethod(_generatePixmaps)

class LineStyle(Choice):
    """For choosing between line styles."""

    _pixmaps = None
    _lines = None
    _linecnvt = None

    def __init__(self, setting, parent):
        if self._pixmaps == None:
            self._generatePixmaps()

        Choice.__init__(self, setting, False,
                        self._lines, parent,
                        pixmaps=self._pixmaps)

    def _generatePixmaps(cls):
        """Generate a list of pixmaps for drop down menu."""
        size = 12
        pixmaps = []
        c = qt.QColor('black')
        for l in cls._lines:
            pix = qt.QPixmap(size*4, size)
            pix.fill()
            painter = qt.QPainter(pix)
            pen = qt.QPen(c, 2, cls._linecnvt[l])
            painter.setPen(pen)
            painter.drawLine(size, size/2, size*3, size/2)
            pixmaps.append(pix)

        cls._pixmaps = pixmaps
    _generatePixmaps = classmethod(_generatePixmaps)

class Color(qt.QHBox):
    """A control which lets the user choose a color.

    A drop down list and a button to bring up a dialog are used
    """

    _pixmaps = None
    _colors = None

    def __init__(self, setting,  parent):
        qt.QHBox.__init__(self, parent)

        if self._pixmaps == None:
            self._generatePixmaps()

        self.setting = setting

        # combo box
        c = self.combo = qt.QComboBox(self)
        c.setEditable(True)
        for color in self._colors:
            c.insertItem(self._pixmaps[color], color, -1)
        c.setCurrentText( self.setting.toText() )
        self.connect(c, qt.SIGNAL('activated(const QString&)'),
                     self.slotActivated )

        # button for selecting colors
        b = self.button = qt.QPushButton(self)
        b.setMaximumWidth(b.height())
        self.connect(b, qt.SIGNAL('clicked()'),
                     self.slotButtonClicked)

        if setting.readonly:
            c.setEnabled(False)
            b.setEnabled(False)
                     
        self.setting.setOnModified(self.onModified)
        self._updateButtonColor()

    def _generatePixmaps(cls):
        """Generate a list of pixmaps for drop down menu.
        Does not generate existing pixmaps
        """

        size = 12
        if cls._pixmaps == None:
            cls._pixmaps = {}
        
        pixmaps = cls._pixmaps
        for c in cls._colors:
            if c not in pixmaps:
                pix = qt.QPixmap(size, size)
                pix.fill( qt.QColor(c) )
                pixmaps[c] = pix

    _generatePixmaps = classmethod(_generatePixmaps)
    
    def _updateButtonColor(self):
        """Update the color on the button from the setting."""

        size = 12
        pix = qt.QPixmap(size, size)
        pix.fill(self.setting.color())

        self.button.setIconSet( qt.QIconSet(pix) )

    def slotButtonClicked(self):
        """Open dialog to edit color."""

        col = qt.QColorDialog.getColor( self.setting.color(),
                                        self )
        if col.isValid():
            # change setting
            name = col.name()
            self.emit( qt.PYSIGNAL('settingChanged'), (self, self.setting, unicode(name)) )

    def slotActivated(self, val):
        """A different value is selected."""
        
        text = unicode(self.combo.currentText())
        val = self.setting.fromText(text)
            
        # value has changed
        if self.setting.val != val:
            self.emit( qt.PYSIGNAL('settingChanged'), (self, self.setting, val) )

    def onModified(self, mod):
        """called when the setting is changed remotely"""

        self.combo.setCurrentText( self.setting.toText() )
        self._updateButtonColor()

class Axis(Choice):
    """Choose an axis to plot against."""

    def __init__(self, setting, document, direction, parent):
        """Initialise and populate combobox."""

        Choice.__init__(self, setting, True, [], parent)
        self.document = document
        self.direction = direction
        self._populateEntries()
        self.connect(document, qt.PYSIGNAL('sigModified'),
                     self.slotModified)

    def _populateEntries(self):
        """Build up a list of possible axes."""

        # get parent widget
        widget = self.setting
        while not widget.isWidget() and widget != None:
            widget = widget.parent

        # get list of axis widgets up the tree
        axes = {}
        while widget != None:
            for w in widget.children:
                try:
                    # succeeds if axis
                    if w.settings.direction == self.direction:
                        axes[w.name] = True
                except AttributeError:
                    pass
            widget = widget.parent

        names = axes.keys()
        names.sort()

        _populateCombo(self, names)

    def slotModified(self, modified):
        """Update list of axes."""
        self._populateEntries()

class ListSet(qt.QWidget):
    """A widget for constructing settings which are lists of other
    properties.

    This code is pretty nasty and horrible, so we abstract it in this
    base widget
    """

    pixsize = 12

    def __init__(self, defaultval, setting, parent):
        """Initialise this base widget.

        defaultval is the default entry to add if add is clicked with
        no current entries

        setting is the setting this widget corresponds to

        parent is the parent widget.
        """
        
        qt.QWidget.__init__(self, parent)
        self.defaultval = defaultval
        self.setting = setting
        self.controls = []
        self.layout = qt.QGridLayout(self, 1, 1, 2, 2)

        # ignore changes if this set
        self.ignorechange = False

        self.populate()
        self.setting.setOnModified(self.onModified)
    
    def populateRow(self, row, val):
        """Populate the row in the control.

        Returns a list of the widgets created.
        """
        return None
    
    def populate(self):
        """Construct the list of controls."""

        # delete all children in case of refresh
        self.controls = []
        for c in self.children():
            if isinstance(c, qt.QWidget):
                self.layout.remove(c)
                c.deleteLater()
        c = None

        # iterate over each row
        row = -1
        for row, val in enumerate(self.setting.val):
            cntrls = self.populateRow(row, val)
            for i in cntrls:
                i.show()
            self.controls.append(cntrls)

        h = qt.QHBox(self)
        self.layout.addMultiCellWidget(h, row+1, row+1, 0,
                                       self.layout.numCols()-1)
        
        # a button to add a new entry
        b = qt.QPushButton('Add new entry', h)
        self.connect(b, qt.SIGNAL('clicked()'), self.onAddClicked)
        b.show()

        # a button to delete the last entry
        b = qt.QPushButton('Delete entry', h)
        self.connect(b, qt.SIGNAL('clicked()'), self.onDeleteClicked)
        b.setEnabled( len(self.setting.val) > 0 )
        b.show()
        h.show()
        
        self._adjustSize()
        
    def _adjustSize(self):
        """Tell the Grid to make us the correct size."""
        
        # EVIL CODE BELOW - KLUDGE!
        # when the widget resizes, it must tell the QTable it is in
        # to adjust its row! Yuck! There must be a better way to do this
        # not sure why the QTable is 3 levels up

        table = self
        for i in range(3):
            table = table.parent()
            if table == None:
                break

        # only do this if the parent is a table
        if table != None and isinstance(table, qttable.QTable):
            # we have to check each widget to see which row we're on
            for r in xrange(table.numRows()):
                for c in xrange(table.numCols()):
                    if table.cellWidget(r, c) == self:
                        table.adjustRow(r)

    def polish(self):
        """Remove tooltip from widget - avoid Qt bugs."""
        qt.QVBox.polish(self)
        qt.QToolTip.remove(self)

    def onAddClicked(self):
        """Add a line style to the list given."""

        rows = list(self.setting.val)
        if len(rows) != 0:
            rows.append(rows[-1])
        else:
            rows.append(self.defaultval)
        self.emit( qt.PYSIGNAL('settingChanged'), (self, self.setting, rows) )

    def onDeleteClicked(self):
        """Remove final entry in settings list."""

        rows = list(self.setting.val)[:-1]
        self.emit( qt.PYSIGNAL('settingChanged'), (self, self.setting, rows) )

    def onModified(self, mod):
        """called when the setting is changed remotely"""

        if not self.ignorechange:
            self.populate()
        else:
            self.ignorechange = False

    def identifyPosn(self, widget):
        """Identify the position this widget is in.

        Returns (row, col) or (None, None) if not found.
        """

        for row, cntrls in enumerate(self.controls):
            for col, cntrl in enumerate(cntrls):
                if cntrl == widget:
                    return (row, col)
        return (None, None)

    def addColorButton(self, row, col, tooltip):
        """Add a color button to the list at the position specified."""

        color = self.setting.val[row][col]
        wcolor = qt.QPushButton(self)
        self.layout.addWidget(wcolor, row, col)
        wcolor.setMaximumWidth(wcolor.height())
        pix = qt.QPixmap(self.pixsize, self.pixsize)
        pix.fill( qt.QColor(color) )
        wcolor.setIconSet( qt.QIconSet(pix) )
        qt.QToolTip.add(wcolor, tooltip)
        self.connect(wcolor, qt.SIGNAL('clicked()'), self.onColorClicked)
        return wcolor

    def addToggleButton(self, row, col, tooltip):
        """Make a toggle button."""

        toggle = self.setting.val[row][col]
        wtoggle = qt.QCheckBox(self)
        self.layout.addWidget(wtoggle, row, col)
        wtoggle.setChecked(toggle)
        qt.QToolTip.add(wtoggle, tooltip)
        self.connect(wtoggle, qt.SIGNAL('toggled(bool)'), self.onToggled)
        return wtoggle

    def addCombo(self, row, col, tooltip, values, pixmaps, texts):
        """Make an enumeration combo - choose from a set of pixmaps."""
        
        val = self.setting.val[row][col]

        wcombo = qt.QComboBox(self)
        self.layout.addWidget(wcombo, row, col)

        if texts == None:
            for pixmap in pixmaps:
                wcombo.insertItem(pixmap)
        else:
            for text, pixmap in zip(texts, pixmaps):
                wcombo.insertItem(pixmap, text)

        wcombo.setCurrentItem(values.index(val))
        qt.QToolTip.add(wcombo, tooltip)
        self.connect(wcombo, qt.SIGNAL('activated(int)'),
                     self.onComboChanged)
        wcombo._vz_values = values
        return wcombo

    def _updateRowCol(self, row, col, val):
        """Update value on row and column."""
        rows = list(self.setting.val)
        items = list(rows[row])
        items[col] = val
        rows[row] = tuple(items)
        self.ignorechange = True
        self.emit( qt.PYSIGNAL('settingChanged'), (self, self.setting, rows) )
        
    def onToggled(self, on):
        """Checkbox toggled."""
        row, col = self.identifyPosn(self.sender())
        self._updateRowCol(row, col, on)

    def onComboChanged(self, val):
        """Update the setting if the combo changes."""
        sender = self.sender()
        row, col = self.identifyPosn(sender)
        self._updateRowCol(row, col, sender._vz_values[val])

    def onColorClicked(self):
        """Color button clicked for line."""
        sender = self.sender()
        row, col = self.identifyPosn(sender)

        rows = self.setting.val
        color = qt.QColorDialog.getColor( qt.QColor(rows[row][col]),
                                          self )
        if color.isValid():
            # change setting
            # this is a bit irritating, as have to do lots of
            # tedious conversions
            color = unicode(color.name())
            self._updateRowCol(row, col, color)

            # change the color
            pix = qt.QPixmap(self.pixsize, self.pixsize)
            pix.fill(qt.QColor(color))
            sender.setIconSet( qt.QIconSet(pix) )
            
class LineSet(ListSet):
    """A list of line styles.
    """

    def __init__(self, setting, parent):
        ListSet.__init__(self, ('solid', '1pt', 'black', False),
                         setting, parent)

    def populateRow(self, row, val):
        """Add the widgets for the row given."""

        # create line pixmaps if not already created
        if LineStyle._pixmaps == None:
            LineStyle._generatePixmaps()

        # make line style selector
        wlinestyle = self.addCombo(row, 0, 'Line style',
                                   LineStyle._lines,
                                   LineStyle._pixmaps, None)
        
        # make line width edit box
        wwidth = qt.QLineEdit(self)
        self.layout.addWidget(wwidth, row, 1)
        wwidth.setText(self.setting.val[row][1])
        qt.QToolTip.add(wwidth, 'Line width')
        self.connect(wwidth, qt.SIGNAL('returnPressed()'),
                     self.onWidthChanged)
        self.connect(wwidth, qt.SIGNAL('lostFocus()'),
                     self.onWidthChanged)
        self.bgcolor = wwidth.paletteBackgroundColor()

        # make color selector button
        wcolor = self.addColorButton(row, 2, 'Line color')

        # make hide checkbox
        whide = self.addToggleButton(row, 3, 'Hide line')

        # return created controls
        return [wlinestyle, wwidth, wcolor, whide]

    def onWidthChanged(self):
        """Width has changed - validate."""

        sender = self.sender()
        row, col = self.identifyPosn(sender)

        text = unicode(sender.text())
        if setting.Distance.isDist(text):
            # valid distance
            sender.setPaletteBackgroundColor(self.bgcolor)
            self._updateRowCol(row, col, text)
        else:
            # invalid distance
            sender.setPaletteBackgroundColor(qt.QColor('red'))

class FillSet(ListSet):
    """A list of fill settings."""

    def __init__(self, setting, parent):
        ListSet.__init__(self, ('solid', 'black', False),
                         setting, parent)

    def populateRow(self, row, val):
        """Add the widgets for the row given."""

        # construct fill pixmaps if not already done
        if FillStyle._pixmaps == None:
            FillStyle._generatePixmaps()
    
        # make fill style selector
        wfillstyle = self.addCombo(row, 0, 'Fill style',
                                   FillStyle._fills,
                                   FillStyle._pixmaps,
                                   FillStyle._fills)
        wfillstyle.setMinimumWidth(self.pixsize)

        # make color selector button
        wcolor = self.addColorButton(row, 1, 'Fill color')

        # make hide checkbox
        whide = self.addToggleButton(row, 2, 'Hide fill')

        # return widgets
        return [wfillstyle, wcolor, whide]
