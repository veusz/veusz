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
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
###############################################################################

# $Id$

"""Module for creating QWidgets for the settings, to enable their values
   to be changed.

    These widgets emit settingChanged(control, setting, val) when the setting is
    changed. The creator should use this to change the setting.
"""

import itertools
import re

import veusz.qtall as qt4

import setting
import veusz.utils as utils

def populateCombo(combo, items):
    """Populate the combo with the list of items given.

    This also makes sure the currently entered text persists
    """

    # existing setting
    currenttext = unicode(combo.currentText())

    # get rid of existing items in list (clear doesn't work here)
    for i in xrange(combo.count()):
        combo.removeItem(0)

    # get index for value, or add value if not set
    try:
        index = items.index(currenttext)
    except ValueError:
        items.append(currenttext)
        index = len(items)-1

    # put in new entries
    combo.addItems(items)
    
    # set index to current value
    combo.setCurrentIndex(index)

class Edit(qt4.QLineEdit):
    """Main control for editing settings which are text."""

    def __init__(self, setting, parent):
        """Initialise the setting widget."""

        qt4.QLineEdit.__init__(self, parent)
        self.setting = setting
        self.bgcolor = self.palette().color(qt4.QPalette.Base)
        #self.bgcolor = self.paletteBackgroundColor()

        # set the text of the widget to the 
        self.setText( setting.toText() )

        self.connect(self, qt4.SIGNAL('editingFinished()'),
                     self.validateAndSet)

        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.setReadOnly(True)

    def validateAndSet(self):
        """Check the text is a valid setting and update it."""

        text = unicode(self.text())
        try:
            val = self.setting.fromText(text)
            self.palette().setColor(qt4.QPalette.Base, self.bgcolor)

            # value has changed
            if self.setting.val != val:
                self.emit( qt4.SIGNAL('settingChanged'),
                           self, self.setting, val )
                #self.setting.val = val

        except setting.InvalidType:
            self.palette().setColor(qt4.QPalette.Base, qt4.QColor('red'))

    def onModified(self, mod):
        """called when the setting is changed remotely"""
        self.setText( self.setting.toText() )

class _EditBox(qt4.QTextEdit):
    """A popup edit box to support editing long text sections.

    Emits closing(text) when the box closes
    """

    def __init__(self, origtext, readonly, parent):
        """Make a popup, framed widget containing a text editor."""

        qt4.QTextEdit.__init__(self, parent)
        self.setWindowFlags(qt4.Qt.Popup)

        self.spacing = self.fontMetrics().height()

        self.origtext = origtext
        self.setPlainText(origtext)

        cursor = self.textCursor()
        cursor.movePosition(qt4.QTextCursor.End)
        self.setTextCursor(cursor)

        if readonly:
            self.setReadOnly(True)

        self.positionSelf(parent)

    def keyPressEvent(self, event):
        """Close if escape or return is pressed."""
        qt4.QTextEdit.keyPressEvent(self, event)

        key = event.key()
        if key == qt4.Qt.Key_Escape:
            # restore original content
            self.setPlainText(self.origtext)
            self.close()
        elif key == qt4.Qt.Key_Return:
            # keep changes
            self.close()

    def sizeHint(self):
        """A reasonable size for the text editor."""
        return qt4.QSize(self.spacing*40, self.spacing*3)

    def positionSelf(self, widget):
        """Open the edit box below the widget."""

        pos = widget.parentWidget().mapToGlobal( widget.pos() )
        desktop = qt4.QApplication.desktop()

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
        self.setFocus()

    def closeEvent(self, event):
        """Tell the calling widget that we are closing, and provide
        the new text."""

        text = unicode(self.toPlainText())
        text = text.replace('\n', '')
        self.emit( qt4.SIGNAL('closing'), text)
        event.accept()

class String(qt4.QWidget):
    """A line editor which allows editting in a larger popup window."""

    def __init__(self, setting, parent):
        qt4.QWidget.__init__(self, parent)
        self.setting = setting

        layout = qt4.QHBoxLayout()
        layout.setSpacing(0)
        layout.setMargin(0)
        self.setLayout(layout)

        self.edit = qt4.QLineEdit()
        layout.addWidget(self.edit)

        b = self.button = qt4.QPushButton('..')
        layout.addWidget(b)
        b.setSizePolicy(qt4.QSizePolicy.Maximum, qt4.QSizePolicy.Maximum)
        b.setMaximumWidth(16)
        b.setCheckable(True)

        self.bgcolor = self.edit.palette().color(qt4.QPalette.Base)
        
        # set the text of the widget to the 
        self.edit.setText( setting.toText() )

        self.connect(self.edit, qt4.SIGNAL('editingFinished()'),
                     self.validateAndSet)
        self.connect(b, qt4.SIGNAL('toggled(bool)'),
                     self.buttonToggled)

        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.edit.setReadOnly(True)

    def buttonToggled(self, on):
        """Button is pressed to bring popup up / down."""

        # if button is down and there's no existing popup, bring up a new one
        if on:
            e = _EditBox( unicode(self.edit.text()),
                          self.setting.readonly, self.button)

            # we get notified with text when the popup closes
            self.connect(e, qt4.SIGNAL('closing'), self.boxClosing)
            e.show()

    def boxClosing(self, text):
        """Called when the popup edit box closes."""

        # update the text if we can
        if not self.setting.readonly:
            self.edit.setText(text)
            self.edit.setFocus()
            self.parentWidget().setFocus()
            self.edit.setFocus()

        self.button.setChecked(False)

    def validateAndSet(self):
        """Check the text is a valid setting and update it."""

        text = unicode(self.edit.text())
        try:
            val = self.setting.fromText(text)
            self.edit.palette().setColor(qt4.QPalette.Base, self.bgcolor)

            # value has changed
            if self.setting.val != val:
                self.emit( qt4.SIGNAL('settingChanged'), self, self.setting, val)

        except setting.InvalidType:
            self.edit.palette().setColor(qt4.QPalette.Base, qt4.QColor('red'))

    def onModified(self, mod):
        """called when the setting is changed remotely"""
        self.edit.setText( self.setting.toText() )

class Bool(qt4.QCheckBox):
    """A check box for changing a bool setting."""
    
    def __init__(self, setting, parent):
        qt4.QCheckBox.__init__(self, parent)

        self.setting = setting
        self.setChecked(setting.val)

        # we get a signal when the button is toggled
        self.connect( self, qt4.SIGNAL('toggled(bool)'),
                      self.slotToggled )

        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.setEnabled(False)

    def slotToggled(self, state):
        """Emitted when checkbox toggled."""
        self.emit( qt4.SIGNAL('settingChanged'), self, self.setting, state )
        
    def onModified(self, mod):
        """called when the setting is changed remotely"""
        self.setChecked( self.setting.val )

class Choice(qt4.QComboBox):
    """For choosing between a set of values."""

    def __init__(self, setting, iseditable, vallist, parent, icons=None):
        qt4.QComboBox.__init__(self, parent)

        self.setting = setting
        self.bgcolor = None

        self.setEditable(iseditable)

        # stops combobox readjusting in size to fit contents
        self.setSizeAdjustPolicy(
            qt4.QComboBox.AdjustToMinimumContentsLengthWithIcon)

        if icons is None:
            # add items to list (text only)
            self.addItems( list(vallist) )
        else:
            # add pixmaps and text to list
            for icon, text in itertools.izip(icons, vallist):
                self.addItem(icon, text)

        # choose the correct setting
        try:
            index = list(vallist).index(setting.toText())
            self.setCurrentIndex(index)
        except ValueError:
            # for cases when this is editable
            # set the text of the widget to the setting
            assert iseditable
            self.setEditText( setting.toText() )

        # if a different item is selected
        self.connect( self, qt4.SIGNAL('activated(const QString&)'),
                      self.slotActivated )

        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.setEnabled(False)

    def focusOutEvent(self, *args):
        """Allows us to check the contents of the widget."""
        qt4.QComboBox.focusOutEvent(self, *args)
        self.slotActivated('')

    def slotActivated(self, val):
        """If a different item is chosen."""

        # control to highlight if there are problems
        highcntrl = self.lineEdit()
        if highcntrl is None:
            highcntrl = self

        # keep track of original background
        if self.bgcolor is None:
            self.bgcolor = highcntrl.palette().color(qt4.QPalette.Base)

        text = unicode(self.currentText())
        try:
            val = self.setting.fromText(text)
            highcntrl.palette().setColor(qt4.QPalette.Base, self.bgcolor)
            
            # value has changed
            if self.setting.val != val:
                self.emit( qt4.SIGNAL('settingChanged'), self, self.setting, val )

        except setting.InvalidType:
            highcntrl.palette().setColor(qt4.QPalette.Base, qt4.QColor('red'))

    def onModified(self, mod):
        """called when the setting is changed remotely"""

        text = self.setting.toText()
        index = self.findText(text)
        if index >= 0:
            self.setCurrentIndex(index)
        if self.isEditable():
            self.setEditText(text)

class MultiLine(qt4.QTextEdit):
    """For editting multi-line settings."""

    def __init__(self, setting, parent):
        """Initialise the widget."""

        qt4.QTextEdit.__init__(self, parent)
        self.bgcolor = self.palette().color(qt4.QPalette.Window)
        self.setting = setting

        self.setWordWrapMode(qt4.QTextOption.NoWrap)
        self.setTabChangesFocus(True)
        
        # set the text of the widget to the 
        self.setPlainText( setting.toText() )

        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.setReadOnly(True)

    def focusOutEvent(self, *args):
        """Allows us to check the contents of the widget."""
        qt4.QTextEdit.focusOutEvent(self, *args)

        text = unicode(self.toPlainText())
        try:
            val = self.setting.fromText(text)
            self.palette().setColor(qt4.QPalette.Window, self.bgcolor)
            
            # value has changed
            if self.setting.val != val:
                self.emit( qt4.SIGNAL('settingChanged'), self, self.setting, val )

        except setting.InvalidType:
            self.palette().setColor(qt4.QPalette.Window, qt4.QColor('red'))

    def onModified(self, mod):
        """called when the setting is changed remotely"""
        self.setPlainText( self.setting.toText() )

class Distance(Choice):
    """For editing distance settings."""

    # used to remove non-numerics from the string
    # we also remove X/ from X/num
    stripnumre = re.compile(r"[0-9]*/|[^0-9.]")

    # remove spaces
    stripspcre = re.compile(r"\s")

    def __init__(self, setting, parent, allowauto=False):
        '''Initialise with blank list, then populate with sensible units.'''
        Choice.__init__(self, setting, True, [], parent)
        self.allowauto = allowauto
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

        if self.allowauto:
            newitems.insert(0, 'Auto')

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
        self.addItems(newitems)
        self.setCurrentIndex(index)

        # must remember to do this!
        self.blockSignals(False)

    def slotActivated(self, val):
        '''Populate the drop down list before activation.'''
        self.updateComboList()
        Choice.slotActivated(self, val)

class Dataset(Choice):
    """Allow the user to choose between the possible datasets."""

    def __init__(self, setting, document, dimensions, datatype, parent):
        """Initialise the combobox. The list is populated with datasets.

        dimensions specifies the dimension of the dataset to list

        Changes on the document refresh the list of datasets."""
        
        Choice.__init__(self, setting, True, [], parent)
        self.document = document
        self.dimensions = dimensions
        self.datatype = datatype
        self._populateEntries()
        self.connect(document, qt4.SIGNAL('sigModified'),
                     self.slotModified)

    def _populateEntries(self):
        """Put the list of datasets into the combobox."""

        # get datasets of the correct dimension
        datasets = []
        for name, ds in self.document.data.iteritems():
            if ds.dimensions == self.dimensions and ds.datatype == self.datatype:
                datasets.append(name)
        datasets.sort()

        populateCombo(self, datasets)

    def slotModified(self, modified):
        """Update the list of datasets if the document is modified."""
        self._populateEntries()
        
class DatasetOrString(qt4.QWidget):
    """Allow use to choose a dataset or enter some text."""

    def __init__(self, setting, document, dimensions, datatype, parent):
        qt4.QWidget.__init__(self, parent)
        self.datachoose = Dataset(setting, document, dimensions, datatype,
                                  None)
        
        b = self.button = qt4.QPushButton('..')
        b.setSizePolicy(qt4.QSizePolicy.Maximum, qt4.QSizePolicy.Maximum)
        b.setMaximumHeight(self.datachoose.height())
        b.setMaximumWidth(16)
        b.setCheckable(True)

        layout = qt4.QHBoxLayout()
        self.setLayout(layout)
        layout.setSpacing(0)
        layout.setMargin(0)
        layout.addWidget(self.datachoose)
        layout.addWidget(b)

        self.bgcolor = self.datachoose.palette().color(qt4.QPalette.Base)

        self.connect(b, qt4.SIGNAL('toggled(bool)'),
                     self.buttonToggled)
        self.connect(self.datachoose, qt4.SIGNAL('settingChanged'),
                     self.slotSettingChanged)

    def slotSettingChanged(self, *args):
        """When datachoose changes, inform any listeners."""
        self.emit( qt4.SIGNAL('settingChanged'), *args )
        
    def buttonToggled(self, on):
        """Button is pressed to bring popup up / down."""

        # if button is down and there's no existing popup, bring up a new one
        if on:
            e = _EditBox( unicode(self.datachoose.currentText()),
                          self.datachoose.setting.readonly, self.button)

            # we get notified with text when the popup closes
            self.connect(e, qt4.SIGNAL('closing'), self.boxClosing)
            e.show()

    def boxClosing(self, text):
        """Called when the popup edit box closes."""

        # update the text if we can
        if not self.datachoose.setting.readonly:
            self.datachoose.setEditText(text)
            self.datachoose.setFocus()
            self.parentWidget().setFocus()
            self.datachoose.setFocus()

        self.button.setChecked(False)

class FillStyle(Choice):
    """For choosing between fill styles."""

    _icons = None
    _fills = None
    _fillcnvt = None

    def __init__(self, setting, parent):
        if self._icons is None:
            self._generateIcons()

        Choice.__init__(self, setting, False,
                        self._fills, parent,
                        icons=self._icons)

    def _generateIcons(cls):
        """Generate a list of pixmaps for drop down menu."""

        size = 12
        icons = []
        c = qt4.QColor('grey')
        for f in cls._fills:
            pix = qt4.QPixmap(size, size)
            pix.fill()
            painter = qt4.QPainter(pix)
            painter.setRenderHint(qt4.QPainter.Antialiasing)
            brush = qt4.QBrush(c, cls._fillcnvt[f])
            painter.fillRect(0, 0, size, size, brush)
            painter.end()
            icons.append( qt4.QIcon(pix) )

        cls._icons = icons
    _generateIcons = classmethod(_generateIcons)

class Marker(Choice):
    """A control to let the user choose a marker."""

    _icons = None

    def __init__(self, setting, parent):
        if self._icons is None:
            self._generateIcons()

        Choice.__init__(self, setting, False,
                        utils.MarkerCodes, parent,
                        icons=self._icons)

    def _generateIcons(cls):
        size = 16
        icons = []
        brush = qt4.QBrush( qt4.QColor('darkgrey') )
        pen = qt4.QPen( qt4.QBrush(qt4.Qt.black), 1. )
        for marker in utils.MarkerCodes:
            pix = qt4.QPixmap(size, size)
            pix.fill()
            painter = qt4.QPainter(pix)
            painter.setRenderHint(qt4.QPainter.Antialiasing)
            painter.setBrush(brush)
            painter.setPen(pen)
            utils.plotMarker(painter, size*0.5, size*0.5, marker, size*0.33)
            painter.end()
            icons.append( qt4.QIcon(pix) )

        cls._icons = icons
    _generateIcons = classmethod(_generateIcons)

class Arrow(Choice):
    """A control to let the user choose an arrowhead."""

    _icons = None

    def __init__(self, setting, parent):
        if self._icons is None:
            self._generateIcons()

        Choice.__init__(self, setting, False,
                        utils.ArrowCodes, parent,
                        icons=self._icons)

    def _generateIcons(cls):
        size = 16
        icons = []
        brush = qt4.QBrush(qt4.Qt.black)
        pen = qt4.QPen( qt4.QBrush(qt4.Qt.black), 1. )
        for arrow in utils.ArrowCodes:
            pix = qt4.QPixmap(size, size)
            pix.fill()
            painter = qt4.QPainter(pix)
            painter.setRenderHint(qt4.QPainter.Antialiasing)
            painter.setBrush(brush)
            painter.setPen(pen)
            utils.plotLineArrow(painter, size*0.4, size*0.5,
                                size*2, 0.,
                                arrowsize=size*0.2,
                                arrowleft=arrow, arrowright=arrow)
            painter.end()
            icons.append( qt4.QIcon(pix) )

        cls._icons = icons
    _generateIcons = classmethod(_generateIcons)

class LineStyle(Choice):
    """For choosing between line styles."""

    _icons = None
    _lines = None
    _linecnvt = None

    size = (24, 8)

    def __init__(self, setting, parent):
        if self._icons is None:
            self._generateIcons()

        Choice.__init__(self, setting, False,
                        self._lines, parent,
                        icons=self._icons)
        self.setIconSize( qt4.QSize(*self.size) )

    def _generateIcons(cls):
        """Generate a list of icons for drop down menu."""

        # import later for dependency issues
        import veusz.setting.collections

        icons = []
        size = cls.size
        setn = veusz.setting.collections.Line('temp')
        setn.get('color').set('black')
        setn.get('width').set('1pt')
        
        for lstyle in cls._lines:
            pix = qt4.QPixmap(*size)
            pix.fill()
            painter = qt4.QPainter(pix)
            painter.setRenderHint(qt4.QPainter.Antialiasing)

            setn.get('style').set(lstyle)
            
            painter.setPen( setn.makeQPen(painter) )
            painter.drawLine( int(size[0]*0.1), size[1]/2,
                              int(size[0]*0.9), size[1]/2 )
            painter.end()
            icons.append( qt4.QIcon(pix) )

        cls._icons = icons
        
    _generateIcons = classmethod(_generateIcons)

class Color(qt4.QWidget):
    """A control which lets the user choose a color.

    A drop down list and a button to bring up a dialog are used
    """

    _icons = None
    _colors = None

    def __init__(self, setting,  parent):
        qt4.QWidget.__init__(self, parent)

        if self._icons is None:
            self._generateIcons()

        self.setting = setting

        # combo box
        c = self.combo = qt4.QComboBox()
        c.setEditable(True)
        for color in self._colors:
            c.addItem(self._icons[color], color)
        self.connect(c, qt4.SIGNAL('activated(const QString&)'),
                     self.slotActivated )

        # choose the correct setting
        try:
            index = self._colors.index(setting.toText())
            self.combo.setCurrentIndex(index)
        except ValueError:
            # not existing colors
            # set the text of the widget to the setting
            self.combo.setEditText( setting.toText() )

        # button for selecting colors
        b = self.button = qt4.QPushButton()
        b.setSizePolicy(qt4.QSizePolicy.Maximum, qt4.QSizePolicy.Maximum)
        b.setMaximumHeight(24)
        b.setMaximumWidth(24)
        self.connect(b, qt4.SIGNAL('clicked()'),
                     self.slotButtonClicked)

        if setting.readonly:
            c.setEnabled(False)
            b.setEnabled(False)
                     
        layout = qt4.QHBoxLayout()
        self.setLayout(layout)
        layout.setSpacing(0)
        layout.setMargin(0)
        layout.addWidget(c)
        layout.addWidget(b)

        self.setting.setOnModified(self.onModified)
        self._updateButtonColor()

    def _generateIcons(cls):
        """Generate a list of icons for drop down menu.
        Does not generate existing icons
        """

        size = 12
        if cls._icons is None:
            cls._icons = {}
        
        icons = cls._icons
        for c in cls._colors:
            if c not in icons:
                pix = qt4.QPixmap(size, size)
                pix.fill( qt4.QColor(c) )
                icons[c] = qt4.QIcon(pix)

    _generateIcons = classmethod(_generateIcons)
    
    def _updateButtonColor(self):
        """Update the color on the button from the setting."""

        size = 12
        pix = qt4.QPixmap(size, size)
        pix.fill(self.setting.color())

        self.button.setIcon( qt4.QIcon(pix) )

    def slotButtonClicked(self):
        """Open dialog to edit color."""

        col = qt4.QColorDialog.getColor(self.setting.color(), self)
        if col.isValid():
            # change setting
            val = unicode( col.name() )
            if self.setting.val != val:
                self.emit( qt4.SIGNAL('settingChanged'), self, self.setting, val)

    def slotActivated(self, val):
        """A different value is selected."""
        
        text = unicode(self.combo.currentText())
        val = self.setting.fromText(text)
            
        # value has changed
        if self.setting.val != val:
            self.emit( qt4.SIGNAL('settingChanged'), self, self.setting, val )

    def onModified(self, mod):
        """called when the setting is changed remotely"""

        self.combo.setEditText( self.setting.toText() )
        self._updateButtonColor()

class WidgetSelector(Choice):
    """For choosing from a list of widgets."""

    def __init__(self, setting, document, parent):
        """Initialise and populate combobox."""

        Choice.__init__(self, setting, True, [], parent)
        self.document = document
        self.connect(document, qt4.SIGNAL('sigModified'),
                     self.slotModified)

    def _populateEntries(self):
        pass
    
    def slotModified(self, modified):
        """Update list of axes."""
        self._populateEntries()

class Image(WidgetSelector):
    """Choose an image."""

    def __init__(self, setting, document, parent):
        """Initialise and populate combobox."""

        WidgetSelector.__init__(self, setting, document, parent)
        self._populateEntries()

    def _populateEntries(self):
        """Build up a list of images for combobox."""

        images = self.setting.getImageList()

        # we only need the list of names
        names = images.keys()
        names.sort()

        populateCombo(self, names)

class Axis(WidgetSelector):
    """Choose an axis to plot against."""

    def __init__(self, setting, document, direction, parent):
        """Initialise and populate combobox."""

        WidgetSelector.__init__(self, setting, document, parent)
        self.direction = direction
        self._populateEntries()

    def _populateEntries(self):
        """Build up a list of possible axes."""

        # get parent widget
        widget = self.setting.parent
        while not widget.isWidget() and widget is not None:
            widget = widget.parent

        # get list of axis widgets up the tree
        axes = {}
        while widget is not None:
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

        populateCombo(self, names)

class ListSet(qt4.QFrame):
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
        
        qt4.QFrame.__init__(self, parent)
        self.setFrameStyle(qt4.QFrame.Box)
        self.defaultval = defaultval
        self.setting = setting
        self.controls = []
        self.layout = qt4.QGridLayout(self)
        self.layout.setMargin( self.layout.margin()/2 )
        self.layout.setSpacing( self.layout.spacing()/4 )

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
            if isinstance(c, qt4.QWidget):
                self.layout.removeWidget(c)
                c.deleteLater()
        c = None

        # iterate over each row
        row = -1
        for row, val in enumerate(self.setting.val):
            cntrls = self.populateRow(row, val)
            for i in cntrls:
                i.show()
            self.controls.append(cntrls)

        # buttons at end
        bbox = qt4.QWidget()
        h = qt4.QHBoxLayout(bbox)
        h.setMargin(0)
        bbox.setLayout(h)
        self.layout.addWidget(bbox, row+1, 0, 1, -1)
        
        # a button to add a new entry
        b = qt4.QPushButton('Add')
        h.addWidget(b)
        self.connect(b, qt4.SIGNAL('clicked()'), self.onAddClicked)
        b.show()

        # a button to delete the last entry
        b = qt4.QPushButton('Delete')
        h.addWidget(b)
        self.connect(b, qt4.SIGNAL('clicked()'), self.onDeleteClicked)
        b.setEnabled( len(self.setting.val) > 0 )
        b.show()

    def polish(self):
        """Remove tooltip from widget - avoid Qt bugs."""
        qt4.QVBox.polish(self)
        qt4.QToolTip.remove(self)

    def onAddClicked(self):
        """Add a line style to the list given."""

        rows = list(self.setting.val)
        if len(rows) != 0:
            rows.append(rows[-1])
        else:
            rows.append(self.defaultval)
        self.emit( qt4.SIGNAL('settingChanged'), self, self.setting, rows )

    def onDeleteClicked(self):
        """Remove final entry in settings list."""

        rows = list(self.setting.val)[:-1]
        self.emit( qt4.SIGNAL('settingChanged'), self, self.setting, rows )

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
        wcolor = qt4.QPushButton()
        self.layout.addWidget(wcolor, row, col)
        wcolor.setMaximumWidth(wcolor.height())
        pix = qt4.QPixmap(self.pixsize, self.pixsize)
        pix.fill( qt4.QColor(color) )
        wcolor.setIcon( qt4.QIcon(pix) )
        wcolor.setToolTip(tooltip)
        wcolor.setSizePolicy(qt4.QSizePolicy.Maximum, qt4.QSizePolicy.Maximum)

        self.connect(wcolor, qt4.SIGNAL('clicked()'), self.onColorClicked)
        return wcolor

    def addToggleButton(self, row, col, tooltip):
        """Make a toggle button."""

        toggle = self.setting.val[row][col]
        wtoggle = qt4.QCheckBox()
        self.layout.addWidget(wtoggle, row, col)
        wtoggle.setChecked(toggle)
        wtoggle.setToolTip(tooltip)
        self.connect(wtoggle, qt4.SIGNAL('toggled(bool)'), self.onToggled)
        return wtoggle

    def addCombo(self, row, col, tooltip, values, icons, texts):
        """Make an enumeration combo - choose from a set of icons."""
        
        val = self.setting.val[row][col]

        wcombo = qt4.QComboBox()
        self.layout.addWidget(wcombo, row, col)

        if texts is None:
            for icon in icons:
                wcombo.addItem(icon, "")
        else:
            for text, icon in itertools.izip(texts, icons):
                wcombo.addItem(icon, text)

        wcombo.setCurrentIndex(values.index(val))
        wcombo.setToolTip(tooltip)
        self.connect(wcombo, qt4.SIGNAL('activated(int)'),
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
        self.emit( qt4.SIGNAL('settingChanged'), self, self.setting, rows )
        
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
        color = qt4.QColorDialog.getColor( qt4.QColor(rows[row][col]),
                                          self )
        if color.isValid():
            # change setting
            # this is a bit irritating, as have to do lots of
            # tedious conversions
            color = unicode(color.name())
            self._updateRowCol(row, col, color)

            # change the color
            pix = qt4.QPixmap(self.pixsize, self.pixsize)
            pix.fill(qt4.QColor(color))
            sender.setIcon( qt4.QIcon(pix) )
            
class LineSet(ListSet):
    """A list of line styles.
    """

    def __init__(self, setting, parent):
        ListSet.__init__(self, ('solid', '1pt', 'black', False),
                         setting, parent)

    def populateRow(self, row, val):
        """Add the widgets for the row given."""

        # create line icons if not already created
        if LineStyle._icons is None:
            LineStyle._generateIcons()

        # make line style selector
        wlinestyle = self.addCombo(row, 0, 'Line style',
                                   LineStyle._lines,
                                   LineStyle._icons, None)
        
        # make line width edit box
        wwidth = qt4.QLineEdit()
        self.layout.addWidget(wwidth, row, 1)
        wwidth.setText(self.setting.val[row][1])
        wwidth.setToolTip('Line width')
        self.connect(wwidth, qt4.SIGNAL('editingFinished()'),
                     self.onWidthChanged)
        self.bgcolor = wwidth.palette().color(qt4.QPalette.Window)

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
            sender.palette().setColor(qt4.QPalette.Window, self.bgcolor)
            self._updateRowCol(row, col, text)
        else:
            # invalid distance
            sender.palette().setColor(qt4.QPalette.Window, qt4.QColor('red'))

class FillSet(ListSet):
    """A list of fill settings."""

    def __init__(self, setting, parent):
        ListSet.__init__(self, ('solid', 'black', False),
                         setting, parent)

    def populateRow(self, row, val):
        """Add the widgets for the row given."""

        # construct fill icons if not already done
        if FillStyle._icons is None:
            FillStyle._generateIcons()
    
        # make fill style selector
        wfillstyle = self.addCombo(row, 0, 'Fill style',
                                   FillStyle._fills,
                                   FillStyle._icons,
                                   FillStyle._fills)
        wfillstyle.setMinimumWidth(self.pixsize)

        # make color selector button
        wcolor = self.addColorButton(row, 1, 'Fill color')

        # make hide checkbox
        whide = self.addToggleButton(row, 2, 'Hide fill')

        # return widgets
        return [wfillstyle, wcolor, whide]

class Datasets(qt4.QWidget):
    """A control for editing a list of datasets."""

    def __init__(self, setting, doc, dimensions, datatype, parent):
        """Construct widget as combination of LineEdit and PushButton
        for browsing."""

        qt4.QWidget.__init__(self, parent)
        self.setting = setting
        self.document = doc
        self.dimensions = dimensions
        self.datatype = datatype

        self.grid = layout = qt4.QGridLayout()
        layout.setHorizontalSpacing(0)
        self.setLayout(layout)

        self.controls = []
        self.last = ()
        self.lastdatasets = []
        # force updating to initialise
        self.onModified(True)
        self.setting.setOnModified(self.onModified)

    def makeRow(self):
        """Make new row at end"""
        combo = qt4.QComboBox()
        combo.setEditable(True)
        addbutton = qt4.QPushButton('+')
        addbutton.setFixedWidth(24)
        addbutton.setToolTip('Add another dataset')
        subbutton = qt4.QPushButton('-')
        subbutton.setToolTip('Remove dataset')
        subbutton.setFixedWidth(24)
        self.controls.append((combo, addbutton, subbutton))
        row = len(self.controls)-1

        self.grid.addWidget(combo, row, 0)
        self.grid.addWidget(addbutton, row, 1)
        self.grid.addWidget(subbutton, row, 2)

        self.connect(combo.lineEdit(), qt4.SIGNAL('editingFinished()'), 
                     lambda: self.datasetChanged(row))
        # if a different item is selected
        self.connect(combo, qt4.SIGNAL('activated(const QString&)'),
                     lambda x: self.datasetChanged(row))

        self.connect(addbutton, qt4.SIGNAL('clicked()'),
                     lambda: self.addPressed(row))
        self.connect(subbutton, qt4.SIGNAL('clicked()'),
                     lambda: self.subPressed(row))

        if len(self.controls) == 2:
            # enable first subtraction button
            self.controls[0][2].setEnabled(True)
        elif len(self.controls) == 1:
            # or disable
            self.controls[0][2].setEnabled(False)

    def deleteRow(self):
        """Remove last row"""
        for w in self.controls[-1]:
            self.grid.removeWidget(w)
        self.controls.pop(-1)

        # disable first subtraction button
        if len(self.controls) == 1:
            self.controls[0][2].setEnabled(False)

    def addPressed(self, row):
        """User adds a new row."""
        val = list(self.setting.val)
        val.insert(row+1, '')
        self.emit( qt4.SIGNAL('settingChanged'), self, self.setting, tuple(val) )

    def subPressed(self, row):
        """User deletes a row."""
        val = list(self.setting.val)
        val.pop(row)
        self.emit( qt4.SIGNAL('settingChanged'), self, self.setting, tuple(val) )

    def datasetChanged(self, row):
        """User enters some text."""
        val = list(self.setting.val)
        val[row] = unicode(self.controls[row][0].lineEdit().text())
        self.emit( qt4.SIGNAL('settingChanged'), self, self.setting, tuple(val) )

    def getDatasets(self):
        """Get applicable datasets (sorted)."""
        datasets = []
        for name, ds in self.document.data.iteritems():
            if (ds.dimensions == self.dimensions and
                ds.datatype == self.datatype):
                datasets.append(name)
        datasets.sort()
        return datasets

    def onModified(self, mod):
        """Called when the setting is changed remotely, or when control is opened"""

        s = self.setting
        datasets = self.getDatasets()

        if self.last == s.val and self.lastdatasets == datasets:
            return
        self.last = s.val
        self.lastdatasets = datasets
        
        while len(s.val) > len(self.controls):
            self.makeRow()
        while len(s.val) < len(self.controls):
            self.deleteRow()

        for cntrls, val in itertools.izip(self.controls, s.val):
            cntrls[0].lineEdit().setText(val)
            populateCombo(cntrls[0], datasets)

class Filename(qt4.QWidget):
    """A widget for selecting a filename with a browse button."""

    def __init__(self, setting, parent):
        """Construct widget as combination of LineEdit and PushButton
        for browsing."""

        qt4.QWidget.__init__(self, parent)
        self.setting = setting

        layout = qt4.QHBoxLayout()
        layout.setSpacing(0)
        layout.setMargin(0)
        self.setLayout(layout)

        # the actual edit control
        self.edit = qt4.QLineEdit()
        self.edit.setText( setting.toText() )
        layout.addWidget(self.edit)
        self.bgcolor = self.edit.palette().color(qt4.QPalette.Base)
        
        # get a sensible shape for the button - yawn
        b = self.button = qt4.QPushButton('..')
        layout.addWidget(b)
        b.setSizePolicy(qt4.QSizePolicy.Maximum, qt4.QSizePolicy.Maximum)
        b.setMaximumWidth(16)

        # connect up signals
        self.connect(self.edit, qt4.SIGNAL('editingFinished()'),
                     self.validateAndSet)
        self.connect(b, qt4.SIGNAL('clicked()'),
                     self.buttonClicked)

        # add completion if we have support (qt >= 4.3)
        if hasattr(qt4, 'QDirModel'):
            c = self.filenamecompleter = qt4.QCompleter(self)
            model = qt4.QDirModel(c)
            c.setModel(model)
            self.edit.setCompleter(c)

        # for read only filernames
        if setting.readonly:
            self.edit.setReadOnly(True)

        self.setting.setOnModified(self.onModified)

    def buttonClicked(self):
        """Button clicked - show file open dialog."""

        filename = qt4.QFileDialog.getOpenFileName(
            self,
            "Choose image",
            self.edit.text(),
            "Images (*.png *.jpg *.jpeg *.bmp *.svg *.tiff *.tif "
            "*.gif *.xbm *.xpm);;"
            "All files (*)")
        if filename:
            val = unicode(filename)
            if self.setting.val != val:
                self.emit( qt4.SIGNAL('settingChanged'), self, self.setting,
                           val )

    def validateAndSet(self):
        """Check the text is a valid setting and update it."""

        text = unicode(self.edit.text())
        try:
            val = self.setting.fromText(text)
            self.edit.palette().setColor(qt4.QPalette.Base, self.bgcolor)

            # value has changed
            if self.setting.val != val:
                self.emit( qt4.SIGNAL('settingChanged'), self, self.setting,
                           val )

        except setting.InvalidType:
            self.edit.palette().setColor(qt4.QPalette.Base, qt4.QColor('red'))

    def onModified(self, mod):
        """called when the setting is changed remotely"""
        self.edit.setText( self.setting.toText() )

class FontFamily(qt4.QFontComboBox):
    """List the font families, showing each font."""

    def __init__(self, setting, parent):
        """Create the combobox."""

        qt4.QFontComboBox.__init__(self, parent)
        self.setting = setting
        self.setFontFilters( qt4.QFontComboBox.ScalableFonts )
        
        # set initial value
        self.onModified(True)

        # stops combobox readjusting in size to fit contents
        self.setSizeAdjustPolicy(
            qt4.QComboBox.AdjustToMinimumContentsLengthWithIcon)

        self.setting.setOnModified(self.onModified)

        # if a different item is selected
        self.connect( self, qt4.SIGNAL('activated(const QString&)'),
                      self.slotActivated )

    def focusOutEvent(self, *args):
        """Allows us to check the contents of the widget."""
        qt4.QFontComboBox.focusOutEvent(self, *args)
        self.slotActivated('')

    def slotActivated(self, val):
        """Update setting if a different item is chosen."""
        newval = unicode(self.currentText())
        if self.setting.val != newval:
            self.emit(qt4.SIGNAL('settingChanged'), self, self.setting, newval)

    def onModified(self, mod):
        """Make control reflect chosen setting."""
        self.setCurrentFont( qt4.QFont(self.setting.toText()) )

class ErrorStyle(Choice):
    """Choose different error bar styles."""
    
    _icons = None         # generated icons
    _errorstyles = None   # copied in by setting.py
    
    def __init__(self, setting, parent):
        if self._icons is None:
            self._generateIcons()

        Choice.__init__(self, setting, False,
                        self._errorstyles, parent,
                        icons=self._icons)

    def _generateIcons(cls):
        """Generate a list of pixmaps for drop down menu."""
        cls._icons = []
        for errstyle in cls._errorstyles:
            cls._icons.append( utils.getIcon('error_%s.svg' % errstyle) )
