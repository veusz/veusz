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

"""Module for creating QWidgets for the settings, to enable their values
   to be changed.

    These widgets emit settingChanged(control, setting, val) when the setting is
    changed. The creator should use this to change the setting.
"""

import re
import numpy as N

from .. import qtall as qt
from .settingdb import settingdb
from .. import utils

def _(text, disambiguation=None, context="Setting"):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

def styleClear(widget):
    """Return widget to default"""
    widget.setStyleSheet("")

def styleError(widget):
    """Show error state on widget."""
    widget.setStyleSheet(
        "background-color: " + settingdb.color('error').name() )

class DotDotButton(qt.QPushButton):
    """A button for opening up more complex editor."""
    def __init__(self, tooltip=None, checkable=True):
        qt.QPushButton.__init__(
            self, "..", flat=True, checkable=checkable,
            maximumWidth=16, maximumHeight=16)
        if tooltip:
            self.setToolTip(tooltip)
        self.setSizePolicy(qt.QSizePolicy.Policy.Maximum, qt.QSizePolicy.Policy.Maximum)
        self.setStyleSheet('QPushButton { padding: 0; }')

class AddButton(qt.QPushButton):
    """A button to add item."""
    def __init__(self):
        qt.QPushButton.__init__(self, "+", flat=True)
        self.setFixedWidth(24)
        self.setToolTip('Add another item')
        self.setStyleSheet('QPushButton { padding: 0; }')

class SubButton(qt.QPushButton):
    """A button to subtract item."""
    def __init__(self):
        qt.QPushButton.__init__(self, "-", flat=True)
        self.setFixedWidth(24)
        self.setToolTip('Remove item')
        self.setStyleSheet('QPushButton { padding: 0; }')

class Edit(qt.QLineEdit):
    """Main control for editing settings which are text."""

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)

    def __init__(self, setting, parent):
        """Initialise the setting widget."""

        qt.QLineEdit.__init__(self, parent)
        self.setting = setting

        self.setText( setting.toUIText() )
        self.editingFinished.connect(self.validateAndSet)
        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.setReadOnly(True)

    def validateAndSet(self):
        """Check the text is a valid setting and update it."""

        text = self.text()
        try:
            val = self.setting.fromUIText(text)
            styleClear(self)
            self.sigSettingChanged.emit(self, self.setting, val)

        except utils.InvalidType:
            styleError(self)

    @qt.pyqtSlot()
    def onModified(self):
        """called when the setting is changed remotely"""
        self.setText( self.setting.toUIText() )

class _EditBox(qt.QTextEdit):
    """A popup edit box to support editing long text sections.

    Emits closing(text) when the box closes
    """

    closing = qt.pyqtSignal(str)

    def __init__(self, origtext, readonly, parent):
        """Make a popup, framed widget containing a text editor."""

        qt.QTextEdit.__init__(self, parent)
        self.setWindowFlags(qt.Qt.WindowType.Popup)
        self.setAttribute(qt.Qt.WidgetAttribute.WA_DeleteOnClose)

        self.spacing = self.fontMetrics().height()

        self.origtext = origtext
        self.setPlainText(origtext)

        cursor = self.textCursor()
        cursor.movePosition(qt.QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

        if readonly:
            self.setReadOnly(True)

        utils.positionFloatingPopup(self, parent)

        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Grab clicks outside this window to close it."""
        if ( isinstance(event, qt.QMouseEvent) and
             event.buttons() != qt.Qt.MouseButton.NoButton ):
            frame = qt.QRect(0, 0, self.width(), self.height())
            if not frame.contains(event.pos()):
                self.close()
                return True
        return qt.QTextEdit.eventFilter(self, obj, event)

    def keyPressEvent(self, event):
        """Close if escape or return is pressed."""
        qt.QTextEdit.keyPressEvent(self, event)

        key = event.key()
        if key == qt.Qt.Key.Key_Escape:
            # restore original content
            self.setPlainText(self.origtext)
            self.close()
        elif key == qt.Qt.Key.Key_Return:
            # keep changes
            self.close()

    def sizeHint(self):
        """A reasonable size for the text editor."""
        return qt.QSize(self.spacing*40, self.spacing*3)

    def closeEvent(self, event):
        """Tell the calling widget that we are closing, and provide
        the new text."""

        text = self.toPlainText()
        text = text.replace('\n', '')
        self.closing.emit(text)
        event.accept()

class String(qt.QWidget):
    """A line editor which allows editting in a larger popup window."""

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)

    def __init__(self, setting, parent):
        qt.QWidget.__init__(self, parent)
        self.setting = setting

        layout = qt.QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)

        self.edit = qt.QLineEdit()
        layout.addWidget(self.edit)

        b = self.button = DotDotButton(tooltip="Edit text")
        layout.addWidget(b)

        # set the text of the widget to the
        self.edit.setText( setting.toUIText() )

        self.edit.editingFinished.connect(self.validateAndSet)
        b.toggled.connect(self.buttonToggled)

        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.edit.setReadOnly(True)

    def buttonToggled(self, on):
        """Button is pressed to bring popup up / down."""

        # if button is down and there's no existing popup, bring up a new one
        if on:
            e = _EditBox(
                self.edit.text(), self.setting.readonly, self.button)

            # we get notified with text when the popup closes
            e.closing.connect(self.boxClosing)
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

        text = self.edit.text()
        try:
            val = self.setting.fromUIText(text)
            styleClear(self.edit)
            self.sigSettingChanged.emit(self, self.setting, val)

        except utils.InvalidType:
            styleError(self.edit)

    @qt.pyqtSlot()
    def onModified(self):
        """called when the setting is changed remotely"""
        self.edit.setText( self.setting.toUIText() )

class Int(qt.QSpinBox):
    """A control for changing an integer."""

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)

    def __init__(self, setting, parent):
        qt.QSpinBox.__init__(self, parent)

        self.ignorechange = False
        self.setting = setting
        self.setMinimum(setting.minval)
        self.setMaximum(setting.maxval)
        self.setValue(setting.val)

        self.valueChanged[int].connect(self.slotChanged)
        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.setEnabled(False)

    def slotChanged(self, value):
        """If check box changes."""
        # this is emitted by setValue, so ignore onModified doing this
        if not self.ignorechange:
            self.sigSettingChanged.emit(self, self.setting, value)

    @qt.pyqtSlot()
    def onModified(self):
        """called when the setting is changed remotely"""
        self.ignorechange = True
        self.setValue( self.setting.val )
        self.ignorechange = False

class FloatSlider(qt.QWidget):
    """A slider control for a numerical value.
    Note: QSlider is integer only, so dragging slider makes values integers
    """

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)

    def __init__(self, setting, parent):
        qt.QWidget.__init__(self, parent)
        self.setting = setting
        self.setting.setOnModified(self.onModified)

        layout = qt.QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        s = self.slider = qt.QSlider(qt.Qt.Orientation.Horizontal)
        s.setMinimum(int(setting.minval/setting.scale))
        s.setMaximum(int(setting.maxval/setting.scale))
        s.setPageStep(int(setting.step/setting.scale))
        s.setTickInterval(int(setting.tick/setting.scale))
        s.setTickPosition(qt.QSlider.TickPosition.TicksAbove)
        layout.addWidget(self.slider)

        self.edit = qt.QLineEdit()
        layout.addWidget(self.edit)

        self.edit.editingFinished.connect(self.validateAndSet)
        self.slider.valueChanged.connect(self.movedPosition)

        self.onModified()

    def validateAndSet(self):
        """Validate text is numeric."""
        try:
            val = self.setting.fromUIText(self.edit.text())
            styleClear(self.edit)
            self.sigSettingChanged.emit(self, self.setting, val)

        except utils.InvalidType:
            styleError(self.edit)

    def movedPosition(self, val):
        """Someone dragged the slider."""
        self.sigSettingChanged.emit(
            self, self.setting, float(val)*self.setting.scale)

    @qt.pyqtSlot()
    def onModified(self):
        self.edit.setText(self.setting.toUIText())
        self.slider.setValue(int(round(
            self.setting.get()/self.setting.scale)))

class Bool(qt.QCheckBox):
    """A check box for changing a bool setting."""

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)

    def __init__(self, setting, parent):
        qt.QCheckBox.__init__(self, parent)

        self.setSizePolicy( qt.QSizePolicy(
            qt.QSizePolicy.Policy.MinimumExpanding, qt.QSizePolicy.Policy.Fixed) )

        self.ignorechange = False
        self.setting = setting
        self.setChecked(setting.val)

        # we get a signal when the button is toggled
        self.toggled.connect(self.slotToggled)

        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.setEnabled(False)

    def slotToggled(self, state):
        """Emitted when checkbox toggled."""
        # this is emitted by setChecked, so ignore onModified doing this
        if not self.ignorechange:
            self.sigSettingChanged.emit(self, self.setting, state)

    @qt.pyqtSlot()
    def onModified(self):
        """called when the setting is changed remotely"""
        self.ignorechange = True
        self.setChecked( self.setting.val )
        self.ignorechange = False

class BoolSwitch(Bool):
    """Bool for switching off/on other settings."""

    def showEvent(self, event):
        Bool.showEvent(self, event)
        self.updateState()

    def slotToggled(self, state):
        Bool.slotToggled(self, state)
        self.updateState()

    def updateState(self):
        """Set hidden state of settings."""
        s1, s2 = self.setting.strue, self.setting.sfalse
        if self.setting.val:
            show, hide = s1, s2
        else:
            show, hide = s2, s1

        if hasattr(self.parent(), 'showHideSettings'):
            self.parent().showHideSettings(show, hide)

class Choice(qt.QComboBox):
    """For choosing between a set of values."""

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)

    def __init__(self, setting, iseditable, vallist, parent, icons=None,
                 uilist=None,
                 descriptions=None):
        qt.QComboBox.__init__(self, parent)

        self.setting = setting
        self.vallist = vallist
        self.uilist = uilist
        self.setEditable(iseditable)

        # stops combobox readjusting in size to fit contents
        self.setSizeAdjustPolicy(
            qt.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)

        # whether to show ui text to replace some items
        toadd = uilist if uilist is not None else vallist

        if icons is None:
            # add items to list (text only)
            self.addItems( list(toadd) )
        else:
            # add pixmaps and text to list
            for icon, text in zip(icons, toadd):
                self.addItem(icon, text)

        # use tooltip descriptions if requested
        if descriptions is not None:
            for i, descr in enumerate(descriptions):
                self.setItemData(i, descr, qt.Qt.ItemDataRole.ToolTipRole)

        # choose the correct setting
        try:
            index = list(vallist).index(setting.toUIText())
            self.setCurrentIndex(index)
        except ValueError:
            # for cases when this is editable
            # set the text of the widget to the setting
            assert iseditable
            self.setEditText( setting.toUIText() )

        # if a different item is selected
        self.textActivated[str].connect(self.slotActivated)

        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.setEnabled(False)

        # make completion case sensitive (to help fix case typos)
        if self.completer():
            self.completer().setCaseSensitivity(qt.Qt.CaseSensitivity.CaseSensitive)

    def focusOutEvent(self, *args):
        """Allows us to check the contents of the widget."""
        qt.QComboBox.focusOutEvent(self, *args)
        self.slotActivated('')

    def slotActivated(self, val):
        """If a different item is chosen."""

        text = self.currentText()

        # convert to ui text if set
        if self.uilist is not None:
            idx = self.uilist.index(text)
            if idx >= 0:
                text = self.vallist[idx]

        try:
            val = self.setting.fromUIText(text)
            styleClear(self)
            self.sigSettingChanged.emit(self, self.setting, val)

        except utils.InvalidType:
            styleError(self)

    @qt.pyqtSlot()
    def onModified(self):
        """called when the setting is changed remotely"""
        text = self.setting.toUIText()

        # convert to ui text
        if self.uilist is not None:
            idx = self.vallist.index(text)
            if idx >= 0:
                text = self.uilist[idx]

        index = self.findText(text)
        if index >= 0:
            self.setCurrentIndex(index)
        if self.isEditable():
            self.setEditText(text)

class ChoiceSwitch(Choice):
    """Show or hide other settings based on value."""

    def showEvent(self, event):
        Choice.showEvent(self, event)
        self.updateState()

    @qt.pyqtSlot()
    def onModified(self):
        """called when the setting is changed remotely"""
        Choice.onModified(self)
        self.updateState()

    def updateState(self):
        """Set hidden state of settings."""
        show, hide = self.setting.showfn(self.setting.val)
        if hasattr(self.parent(), 'showHideSettings'):
            self.parent().showHideSettings(show, hide)

class FillStyleExtended(ChoiceSwitch):
    """Extended fill style list."""

    _icons = None

    def __init__(self, setting, parent):
        if self._icons is None:
            self._generateIcons()

        ChoiceSwitch.__init__(
            self, setting, False,
            utils.extfillstyles, parent,
            icons=self._icons
        )

    @classmethod
    def _generateIcons(cls):
        """Generate a list of pixmaps for drop down menu."""

        from .. import document
        from . import collections
        brush = collections.BrushExtended("")
        brush.color = 'black'
        brush.patternspacing = '5pt'
        brush.linewidth = '0.5pt'

        size = 12
        cls._icons = icons = []

        path = qt.QPainterPath()
        path.addRect(0, 0, size, size)

        doc = document.Document()
        phelper = document.PaintHelper(doc, (1,1))

        for f in utils.extfillstyles:
            pix = qt.QPixmap(size, size)
            pix.fill()
            painter = document.DirectPainter(pix)
            painter.setRenderHint(qt.QPainter.RenderHint.Antialiasing)
            painter.updateMetaData(phelper)
            brush.style = f
            utils.brushExtFillPath(painter, brush, path)
            painter.end()
            icons.append(qt.QIcon(pix))

class MultiLine(qt.QTextEdit):
    """For editting multi-line settings."""

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)

    def __init__(self, setting, parent):
        """Initialise the widget."""

        qt.QTextEdit.__init__(self, parent)
        self.setting = setting

        self.setWordWrapMode(qt.QTextOption.WrapMode.NoWrap)
        self.setTabChangesFocus(True)

        # set the text of the widget to the
        self.setPlainText( setting.toUIText() )

        self.setting.setOnModified(self.onModified)

        if setting.readonly:
            self.setReadOnly(True)

        self.document().contentsChanged.connect(self.onSizeChange)
        self.document().documentLayout().documentSizeChanged.connect(
            self.onSizeChange)

        self.heightmin = 0
        self.heightmax = 2048

        # recalculate size of document to fix size
        self.document().adjustSize()
        self.onSizeChange()

    def onSizeChange(self):
        """Make size match content size."""
        m = self.contentsMargins()
        docheight = self.document().size().height() + m.top() + m.bottom()
        docheight = min(self.heightmax, max(self.heightmin, docheight))
        self.setFixedHeight(int(docheight))

    def focusOutEvent(self, *args):
        """Allows us to check the contents of the widget."""
        qt.QTextEdit.focusOutEvent(self, *args)

        text = self.toPlainText()
        try:
            val = self.setting.fromUIText(text)
            styleClear(self)
            self.sigSettingChanged.emit(self, self.setting, val)

        except utils.InvalidType:
            styleError(self)

    @qt.pyqtSlot()
    def onModified(self):
        """called when the setting is changed remotely"""
        self.setPlainText( self.setting.toUIText() )

class Notes(MultiLine):
    """For editing notes."""

    def __init__(self, setting, parent):
        MultiLine.__init__(self, setting, parent)
        self.setWordWrapMode(qt.QTextOption.WrapMode.WordWrap)

class Distance(Choice):
    """For editing distance settings."""

    # used to remove non-numerics from the string
    # we also remove X/ from X/num
    stripnumre = re.compile(r"[0-9]*/|[^0-9.,]")

    # remove spaces
    stripspcre = re.compile(r"\s")

    def __init__(self, setting, parent, allowauto=False, physical=False):
        '''Initialise with blank list, then populate with sensible units.'''
        Choice.__init__(self, setting, True, [], parent)
        self.allowauto = allowauto
        self.physical = physical
        self.updateComboList()

    def updateComboList(self):
        '''Populates combo list with sensible list of other possible units.'''

        # turn off signals, so our modifications don't create more signals
        self.blockSignals(True)

        # get current text
        text = self.currentText()

        # get rid of non-numeric things from the string
        num = self.stripnumre.sub('', text)

        # here are a list of possible different units the user can choose
        # between. should this be in utils?
        newitems = [ num+'pt', num+'cm', num+'mm',
                     num+'in' ]
        if not self.physical:
            newitems += [ num+'%', '1/'+num ]

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

class DistancePt(Choice):
    """For editing distances with defaults in points."""

    points = (
        '0pt', '0.25pt', '0.5pt', '1pt', '1.5pt', '2pt', '3pt',
        '4pt', '5pt', '6pt', '8pt', '10pt', '12pt', '14pt', '16pt',
        '18pt', '20pt', '22pt', '24pt', '26pt', '28pt', '30pt',
        '34pt', '40pt', '44pt', '50pt', '60pt', '70pt'
    )

    def __init__(self, setting, parent, allowauto=False):
        '''Initialise with blank list, then populate with sensible units.'''
        Choice.__init__(self, setting, True, DistancePt.points, parent)

class Dataset(qt.QWidget):
    """Allow the user to choose between the possible datasets."""

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)

    def __init__(self, setting, document, dimensions, datatype, parent):
        """Initialise the combobox. The list is populated with datasets.

        dimensions specifies the dimension of the dataset to list

        Changes on the document refresh the list of datasets."""

        qt.QWidget.__init__(self, parent)

        self.choice = Choice(setting, True, [], None)
        self.choice.sigSettingChanged.connect(self.sigSettingChanged)

        b = self.button = DotDotButton(tooltip=_("Select using dataset browser"))
        b.toggled.connect(self.slotButtonToggled)

        self.document = document
        self.dimensions = dimensions
        self.datatype = datatype
        self.lastdatasets = None
        self._populateEntries()
        document.signalModified.connect(self.slotModified)

        layout = qt.QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.choice)
        layout.addWidget(b)
        self.setLayout(layout)

    def _populateEntries(self):
        """Put the list of datasets into the combobox."""

        # get datasets of the correct dimension
        datasets = []
        for name, ds in self.document.data.items():
            okdims = (
                ds.dimensions == self.dimensions or
                self.dimensions == 'all'
            )
            oktype = (
                ds.datatype == self.datatype or
                self.datatype == 'all' or
                ds.datatype in self.datatype
            )

            if okdims and oktype:
                datasets.append(name)

        datasets.sort()

        if datasets != self.lastdatasets:
            utils.populateCombo(self.choice, datasets)
            self.lastdatasets = datasets

    @qt.pyqtSlot(int)
    def slotModified(self, modified):
        """Update the list of datasets if the document is modified."""
        self._populateEntries()

    def slotButtonToggled(self, on):
        """Bring up list of datasets."""
        if on:
            from ..qtwidgets.datasetbrowser import DatasetBrowserPopup
            d = DatasetBrowserPopup(
                self.document,
                self.choice.currentText(),
                self.button,
                filterdims=set((self.dimensions,)),
                filterdtype=set((self.datatype,)) )
            d.closing.connect(self.boxClosing)
            d.newdataset.connect(self.newDataset)
            d.show()

    def boxClosing(self):
        """Called when the popup edit box closes."""
        self.button.setChecked(False)

    def newDataset(self, dsname):
        """New dataset selected."""
        self.sigSettingChanged.emit(self, self.choice.setting, dsname)

class DatasetOrString(Dataset):
    """Allow use to choose a dataset or enter some text."""

    def __init__(self, setting, document, parent):
        Dataset.__init__(self, setting, document, 1, 'all', parent)

        b = self.textbutton = DotDotButton()
        self.layout().addWidget(b)
        b.toggled.connect(self.textButtonToggled)

    def textButtonToggled(self, on):
        """Button is pressed to bring popup up / down."""

        # if button is down and there's no existing popup, bring up a new one
        if on:
            e = _EditBox(
                self.choice.currentText(),
                self.choice.setting.readonly, self.textbutton)

            # we get notified with text when the popup closes
            e.closing.connect(self.textBoxClosing)
            e.show()

    def textBoxClosing(self, text):
        """Called when the popup edit box closes."""

        self.textbutton.setChecked(False)

        # update the text if we can
        if not self.choice.setting.readonly:
            self.choice.setEditText(text)
            self.choice.setFocus()
            self.parentWidget().setFocus()
            self.choice.setFocus()

class FillStyle(Choice):
    """For choosing between fill styles."""

    _icons = None
    _fills = None
    _fillcnvt = None

    def __init__(self, setting, parent):
        if self._icons is None:
            self._generateIcons()

        Choice.__init__(
            self, setting, False,
            self._fills, parent,
            icons=self._icons
        )

    @classmethod
    def _generateIcons(cls):
        """Generate a list of pixmaps for drop down menu."""

        size = 12
        icons = []
        c = qt.QColor('grey')
        for f in cls._fills:
            pix = qt.QPixmap(size, size)
            pix.fill()
            painter = qt.QPainter(pix)
            painter.setRenderHint(qt.QPainter.RenderHint.Antialiasing)
            brush = qt.QBrush(c, cls._fillcnvt[f])
            painter.fillRect(0, 0, size, size, brush)
            painter.end()
            icons.append( qt.QIcon(pix) )

        cls._icons = icons

class Marker(Choice):
    """A control to let the user choose a marker."""

    _icons = None

    def __init__(self, setting, parent):
        if self._icons is None:
            self._generateIcons()

        Choice.__init__(
            self, setting, False,
            utils.MarkerCodes, parent,
            icons=self._icons
        )

    @classmethod
    def _generateIcons(cls):
        size = 16
        icons = []
        brush = qt.QBrush( qt.QColor('darkgrey') )
        pen = qt.QPen( qt.QBrush(qt.Qt.GlobalColor.black), 1. )
        for marker in utils.MarkerCodes:
            pix = qt.QPixmap(size, size)
            pix.fill()
            painter = qt.QPainter(pix)
            painter.setRenderHint(qt.QPainter.RenderHint.Antialiasing)
            painter.setBrush(brush)
            painter.setPen(pen)
            utils.plotMarker(painter, size*0.5, size*0.5, marker, size*0.33)
            painter.end()
            icons.append( qt.QIcon(pix) )

        cls._icons = icons

class Arrow(Choice):
    """A control to let the user choose an arrowhead."""

    _icons = None

    def __init__(self, setting, parent):
        if self._icons is None:
            self._generateIcons()

        Choice.__init__(
            self, setting, False,
            utils.ArrowCodes, parent,
            icons=self._icons
        )

    @classmethod
    def _generateIcons(cls):
        size = 16
        icons = []
        brush = qt.QBrush(qt.Qt.GlobalColor.black)
        pen = qt.QPen( qt.QBrush(qt.Qt.GlobalColor.black), 1. )
        for arrow in utils.ArrowCodes:
            pix = qt.QPixmap(size, size)
            pix.fill()
            painter = qt.QPainter(pix)
            painter.setRenderHint(qt.QPainter.RenderHint.Antialiasing)
            painter.setBrush(brush)
            painter.setPen(pen)
            utils.plotLineArrow(
                painter, size*0.4, size*0.5,
                size*2, 0.,
                arrowsize=size*0.2,
                arrowleft=arrow, arrowright=arrow)
            painter.end()
            icons.append( qt.QIcon(pix) )

        cls._icons = icons

class LineStyle(Choice):
    """For choosing between line styles."""

    _icons = None
    _lines = None
    _linecnvt = None

    size = (24, 8)

    def __init__(self, setting, parent):
        if self._icons is None:
            self._generateIcons()

        Choice.__init__(
            self, setting, False,
            self._lines, parent,
            icons=self._icons
        )
        self.setIconSize( qt.QSize(*self.size) )

    @classmethod
    def _generateIcons(cls):
        """Generate a list of icons for drop down menu."""

        # import later for dependency issues
        from . import collections
        from .. import document

        icons = []
        size = cls.size
        setn = collections.Line('temp')
        setn.get('color').set('black')
        setn.get('width').set('1pt')

        doc = document.Document()
        for lstyle in cls._lines:
            pix = qt.QPixmap(*size)
            pix.fill()

            painter = document.DirectPainter(pix)
            painter.setRenderHint(qt.QPainter.RenderHint.Antialiasing)

            phelper = document.PaintHelper(doc, (1, 1))
            painter.updateMetaData(phelper)

            setn.get('style').set(lstyle)

            painter.setPen( setn.makeQPen(painter) )
            painter.drawLine(
                int(size[0]*0.1), size[1]//2,
                int(size[0]*0.9), size[1]//2)
            painter.end()
            icons.append( qt.QIcon(pix) )

        cls._icons = icons

class _ColNotifier(qt.QObject):
    sigNewColor = qt.pyqtSignal(str)


class Color(qt.QWidget):
    """A control which lets the user choose a color.

    A drop down list and a button to bring up a dialog are used
    """

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)

    def __init__(self, setting, parent):
        qt.QWidget.__init__(self, parent)

        self.setting = setting
        self.document = setting.getDocument()
        self.colors = self.document.evaluate.colors

        # combo box
        c = self.combo = qt.QComboBox()
        c.setEditable(True)
        c.textActivated[str].connect(self.slotActivated)

        # button for selecting colors
        b = self.button = qt.QPushButton()
        b.setFlat(True)
        b.setSizePolicy(qt.QSizePolicy.Policy.Maximum, qt.QSizePolicy.Policy.Maximum)
        b.setMaximumHeight(24)
        b.setMaximumWidth(24)
        b.clicked.connect(self.slotButtonClicked)

        c.setModel(self.colors.model)
        self.setColor(self.setting.val)

        if setting.readonly:
            c.setEnabled(False)
            b.setEnabled(False)

        layout = qt.QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(c)
        layout.addWidget(b)

        self.setColor(setting.toUIText())
        self.setLayout(layout)
        self.setting.setOnModified(self.onModified)

    def slotButtonClicked(self):
        """Open dialog to edit color."""

        col = qt.QColorDialog.getColor(
            self.colors.get(self.setting.val), self)
        if col.isValid():
            # change setting
            val = col.name()
            self.sigSettingChanged.emit(self, self.setting, val)

    def slotActivated(self, text):
        """A different value is selected."""
        val = self.setting.fromUIText(text)
        self.sigSettingChanged.emit(self, self.setting, val)

    def setColor(self, color):
        """Update control with color given."""

        index = self.combo.findText(color)
        if index < 0:
            # add text to combo if not there
            self.combo.addItem(color)
            index = self.combo.findText(color)

        self.combo.setCurrentIndex(index)

        icon = self.combo.itemData(index, qt.Qt.ItemDataRole.DecorationRole)
        self.button.setIcon(icon)

    @qt.pyqtSlot()
    def onModified(self):
        """called when the setting is changed remotely"""
        self.setColor( self.setting.toUIText() )

class WidgetSelector(Choice):
    """For choosing from a list of widgets."""

    def __init__(self, setting, document, parent):
        """Initialise and populate combobox."""

        Choice.__init__(self, setting, True, [], parent)
        self.document = document
        document.signalModified.connect(self.slotModified)

    def _populateEntries(self):
        pass

    @qt.pyqtSlot(int)
    def slotModified(self, modified):
        """Update list of axes."""
        self._populateEntries()

class WidgetChoice(WidgetSelector):
    """Choose a widget."""

    def __init__(self, setting, document, parent):
        """Initialise and populate combobox."""

        WidgetSelector.__init__(self, setting, document, parent)
        self._populateEntries()

    def _populateEntries(self):
        """Build up a list of widgets for combobox."""

        widgets = self.setting.getWidgetList()

        # we only need the list of names
        names = list(widgets.keys())
        names.sort()

        utils.populateCombo(self, names)

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
        while not widget.iswidget and widget is not None:
            widget = widget.parent

        # get list of axis widgets up the tree
        axes = set()
        while widget is not None:
            for w in widget.children:
                if ( w.isaxis and (
                        self.direction == 'both' or
                        w.settings.direction == self.direction) ):
                    axes.add(w.name)
            widget = widget.parent

        names = sorted(axes)
        utils.populateCombo(self, names)

class ListSet(qt.QFrame):
    """A widget for constructing settings which are lists of other
    properties.

    This code is pretty nasty and horrible, so we abstract it in this
    base widget
    """

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)

    pixsize = 12

    def __init__(self, defaultval, setting, parent):
        """Initialise this base widget.

        defaultval is the default entry to add if add is clicked with
        no current entries

        setting is the setting this widget corresponds to

        parent is the parent widget.
        """

        qt.QFrame.__init__(self, parent)
        self.setFrameStyle(qt.QFrame.Shape.Box)
        self.defaultval = defaultval
        self.setting = setting
        self.controls = []
        self.layout = qt.QGridLayout(self)
        s = self.layout.contentsMargins().left()//2
        self.layout.setContentsMargins(s,s,s,s)
        self.layout.setSpacing( self.layout.spacing()//4 )

        self.doneinit = False
        self.onModified()
        self.setting.setOnModified(self.onModified)

    def populateRow(self, row):
        """Populate the row in the control.

        Returns a list of the widgets created.
        """
        return []

    def updateRow(self, cntrls, val):
        """Set controls on row, given values for row, val."""
        pass

    def populate(self):
        """Construct the list of controls."""

        if len(self.setting.val) == len(self.controls) and self.doneinit:
            # no change in number of controls
            return
        self.doneinit = True   # we need to add the add/remove buttons below

        # delete all children in case of refresh
        self.controls = []
        for c in self.children():
            if isinstance(c, qt.QWidget):
                self.layout.removeWidget(c)
                c.deleteLater()
        c = None

        # iterate over each row
        row = -1
        for row, val in enumerate(self.setting.val):
            cntrls = self.populateRow(row)
            self.updateRow(cntrls, val)
            for col in range(len(cntrls)):
                self.layout.addWidget(cntrls[col], row, col)
            for c in cntrls:
                c.show()
            self.controls.append(cntrls)

        # buttons at end
        bbox = qt.QWidget()
        h = qt.QHBoxLayout(bbox)
        h.setContentsMargins(0,0,0,0)
        bbox.setLayout(h)
        self.layout.addWidget(bbox, row+1, 0, 1, -1)

        # a button to add a new entry
        b = qt.QPushButton('Add')
        h.addWidget(b)
        b.clicked.connect(self.onAddClicked)
        b.show()

        # a button to delete the last entry
        b = qt.QPushButton('Delete')
        h.addWidget(b)
        b.clicked.connect(self.onDeleteClicked)
        b.setEnabled(len(self.setting.val) > 0)
        b.show()

    def onAddClicked(self):
        """Add a line style to the list given."""

        rows = list(self.setting.val)
        if len(rows) != 0:
            rows.append(rows[-1])
        else:
            rows.append(self.defaultval)
        self.sigSettingChanged.emit(self, self.setting, rows)

    def onDeleteClicked(self):
        """Remove final entry in settings list."""

        rows = list(self.setting.val)[:-1]
        self.sigSettingChanged.emit(self, self.setting, rows)

    @qt.pyqtSlot()
    def onModified(self):
        """called when the setting is changed remotely"""
        self.populate()
        for cntrls, row in zip(self.controls, self.setting.val):
            self.updateRow(cntrls, row)

    def identifyPosn(self, widget):
        """Identify the position this widget is in.

        Returns (row, col) or (None, None) if not found.
        """

        for row, cntrls in enumerate(self.controls):
            for col, cntrl in enumerate(cntrls):
                if cntrl == widget:
                    return (row, col)
        return (None, None)

    def addColorButton(self, tooltip):
        """Add a color button to the list at the position specified."""
        wcolor = qt.QPushButton()
        wcolor.setFlat(True)
        wcolor.setSizePolicy(qt.QSizePolicy.Policy.Maximum, qt.QSizePolicy.Policy.Maximum)
        wcolor.setMaximumHeight(24)
        wcolor.setMaximumWidth(24)
        wcolor.setToolTip(tooltip)
        wcolor.clicked.connect(self.onColorClicked)
        return wcolor

    def updateColorButton(self, cntrl, color):
        """Given color control, update color."""

        pix = qt.QPixmap(self.pixsize, self.pixsize)
        qcolor = self.setting.getDocument().evaluate.colors.get(color)
        pix.fill(qcolor)
        cntrl.setIcon(qt.QIcon(pix))

    def addToggleButton(self, tooltip):
        """Make a toggle button."""
        wtoggle = qt.QCheckBox()
        wtoggle.setToolTip(tooltip)
        wtoggle.toggled.connect(self.onToggled)
        return wtoggle

    def updateToggleButton(self, cntrl, val):
        """Update toggle with value."""
        cntrl.setChecked(val)

    def addCombo(self, tooltip, values, icons, texts):
        """Make an enumeration combo - choose from a set of icons."""

        wcombo = qt.QComboBox()
        wcombo.setToolTip(tooltip)
        wcombo.activated[int].connect(self.onComboChanged)

        if texts is None:
            for icon in icons:
                wcombo.addItem(icon, "")
        else:
            for text, icon in zip(texts, icons):
                wcombo.addItem(icon, text)

        wcombo._vz_values = values
        return wcombo

    def updateCombo(self, cntrl, val):
        """Update selected item in combo."""
        cntrl.setCurrentIndex(cntrl._vz_values.index(val))

    def _updateRowCol(self, row, col, val):
        """Update value on row and column."""
        rows = list(self.setting.val)
        items = list(rows[row])
        items[col] = val
        rows[row] = tuple(items)
        self.sigSettingChanged.emit(self, self.setting, rows)

    def onToggled(self, on):
        """Checkbox toggled."""
        row, col = self.identifyPosn(self.sender())
        if row is not None:
            self._updateRowCol(row, col, on)

    def onComboChanged(self, val):
        """Update the setting if the combo changes."""
        sender = self.sender()
        row, col = self.identifyPosn(sender)
        if row is not None:
            self._updateRowCol(row, col, sender._vz_values[val])

    def onColorClicked(self):
        """Color button clicked for line."""
        sender = self.sender()
        row, col = self.identifyPosn(sender)

        rows = self.setting.val
        qcolor = self.setting.getDocument().evaluate.colors.get(
            rows[row][col])
        color = qt.QColorDialog.getColor(
            qcolor,
            self,
            "Choose color",
            qt.QColorDialog.ColorDialogOption.ShowAlphaChannel )
        if color.isValid():
            # change setting
            # this is a bit irritating, as have to do lots of
            # tedious conversions
            color = utils.extendedColorFromQColor(color)
            self._updateRowCol(row, col, color)

            # change the color
            pix = qt.QPixmap(self.pixsize, self.pixsize)
            qcolor = self.setting.getDocument().evaluate.colors.get(color)
            pix.fill(qcolor)
            sender.setIcon(qt.QIcon(pix))

class LineSet(ListSet):
    """A list of line styles.
    """

    def __init__(self, setting, parent):
        ListSet.__init__(
            self, ('solid', '1pt', 'black', False), setting, parent)

    def populateRow(self, row):
        """Add the widgets for the row given."""

        # create line icons if not already created
        if LineStyle._icons is None:
            LineStyle._generateIcons()

        # make line style selector
        wlinestyle = self.addCombo(
            _('Line style'), LineStyle._lines, LineStyle._icons, None)

        # make line width edit box
        wwidth = qt.QLineEdit()
        wwidth.setToolTip(_('Line width'))
        wwidth.editingFinished.connect(self.onWidthChanged)

        # make color selector button
        wcolor = self.addColorButton(_('Line color'))

        # make hide checkbox
        whide = self.addToggleButton( _('Hide line'))

        # return created controls
        return [wlinestyle, wwidth, wcolor, whide]

    def updateRow(self, cntrls, val):
        """Update controls with row settings."""
        self.updateCombo(cntrls[0], val[0])
        cntrls[1].setText(val[1])
        self.updateColorButton(cntrls[2], val[2])
        self.updateToggleButton(cntrls[3], val[3])

    def onWidthChanged(self):
        """Width has changed - validate."""

        sender = self.sender()
        row, col = self.identifyPosn(sender)

        text = sender.text()
        from . import setting
        if setting.Distance.isDist(text):
            # valid distance
            styleClear(sender)
            self._updateRowCol(row, col, text)
        else:
            # invalid distance
            styleError(sender)

class _FillBox(qt.QScrollArea):
    """Pop up box for extended fill settings."""

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)
    closing = qt.pyqtSignal(int)

    def __init__(self, doc, thesetting, row, button, parent):
        """Initialse widget. This is based on a PropertyList widget.

        FIXME: we have to import at runtime, so we should improve
        the inheritance here. Is using PropertyList window a hack?
        """

        qt.QScrollArea.__init__(self, parent)
        self.setWindowFlags(qt.Qt.WindowType.Popup)
        self.setAttribute(qt.Qt.WidgetAttribute.WA_DeleteOnClose)
        self.parent = parent
        self.row = row
        self.setting = thesetting

        self.extbrush = thesetting.returnBrushExtended(row)

        # need to add a real parent, so that the colors can be resolved
        self.extbrush.parent = thesetting.parent

        from ..windows.treeeditwindow import SettingsProxySingle, PropertyList

        fbox = self
        class DirectSetProxy(SettingsProxySingle):
            """Class to intercept changes of settings from UI."""
            def onSettingChanged(self, control, setting, val):
                # set value in setting
                setting.val = val
                # tell box to update setting
                fbox.onSettingChanged()

        # actual widget for changing the fill
        plist = PropertyList(doc)
        plist.updateProperties( DirectSetProxy(doc, self.extbrush) )
        self.setWidget(plist)

        utils.positionFloatingPopup(self, button)
        self.installEventFilter(self)

    def onSettingChanged(self):
        """Called when user changes a fill property."""

        # get value of brush and get data for row
        e = self.extbrush
        rowdata = [e.style, e.color, e.hide]
        if e.style != 'solid' or e.transparency > 0:
            rowdata += [
                e.transparency, e.linewidth, e.linestyle,
                e.patternspacing, e.backcolor,
                e.backtransparency, e.backhide ]
        rowdata = tuple(rowdata)

        if self.setting.val[self.row] != rowdata:
            # if row different, send update signal
            val = list(self.setting.val)
            val[self.row] = rowdata
            self.sigSettingChanged.emit(self, self.setting, val)

    def eventFilter(self, obj, event):
        """Grab clicks outside this window to close it."""
        if ( isinstance(event, qt.QMouseEvent) and
             event.buttons() != qt.Qt.MouseButton.NoButton ):
            frame = qt.QRect(0, 0, self.width(), self.height())
            if not frame.contains(event.pos()):
                self.close()
                return True
        return qt.QScrollArea.eventFilter(self, obj, event)

    def keyPressEvent(self, event):
        """Close if escape or return is pressed."""
        qt.QScrollArea.keyPressEvent(self, event)

        key = event.key()
        if key == qt.Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, event):
        """Tell the calling widget that we are closing, and provide
        the new text."""

        self.closing.emit(self.row)
        qt.QScrollArea.closeEvent(self, event)

class FillSet(ListSet):
    """A list of fill settings."""

    def __init__(self, setting, parent):
        ListSet.__init__(
            self, ('solid', 'black', False), setting, parent)

    def populateRow(self, row):
        """Add the widgets for the row given."""

        # construct fill icons if not already done
        if FillStyleExtended._icons is None:
            FillStyleExtended._generateIcons()

        # make fill style selector
        wfillstyle = self.addCombo(
            _("Fill style"),
            utils.extfillstyles, FillStyleExtended._icons, utils.extfillstyles)
        wfillstyle.setMinimumWidth(self.pixsize)

        # make color selector button
        wcolor = self.addColorButton(_("Fill color"))

        # make hide checkbox
        whide = self.addToggleButton(_("Hide fill"))

        # extended options
        wmore = DotDotButton(tooltip=_("More options"))
        wmore.toggled.connect(lambda on, row=row: self.editMore(on, row))

        # return widgets
        return [wfillstyle, wcolor, whide, wmore]

    def updateRow(self, cntrls, val):
        self.updateCombo(cntrls[0], val[0])
        self.updateColorButton(cntrls[1], val[1])
        self.updateToggleButton(cntrls[2], val[2])

    def buttonAtRow(self, row):
        """Get .. button on row."""
        return self.layout.itemAtPosition(row, 3).widget()

    def editMore(self, on, row):
        if on:
            fb = _FillBox(
                self.setting.getDocument(), self.setting,
                row, self.buttonAtRow(row), self.parent())
            fb.closing.connect(self.boxClosing)
            fb.sigSettingChanged.connect(self.sigSettingChanged)
            fb.show()

    def boxClosing(self, row):
        """Called when the popup edit box closes."""
        # uncheck the .. button
        self.buttonAtRow(row).setChecked(False)

class MultiSettingWidget(qt.QWidget):
    """A widget for storing multiple values in a tuple,
    with + and - signs by each entry."""

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)

    def __init__(self, setting, doc, *args):
        """Construct widget as combination of LineEdit and PushButton
        for browsing."""

        qt.QWidget.__init__(self, *args)
        self.setting = setting
        self.document = doc

        self.grid = layout = qt.QGridLayout()
        layout.setHorizontalSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)

        self.last = ()
        self.controls = []
        self.setting.setOnModified(self.onModified)

    def makeRow(self):
        """Make new row at end"""
        row = len(self.controls)
        cntrl = self.makeControl(row)
        cntrl.installEventFilter(self)
        addbutton = AddButton()
        subbutton = SubButton()

        self.controls.append((cntrl, addbutton, subbutton))

        self.grid.addWidget(cntrl, row, 0)
        self.grid.addWidget(addbutton, row, 1)
        self.grid.addWidget(subbutton, row, 2)

        addbutton.clicked.connect(lambda: self.addPressed(row))
        subbutton.clicked.connect(lambda: self.subPressed(row))

        if len(self.controls) == 2:
            # enable first subtraction button
            self.controls[0][2].setEnabled(True)
        elif len(self.controls) == 1:
            # or disable
            self.controls[0][2].setEnabled(False)

    def eventFilter(self, obj, event):
        """Capture loss of focus by controls."""
        if event.type() == qt.QEvent.Type.FocusOut:
            for row, c in enumerate(self.controls):
                if c[0] is obj:
                    self.dataChanged(row)
                    break
        return qt.QWidget.eventFilter(self, obj, event)

    def deleteRow(self):
        """Remove last row"""
        for w in self.controls[-1]:
            self.grid.removeWidget(w)
            w.deleteLater()
        self.controls.pop(-1)

        # disable first subtraction button
        if len(self.controls) == 1:
            self.controls[0][2].setEnabled(False)

    def addPressed(self, row):
        """User adds a new row."""
        val = list(self.setting.val)
        val.insert(row+1, '')
        self.sigSettingChanged.emit(self, self.setting, tuple(val))

    def subPressed(self, row):
        """User deletes a row."""
        val = list(self.setting.val)
        val.pop(row)
        self.sigSettingChanged.emit(self, self.setting, tuple(val))

    @qt.pyqtSlot()
    def onModified(self):
        """Called when the setting is changed remotely,
        or when control is opened"""

        s = self.setting

        if self.last == s.val:
            return
        self.last = s.val

        # update number of rows
        while len(self.setting.val) > len(self.controls):
            self.makeRow()
        while len(self.setting.val) < len(self.controls):
            self.deleteRow()

        # update values
        self.updateControls()

    def makeControl(self, row):
        """Override this to make an editing widget."""
        return None

    def updateControls(self):
        """Override this to update values in controls."""
        pass

    def readControl(self, cntrl):
        """Read value from control."""
        return None

    def dataChanged(self, row):
        """Update row of setitng with new data"""
        val = list(self.setting.val)
        val[row] = self.readControl( self.controls[row][0] )
        self.sigSettingChanged.emit(self, self.setting, tuple(val))

class Datasets(MultiSettingWidget):
    """A control for editing a list of datasets."""

    def __init__(self, setting, doc, dimensions, datatype, *args):
        """Contruct set of comboboxes"""

        MultiSettingWidget.__init__(self, setting, doc, *args)
        self.dimensions = dimensions
        self.datatype = datatype

        self.lastdatasets = []
        # force updating to initialise
        self.onModified()

    def makeControl(self, row):
        """Make QComboBox edit widget."""
        combo = qt.QComboBox()
        combo.setEditable(True)
        combo.lineEdit().editingFinished.connect(lambda: self.dataChanged(row))
        # if a different item is selected
        combo.textActivated[str].connect(lambda x: self.dataChanged(row))
        utils.populateCombo(combo, self.getDatasets())
        return combo

    def readControl(self, control):
        """Get text for control."""
        return control.lineEdit().text()

    def getDatasets(self):
        """Get applicable datasets (sorted)."""
        datasets = []
        for name, ds in self.document.data.items():
            if (ds.dimensions == self.dimensions and
                ds.datatype == self.datatype):
                datasets.append(name)
        datasets.sort()
        return datasets

    def updateControls(self):
        """Set values of controls."""
        for cntrls, val in zip(self.controls, self.setting.val):
            cntrls[0].lineEdit().setText(val)

    @qt.pyqtSlot()
    def onModified(self):
        """Called when the setting is changed remotely,
        or when control is opened"""

        MultiSettingWidget.onModified(self)

        datasets = self.getDatasets()

        if self.lastdatasets == datasets:
            return
        self.lastdatasets = datasets

        # update list of datasets
        for cntrls in self.controls:
            utils.populateCombo(cntrls[0], datasets)

class Strings(MultiSettingWidget):
    """A list of strings."""

    def __init__(self, setting, doc, *args):
        """Construct widget as combination of LineEdit and PushButton
        for browsing."""

        MultiSettingWidget.__init__(self, setting, doc, *args)
        self.onModified()

    def makeControl(self, row):
        """Make edit widget."""
        lineedit = qt.QLineEdit()
        lineedit.editingFinished.connect(lambda: self.dataChanged(row))
        return lineedit

    def readControl(self, control):
        """Get text for control."""
        return control.text()

    def updateControls(self):
        """Set values of controls."""
        for cntrls, val in zip(self.controls, self.setting.val):
            cntrls[0].setText(val)

class Filename(qt.QWidget):
    """A widget for selecting a filename with a browse button."""

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)

    def __init__(self, setting, mode, parent):
        """Construct widget as combination of LineEdit and PushButton
        for browsing.

        mode is 'image' or 'file'
        """

        qt.QWidget.__init__(self, parent)
        self.mode = mode
        self.setting = setting

        layout = qt.QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)

        # the actual edit control
        self.edit = qt.QLineEdit()
        self.edit.setText( setting.toUIText() )
        layout.addWidget(self.edit)

        b = self.button = DotDotButton(
            checkable=False, tooltip=_("Browse for file"))
        layout.addWidget(b)

        # connect up signals
        self.edit.editingFinished.connect(self.validateAndSet)
        b.clicked.connect(self.buttonClicked)

        # completion support
        c = self.filenamecompleter = qt.QCompleter(self)
        model = qt.QFileSystemModel()
        c.setModel(model)
        self.edit.setCompleter(c)

        # for read only filenames
        if setting.readonly:
            self.edit.setReadOnly(True)

        self.setting.setOnModified(self.onModified)

    def buttonClicked(self):
        """Button clicked - show file open dialog."""

        title = _('Choose file')
        filefilter = _("All files (*)")
        if self.mode == 'image':
            title = _('Choose image')
            filefilter = (
                "Images (*.png *.jpg *.jpeg *.bmp *.svg *.tiff *.tif "
                "*.gif *.xbm *.xpm);;" + filefilter)
        
        elif self.mode == 'svg':
            title = _('Choose SVG file')
            filefilter = ("Images (*.svg);;" + filefilter)

        retn = qt.QFileDialog.getOpenFileName(
            self, title, self.edit.text(), filefilter)

        if retn:
            filename = retn[0]
            self.sigSettingChanged.emit(self, self.setting, filename)

    def validateAndSet(self):
        """Check the text is a valid setting and update it."""

        text = self.edit.text()
        try:
            val = self.setting.fromUIText(text)
            styleClear(self.edit)
            self.sigSettingChanged.emit(self, self.setting, val)

        except utils.InvalidType:
            styleError(self.edit)

    @qt.pyqtSlot()
    def onModified(self):
        """called when the setting is changed remotely"""
        self.edit.setText( self.setting.toUIText() )

class FontFamily(qt.QFontComboBox):
    """List the font families, showing each font."""

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)

    def __init__(self, setting, parent):
        """Create the combobox."""

        qt.QFontComboBox.__init__(self, parent)
        self.setting = setting
        self.setFontFilters( qt.QFontComboBox.FontFilter.ScalableFonts )

        # set initial value
        self.onModified()

        # stops combobox readjusting in size to fit contents
        self.setSizeAdjustPolicy(
            qt.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)

        self.setting.setOnModified(self.onModified)

        # if a different item is selected
        self.textActivated[str].connect(self.slotActivated)

    def focusOutEvent(self, *args):
        """Allows us to check the contents of the widget."""
        qt.QFontComboBox.focusOutEvent(self, *args)
        self.slotActivated('')

    def slotActivated(self, val):
        """Update setting if a different item is chosen."""
        newval = self.currentText()
        self.sigSettingChanged.emit(self, self.setting, newval)

    @qt.pyqtSlot()
    def onModified(self):
        """Make control reflect chosen setting."""
        self.setCurrentFont( qt.QFont(self.setting.toUIText()) )

class FontStyle(qt.QComboBox):
    """Font style associated with font family."""

    sigSettingChanged = qt.pyqtSignal(qt.QObject, object, object)
    deftext = _('default')

    def __init__(self, setting, familysetting, parent):
        """Create the combobox."""

        qt.QComboBox.__init__(self, parent)
        self.setEditable(True)
        self.setting = setting
        self.familysetting = familysetting

        self.onModified()

        self.setting.setOnModified(self.onModified)
        self.familysetting.setOnModified(self.onModified)

        # if a different item is selected
        self.textActivated[str].connect(self.slotActivated)

    def slotActivated(self, val):
        """Update setting if a different item is chosen."""
        newval = self.currentText().strip()
        if newval == self.deftext:
            newval = ''
        self.sigSettingChanged.emit(self, self.setting, newval)

    @qt.pyqtSlot()
    def onModified(self):
        """Make control reflect chosen setting."""

        font_family = self.familysetting.get()
        styles = [self.deftext] + sorted(
            qt.QFontDatabase.styles(font_family))

        val = self.setting.get().strip()
        if not val:
            val = 'default'
        elif val not in styles:
            styles.append(val)

        utils.populateCombo(self, styles)
        idx = self.findText(val)
        self.setCurrentIndex(idx)

class ErrorStyle(Choice):
    """Choose different error bar styles."""

    _icons = None         # generated icons
    _errorstyles = None   # copied in by setting.py

    def __init__(self, setting, parent):
        if self._icons is None:
            self._generateIcons()

        Choice.__init__(
            self, setting, False,
            self._errorstyles, parent,
            icons=self._icons
        )

    @classmethod
    def _generateIcons(cls):
        """Generate a list of pixmaps for drop down menu."""
        cls._icons = []
        for errstyle in cls._errorstyles:
            cls._icons.append( utils.getIcon('error_%s' % errstyle) )

class Colormap(Choice):
    """Give the user a preview of colormaps.

    Based on Choice to make life easier
    """

    _icons = {}

    size = (32, 12)

    def __init__(self, setn, document, parent):
        names = sorted(document.evaluate.colormaps)

        icons = Colormap._generateIcons(document, names)
        Choice.__init__(
            self, setn, True,
            names, parent,
            icons=icons
        )
        self.setIconSize( qt.QSize(*self.size) )

    @classmethod
    def _generateIcons(kls, document, names):
        """Generate a list of icons for drop down menu."""

        # create a fake dataset smoothly varying from 0 to size[0]-1
        size = kls.size
        fakedataset = N.fromfunction(lambda x, y: y, (size[1], size[0]))

        # keep track of icons to return
        retn = []

        # iterate over colour maps
        for name in names:
            val = document.evaluate.colormaps.get(name, None)
            if name in kls._icons:
                icon = kls._icons[name]
            else:
                if val is None:
                    # empty icon
                    pixmap = qt.QPixmap(*size)
                    pixmap.fill(qt.Qt.GlobalColor.transparent)
                else:
                    # generate icon
                    image = utils.applyColorMap(
                        val, 'linear', fakedataset, 0., size[0]-1., 0)
                    pixmap = qt.QPixmap.fromImage(image)
                icon = qt.QIcon(pixmap)
                kls._icons[name] = icon
            retn.append(icon)
        return retn

class AxisBound(Choice):
    """Control for setting bounds of axis.

    This is to allow dates etc
    """

    def __init__(self, setting, *args):
        Choice.__init__(self, setting, True, ['Auto'], *args)

        modesetn = setting.parent.get('mode')
        modesetn.setOnModified(self.modeChange)

    @qt.pyqtSlot()
    def modeChange(self):
        """Called if the mode of the axis changes.
        Re-set text as float or date."""

        if self.currentText().lower() != 'auto':
            self.setEditText( self.setting.toUIText() )

