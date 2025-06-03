#    Copyright (C) 2004 Jeremy S. Sanders
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

"""Window to edit the document using a tree, widget properties
and formatting properties."""

import os
import tempfile

from .. import qtall as qt

from .. import widgets
from .. import utils
from .. import document
from .. import setting

from .widgettree import WidgetTreeModel, WidgetTreeView

def _(text, disambiguation=None, context='TreeEditWindow'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

class SettingsProxy:
    """Object to handle communication between widget/settings
    or sets of widgets/settings."""

    def childProxyList(self):
        """Return a list settings and setting variables proxified."""

    def settingsProxyList(self):
        """Return list of SettingsProxy objects for sub Settings."""

    def settingList(self):
        """Return list of Setting objects."""

    def actionsList(self):
        """Return list of Action objects."""

    def onSettingChanged(self, control, setting, val):
        """Called when a setting has been modified."""

    def onAction(self, action, console):
        """Called if action pressed. Console window is given."""

    @property
    def name(self):
        """Return name of Settings."""

    def pixmap(self):
        """Return pixmap for Settings."""

    def usertext(self):
        """Return text for user."""

    def setnsmode(self):
        """Return setnsmode of Settings."""

    def multivalued(self, name):
        """Is setting with name multivalued?"""
        return False

    def resetToDefault(self, name):
        """Reset setting to default."""

class SettingsProxySingle(SettingsProxy):
    """A proxy wrapping settings for a single widget."""

    def __init__(self, document, settings, actions=None):
        """Initialise settings proxy.
        settings is the widget settings, actions is its actions."""
        self.document = document
        self.settings = settings
        self.actions = actions

    def childProxyList(self):
        """Return a list settings and setting variables proxified."""
        retn = []
        s = self.settings
        for n in s.getNames():
            o = s.get(n)
            if isinstance(o, setting.Settings):
                retn.append( SettingsProxySingle(self.document, o) )
            else:
                retn.append(o)
        return retn

    def settingsProxyList(self):
        """Return list of SettingsProxy objects."""
        return [
            SettingsProxySingle(self.document, s)
            for s in self.settings.getSettingsList()
        ]

    def settingList(self):
        """Return list of Setting objects."""
        return self.settings.getSettingList()

    def actionsList(self):
        """Return list of actions."""
        return self.actions

    def onSettingChanged(self, control, setting, val):
        """Change setting in document."""
        if setting.val != val:
            self.document.applyOperation(
                document.OperationSettingSet(setting, val))

    def onAction(self, action, console):
        """Run action on console."""
        console.runFunction(action.function)

    @property
    def name(self):
        """Return name."""
        return self.settings.name

    def pixmap(self):
        """Return pixmap."""
        return self.settings.pixmap

    def usertext(self):
        """Return text for user."""
        return self.settings.usertext

    def setnsmode(self):
        """Return setnsmode of Settings."""
        return self.settings.setnsmode

    def resetToDefault(self, name):
        """Reset setting to default."""
        setn = self.settings.get(name)
        self.document.applyOperation(
            document.OperationSettingSet(setn, setn.default))

class SettingsProxyMulti(SettingsProxy):
    """A proxy wrapping settings for multiple widgets."""

    def __init__(self, document, widgets, _root=''):
        """Initialise settings proxy.
        widgets is a list of widgets to proxy for."""
        self.document = document
        self.widgets = widgets
        self._root = _root

        self._settingsatlevel = self._getSettingsAtLevel()
        self._cachesettings = self._cachesetting = self._cachechild = None

    def _getSettingsAtLevel(self):
        """Return settings of widgets at level given."""
        if self._root:
            levels = self._root.split('/')
        else:
            levels = []
        setns = []
        for w in self.widgets:
            s = w.settings
            for lev in levels:
                s = s.get(lev)
            setns.append(s)
        return setns

    def _objList(self, filterclasses):
        """Return a list of objects with the type in filterclasses."""

        setns = self._settingsatlevel

        # get list of names with appropriate class
        names = []
        for n in setns[0].getNames():
            o = setns[0].get(n)
            for c in filterclasses:
                if isinstance(o, c):
                    names.append(n)
                    break

        sset = set(names)
        for s in setns[1:]:
            sset &= set(s.getNames())
        names = [n for n in names if n in sset]

        proxylist = []
        for n in names:
            o = setns[0].get(n)
            if isinstance(o, setting.Settings):
                # construct new proxy settings (adding on name of root)
                newroot = n
                if self._root:
                    newroot = self._root + '/' + newroot
                v = SettingsProxyMulti(
                    self.document, self.widgets, _root=newroot)
            else:
                # use setting from first settings as template
                v = o

            proxylist.append(v)
        return proxylist

    def childProxyList(self):
        """Make a list of proxy settings."""
        if self._cachechild is None:
            self._cachechild = self._objList(
                (setting.Settings, setting.Setting) )
        return self._cachechild

    def settingsProxyList(self):
        """Get list of settings proxy."""
        if self._cachesettings is None:
            self._cachesettings = self._objList( (setting.Settings,) )
        return self._cachesettings

    def settingList(self):
        """Set list of common Setting objects for each widget."""
        if self._cachesetting is None:
            self._cachesetting = self._objList( (setting.Setting,) )
        return self._cachesetting

    def actionsList(self):
        """Get list of common actions."""
        anames = None
        for widget in self.widgets:
            a = set([a.name for a in widget.actions])
            if anames is None:
                anames = a
            else:
                anames &= a
        actions = [a for a in self.widgets[0].actions if a.name in anames]
        return actions

    def onSettingChanged(self, control, setting, val):
        """Change setting in document."""
        # construct list of operations to change each setting
        ops = []
        sname = setting.name
        if self._root:
            sname = self._root + '/' + sname
        for w in self.widgets:
            s = self.document.resolveSettingPath(None, w.path+'/'+sname)
            if s.val != val:
                ops.append(document.OperationSettingSet(s, val))
        # apply all operations
        if ops:
            self.document.applyOperation(
                document.OperationMultiple(ops, descr=_('change settings')))

    def onAction(self, action, console):
        """Run actions with same name."""
        aname = action.name
        for w in self.widgets:
            for a in w.actions:
                if a.name == aname:
                    console.runFunction(a.function)

    @property
    def name(self):
        return self._settingsatlevel[0].name

    def pixmap(self):
        """Return pixmap."""
        return self._settingsatlevel[0].pixmap

    def usertext(self):
        """Return text for user."""
        return self._settingsatlevel[0].usertext

    def setnsmode(self):
        """Return setnsmode."""
        return self._settingsatlevel[0].setnsmode

    def multivalued(self, name):
        """Is setting multivalued?"""
        slist = [s.get(name) for s in self._settingsatlevel]
        first = slist[0].get()
        for s in slist[1:]:
            if s.get() != first:
                return True
        return False

    def resetToDefault(self, name):
        """Reset settings to default."""
        ops = []
        for s in self._settingsatlevel:
            setn = s.get(name)
            ops.append(document.OperationSettingSet(setn, setn.default))
        self.document.applyOperation(
            document.OperationMultiple(ops, descr=_("reset to default")))

class VisibilityButton(qt.QPushButton):
    """Button for toggling 'hide' settings."""

    def __init__(self, setnsproxy, setn, *args, **argsv):
        qt.QPushButton.__init__(self, *args, **argsv)
        self.setContentsMargins(0,0,0,0)
        self.setSizePolicy(
            qt.QSizePolicy.Policy.Minimum, qt.QSizePolicy.Policy.Minimum)
        self.setIconSize(qt.QSize(24,12))
        self.setFlat(True)
        self.setFocusPolicy(qt.Qt.FocusPolicy.NoFocus)

        self.setnsproxy = setnsproxy
        self.setn = setn

        setn.setOnModified(self.updateIcon)

        self.updateIcon()
        self.clicked.connect(self.onClicked)

    def updateIcon(self):
        """Update state given current setting."""
        hidden = self.setn.get()
        if hidden:
            icon = 'veusz-eye-grey'
            tooltip = _('Hidden (click to show; set hide to False)')
        else:
            icon = 'veusz-eye'
            tooltip = _('Visible (click to hide; set hide to True)')

        self.setIcon(utils.getIcon(icon))
        self.setToolTip(tooltip)

    def onClicked(self):
        """Toggle the setting."""
        hidden = self.setn.get()
        self.setnsproxy.onSettingChanged(None, self.setn, not hidden)

class PropertyList(qt.QWidget):
    """Edit the widget properties using a set of controls."""

    def __init__(self, document, showformatsettings=True,
                 *args):
        qt.QWidget.__init__(self, *args)
        self.document = document
        self.showformatsettings = showformatsettings

        self.layout = qt.QGridLayout(self)
        self.layout.setSpacing( self.layout.spacing()//2 )
        self.layout.setContentsMargins(4,4,4,4)

        self.childlist = []
        self.setncntrls = {}     # map setting name to controls

    def getConsole(self):
        """Find console window. This is horrible: HACK."""
        win = self.parent()
        while not hasattr(win, 'console'):
            win = win.parent()
        return win.console

    def _addActions(self, setnsproxy, row):
        """Add a list of actions."""
        for action in setnsproxy.actionsList():
            text = action.name
            if action.usertext:
                text = action.usertext

            lab = qt.QLabel(text)
            self.layout.addWidget(lab, row, 0)
            self.childlist.append(lab)

            button = qt.QPushButton(text)
            button.setToolTip(action.descr)
            button.clicked.connect(
                lambda checked=True, a=action:
                setnsproxy.onAction(a, self.getConsole()))

            self.layout.addWidget(button, row, 1)
            self.childlist.append(button)

            row += 1
        return row

    def _addControl(self, setnsproxy, setn, row):
        """Add a control for a setting."""
        cntrl = setn.makeControl(None)
        if cntrl:
            lab = SettingLabel(self.document, setn, setnsproxy)
            self.layout.addWidget(lab, row, 0)
            self.childlist.append(lab)

            cntrl.sigSettingChanged.connect(setnsproxy.onSettingChanged)
            self.layout.addWidget(cntrl, row, 1)
            self.childlist.append(cntrl)
            self.setncntrls[setn.name] = (lab, cntrl)

            row += 1
        return row

    def _addGroupedSettingsControl(self, grpdsetting, row):
        """Add a control for a set of grouped settings."""

        slist = grpdsetting.settingList()

        # make first widget with expandable button

        # this is a label with a + button by this side
        setnlab = SettingLabel(self.document, slist[0], grpdsetting)
        expandbutton = qt.QPushButton(
            "+", checkable=True, flat=True, maximumWidth=16)

        l = qt.QHBoxLayout(spacing=0)
        l.setContentsMargins(0,0,0,0)
        l.addWidget( expandbutton )
        l.addWidget( setnlab )
        lw = qt.QWidget()
        lw.setLayout(l)
        self.layout.addWidget(lw, row, 0)
        self.childlist.append(lw)

        # make main control
        cntrl = slist[0].makeControl(None)
        cntrl.sigSettingChanged.connect(grpdsetting.onSettingChanged)
        self.layout.addWidget(cntrl, row, 1)
        self.childlist.append(cntrl)

        row += 1

        # set of controls for remaining settings
        l = qt.QGridLayout()
        grp_row = 0
        for setn in slist[1:]:
            cntrl = setn.makeControl(None)
            if cntrl:
                lab = SettingLabel(self.document, setn, grpdsetting)
                l.addWidget(lab, grp_row, 0)
                cntrl.sigSettingChanged.connect(grpdsetting.onSettingChanged)
                l.addWidget(cntrl, grp_row, 1)
                grp_row += 1

        grpwidget = qt.QFrame(
            frameShape = qt.QFrame.Shape.Panel,
            frameShadow = qt.QFrame.Shadow.Raised,
            visible=False )
        grpwidget.setLayout(l)

        def ontoggle(checked):
            """Toggle button text and make grp visible/invisible."""
            expandbutton.setText( ("+","-")[checked] )
            grpwidget.setVisible( checked )

        expandbutton.toggled.connect(ontoggle)

        # add group to standard layout
        self.layout.addWidget(grpwidget, row, 0, 1, -1)
        self.childlist.append(grpwidget)
        row += 1
        return row

    def updateProperties(self, setnsproxy, title=None, showformatting=True,
                         onlyformatting=False):
        """Update the list of controls with new ones for the SettingsProxy."""

        # keep a reference to keep it alive
        self._setnsproxy = setnsproxy

        # delete all child widgets
        self.setUpdatesEnabled(False)

        while len(self.childlist) > 0:
            c = self.childlist.pop()
            self.layout.removeWidget(c)
            c.deleteLater()
            del c

        if setnsproxy is None:
            self.setUpdatesEnabled(True)
            return

        row = 0
        self.setncntrls = {}
        self.layout.setEnabled(False)

        # list of settings to show
        setlist = setnsproxy.childProxyList()
        names = [x.name for x in setlist]

        # we special case hide by always showing it at the top
        hideidx = utils.listIndex(names, 'hide')

        # add a title if requested
        if title is not None:
            lab = qt.QLabel(title[0], toolTip=title[1])
            lab.setSizePolicy(
                qt.QSizePolicy.Policy.Expanding, qt.QSizePolicy.Policy.Minimum)

            titlewidget = qt.QFrame(
                frameShape=qt.QFrame.Shape.Panel, frameShadow=qt.QFrame.Shadow.Sunken)
            titlelayout = qt.QHBoxLayout()
            titlelayout.setSpacing(0)
            if hideidx >= 0:
                button = VisibilityButton(setnsproxy, setlist[hideidx])
                titlelayout.addWidget(button)
            titlelayout.addWidget(lab)
            titlelayout.setContentsMargins(0,0,0,0)
            titlewidget.setLayout(titlelayout)

            self.layout.addWidget(titlewidget, row, 0, 1, -1)
            row += 1

        # add actions if parent is widget
        if setnsproxy.actionsList() and not showformatting:
            row = self._addActions(setnsproxy, row)

        if setnsproxy.settingsProxyList() and self.showformatsettings:
            # if we have subsettings, use tabs
            tabbed = TabbedFormatting(self.document, setnsproxy)
            self.layout.addWidget(tabbed, row, 1, 1, 2)
            row += 1
            self.childlist.append(tabbed)
        else:

            def addrow(setn):
                # only add if formatting setting and formatting allowed
                # and not formatting and not formatting not allowed
                nonlocal row
                if ( isinstance(setn, setting.Setting) and (
                        (setn.formatting and (showformatting or onlyformatting))
                        or (not setn.formatting and not onlyformatting)) and
                     not setn.hidden ):
                    row = self._addControl(setnsproxy, setn, row)
                elif ( isinstance(setn, SettingsProxy) and
                       setn.setnsmode() == 'groupedsetting' and
                       not onlyformatting ):
                    row = self._addGroupedSettingsControl(setn, row)

            # else add settings proper as a list
            for name, setn in zip(names, setlist):
                if name != 'hide':
                    addrow(setn)

            if hideidx >= 0:
                addrow(setlist[hideidx])

        # add empty widget to take rest of space
        w = qt.QWidget( sizePolicy=qt.QSizePolicy(
            qt.QSizePolicy.Policy.Maximum, qt.QSizePolicy.Policy.MinimumExpanding) )
        self.layout.addWidget(w, row, 0)
        self.childlist.append(w)

        self.setUpdatesEnabled(True)
        self.layout.setEnabled(True)

    def showHideSettings(self, setnshow, setnhide):
        """Show or hide controls for settings."""
        for vis, setns in ( (True, setnshow), (False, setnhide) ):
            for setn in setns:
                if setn in self.setncntrls:
                    for cntrl in self.setncntrls[setn]:
                        cntrl.setVisible(vis)

class TabbedFormatting(qt.QTabWidget):
    """Class to have tabbed set of settings."""

    def __init__(self, document, setnsproxy, shownames=False):
        qt.QTabWidget.__init__(self)
        self.setUsesScrollButtons(True)
        self.document = document

        if setnsproxy is None:
            return

        # get list of settings
        self.setnsproxy = setnsproxy
        setnslist = setnsproxy.settingsProxyList()

        # add formatting settings if necessary
        numformat = len(
            [setn for setn in setnsproxy.settingList() if setn.formatting] )
        if numformat > 0:
            # add on a formatting tab
            setnslist.insert(0, setnsproxy)

        self.currentChanged.connect(self.slotCurrentChanged)

        # subsettings for tabs
        self.tabsubsetns = []

        # collected titles and tooltips for tabs
        self.tabtitles = []
        self.tabtooltips = []

        # tabs which have been initialized
        self.tabinit = set()

        # for modifiying tab icons depending on hidden status
        hide_mapper = qt.QSignalMapper(self)
        self.hide_map = {}

        for subset in setnslist:
            if subset.setnsmode() not in ('formatting', 'widgetsettings'):
                continue
            self.tabsubsetns.append(subset)

            hidesetn = None
            for s in subset.settingList():
                if s.name == 'hide':
                    hidesetn = s
                    break

            # details of tab
            if subset is setnsproxy:
                # main tab formatting, so this is special
                pixmap = 'settings_main'
                tabname = title = _('Main')
                tooltip = _('Main formatting')
            else:
                # others
                if hasattr(subset, 'pixmap'):
                    pixmap = subset.pixmap()
                else:
                    pixmap = None
                tabname = subset.name
                tooltip = title = subset.usertext()

            # hide name in tab
            if not shownames:
                tabname = ''

            self.tabtitles.append(title)
            self.tabtooltips.append(tooltip)

            # create tab
            indx = self.addTab(qt.QWidget(), tabname)

            # tab icon updates to visible/invisible depending on
            # whether subsetting is hidden or not
            if hidesetn is not None:
                hidesetn.onmodified.onModified.connect(
                    hide_mapper.map)
                hide_mapper.setMapping(hidesetn.onmodified, indx)

            self.hide_map[indx] = (pixmap, hidesetn)
            self.updateIcon(indx)
            self.setTabToolTip(indx, tooltip)

        hide_mapper.mappedInt[int].connect(self.updateIcon)

    def updateIcon(self, indx):
        """Update icon for tab, depending on status of hide setting."""
        pixmap, hidesetn = self.hide_map[indx]
        icon = utils.getIcon(pixmap)
        if hidesetn is not None and hidesetn.get():
            icon = qt.QIcon(utils.DisabledIconEngine(icon))
        self.setTabIcon(indx, icon)

    def slotCurrentChanged(self, tab):
        """Lazy loading of tab when displayed."""
        if tab in self.tabinit:
            # already initialized
            return
        self.tabinit.add(tab)

        # settings to show
        subsetn = self.tabsubsetns[tab]
        # whether these are the main settings
        mainsettings = subsetn is self.setnsproxy

        # add this property list to the scroll widget for tab
        plist = PropertyList(self.document, showformatsettings=not mainsettings)
        plist.updateProperties(
            subsetn,
            title=(self.tabtitles[tab], self.tabtooltips[tab]),
            onlyformatting=mainsettings)

        # create scrollable area
        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(plist)

        # layout for tab widget
        layout = qt.QVBoxLayout()
        layout.setContentsMargins(2,2,2,2)
        layout.addWidget(scroll)

        # finally use layout containing items for tab
        self.widget(tab).setLayout(layout)

class FormatDock(qt.QDockWidget):
    """A window for formatting the current widget.
    Provides tabbed formatting properties
    """

    def __init__(self, document, treeedit, *args):
        qt.QDockWidget.__init__(self, *args)
        self.setWindowTitle(_("Formatting - Veusz"))
        self.setObjectName("veuszformattingdock")

        self.document = document
        self.tabwidget = None

        self.container = qt.QWidget()
        self.layout = qt.QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.container.setLayout(self.layout)
        self.setWidget(self.container)

        # update our view when the tree edit window selection changes
        treeedit.widgetsSelected.connect(self.selectedWidgets)

    def selectedWidgets(self, widgets, setnsproxy):
        """Created tabbed widgets for formatting for each subsettings."""

        # get current tab (so we can set it afterwards)
        if self.tabwidget:
            tab = self.tabwidget.currentIndex()
        else:
            tab = 0

        # delete old tabwidget
        oldItem = self.layout.takeAt(0)
        if oldItem:
            oldWidget = oldItem.widget()
            if oldWidget:
                oldWidget.setParent(None)
                oldWidget.deleteLater()
                self.tabwidget = None

        self.tabwidget = TabbedFormatting(self.document, setnsproxy)
        self.layout.addWidget(self.tabwidget)

        # wrap tab from zero to max number
        tab = max( min(self.tabwidget.count()-1, tab), 0 )
        self.tabwidget.setCurrentIndex(tab)

class PropertiesDock(qt.QDockWidget):
    """A window for editing properties for widgets."""

    def __init__(self, document, treeedit, *args):
        qt.QDockWidget.__init__(self, *args)
        self.setWindowTitle(_("Properties - Veusz"))
        self.setObjectName("veuszpropertiesdock")

        self.document = document

        # update our view when the tree edit window selection changes
        treeedit.widgetsSelected.connect(self.slotWidgetsSelected)

        # construct scrollable area
        self.scroll = qt.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.setWidget(self.scroll)

        # construct properties list in scrollable area
        self.proplist = PropertyList(document, showformatsettings=False)
        self.scroll.setWidget(self.proplist)

    def slotWidgetsSelected(self, widgets, setnsproxy):
        """Update properties when selected widgets change."""
        self.proplist.updateProperties(setnsproxy, showformatting=False)

class TreeEditDock(qt.QDockWidget):
    """A dock window presenting widgets as a tree."""

    widgetsSelected = qt.pyqtSignal(list, object)
    sigPageChanged = qt.pyqtSignal(int)

    def __init__(self, document, parentwin):
        """Initialise dock given document and parent widget."""
        qt.QDockWidget.__init__(self, parentwin)
        self.parentwin = parentwin
        self.setWindowTitle(_("Editing - Veusz"))
        self.setObjectName("veuszeditingwindow")
        self.selwidgets = []

        self.document = document

        # construct tree
        self.treemodel = WidgetTreeModel(document)
        self.treeview = WidgetTreeView(self.treemodel)
        self.document.sigWiped.connect(self.slotDocumentWiped)

        # receive change in selection
        self.treeview.selectionModel().selectionChanged.connect(
            self.slotTreeItemsSelected)

        # set tree as main widget
        self.setWidget(self.treeview)

        # toolbar to create widgets
        self.addtoolbar = qt.QToolBar(
            _("Insert toolbar - Veusz"), parentwin)
        # note wrong description!: backwards compatibility
        self.addtoolbar.setObjectName("veuszeditingtoolbar")

        # toolbar for editting widgets
        self.edittoolbar = qt.QToolBar(
            _("Edit toolbar - Veusz"), parentwin)
        self.edittoolbar.setObjectName("veuszedittoolbar")

        self._constructToolbarMenu()
        parentwin.addToolBarBreak(qt.Qt.ToolBarArea.TopToolBarArea)
        parentwin.addToolBar(qt.Qt.ToolBarArea.TopToolBarArea, self.addtoolbar)
        parentwin.addToolBar(qt.Qt.ToolBarArea.TopToolBarArea, self.edittoolbar)

        # this sets various things up
        self.selectWidget(document.basewidget)

        # update paste button when clipboard changes
        qt.QApplication.clipboard().dataChanged.connect(
            self.updatePasteButton)
        self.updatePasteButton()

        # make context menu
        self.makeContextMenu()

    def makeContextMenu(self):
        m = self.contextmenu = qt.QMenu(self)

        # selection
        m.addMenu(self.parentwin.menus['edit.select'])
        m.addSeparator()

        # actions on widget(s)
        for act in (
                'edit.cut', 'edit.copy', 'edit.copy_as_image', 'edit.paste',
                'edit.moveup', 'edit.movedown', 'edit.delete',
                'edit.rename'
        ):
            m.addAction(self.vzactions[act])

        m.addSeparator()
        m.addAction(self.vzactions['edit.show'])
        m.addAction(self.vzactions['edit.hide'])

    def slotDocumentWiped(self):
        """If the document is wiped, reselect root widget."""
        self.selectWidget(self.document.basewidget)

    def slotTreeItemsSelected(self, current, previous):
        """New item selected in tree.

        This updates the list of properties
        """

        # get selected widgets
        self.selwidgets = swidget = [
            self.treemodel.getWidget(idx)
            for idx in self.treeview.selectionModel().selectedRows() ]

        if len(swidget) == 0:
            setnsproxy = None
        elif len(swidget) == 1:
            setnsproxy = SettingsProxySingle(
                self.document, swidget[0].settings, actions=swidget[0].actions)
        else:
            setnsproxy = SettingsProxyMulti(self.document, swidget)

        self._enableCorrectButtons()
        self._checkPageChange()

        self.widgetsSelected.emit(swidget, setnsproxy)

    def contextMenuEvent(self, event):
        """Bring up context menu."""
        if self.selwidgets:
            self.contextmenu.exec(self.mapToGlobal(event.pos()))
            event.accept()

    def _checkPageChange(self):
        """Check to see whether page has changed."""

        w = None
        if self.selwidgets:
            w = self.selwidgets[0]
        while w is not None and not isinstance(w, widgets.Page):
            w = w.parent

        if w is not None:
            # have page, so check what number we are in basewidget children
            try:
                i = self.document.basewidget.children.index(w)
                self.sigPageChanged.emit(i)
            except ValueError:
                pass

    def _enableCorrectButtons(self):
        """Make sure the create graph buttons are correctly enabled."""

        selw = None
        if self.selwidgets:
            selw = self.selwidgets[0]

        # has to be visible if is to be enabled (yuck)
        self.vzactions['add.nonorthpoint'].setVisible(True)
        self.vzactions['add.point3d'].setVisible(True)

        # check whether each button can have this widget
        # (or a parent) as parent
        for wc, action in self.addslots.items():
            w = selw
            while w is not None and not wc.willAllowParent(w):
                w = w.parent

            self.vzactions['add.%s' % wc.typename].setEnabled(w is not None)

        self.vzactions['add.axismenu'].setEnabled(
            self.vzactions['add.axis'].isEnabled())

        # exclusive widgets
        nonorth = self.vzactions['add.nonorthpoint'].isEnabled()
        in3dgraph = self.vzactions['add.point3d'].isEnabled()
        self.vzactions['add.nonorthpoint'].setVisible(nonorth)
        self.vzactions['add.point3d'].setVisible(in3dgraph)
        self.vzactions['add.xy'].setVisible(not nonorth and not in3dgraph)
        self.vzactions['add.nonorthfunc'].setVisible(nonorth)
        self.vzactions['add.function'].setVisible(not nonorth and not in3dgraph)
        self.vzactions['add.function3d'].setVisible(in3dgraph)
        self.vzactions['add.axismenu'].setVisible(not in3dgraph)
        self.vzactions['add.axis3d'].setVisible(in3dgraph)
        self.vzactions['add.image'].setVisible(not in3dgraph)
        self.vzactions['add.surface3d'].setVisible(in3dgraph)
        self.vzactions['add.contour'].setVisible(not in3dgraph)
        self.vzactions['add.volume3d'].setVisible(in3dgraph)

        # certain actions shouldn't work on root
        isnotroot = not any([isinstance(w, widgets.Root)
                             for w in self.selwidgets])

        for act in ('edit.cut', 'edit.copy', 'edit.copy_as_image', 'edit.delete',
                    'edit.moveup', 'edit.movedown', 'edit.rename'):
            self.vzactions[act].setEnabled(isnotroot)

        # show or hide widgets
        hidden = [
            w.settings.hide
            for w in self.selwidgets if 'hide' in w.settings
        ]
        self.vzactions['edit.show'].setEnabled(any(hidden))
        self.vzactions['edit.hide'].setEnabled(
            len(hidden)>0 and not all(hidden))

        self.updatePasteButton()

    def _constructToolbarMenu(self):
        """Add items to edit/add graph toolbar and menu."""

        def slotklass(klass):
            return lambda: self.slotMakeWidgetButton(klass)

        iconsize = setting.settingdb['toolbar_size']
        self.addtoolbar.setIconSize( qt.QSize(iconsize, iconsize) )
        self.edittoolbar.setIconSize( qt.QSize(iconsize, iconsize) )

        self.addslots = {}
        self.vzactions = actions = self.parentwin.vzactions
        for widgettype in (
                'page', 'grid', 'graph', 'axis',
                'axis-broken', 'axis-function',
                'xy', 'bar', 'histo', 'fit', 'function', 'boxplot',
                'image', 'contour', 'vectorfield',
                'key', 'label', 'colorbar',
                'rect', 'ellipse', 'imagefile', 'svgfile',
                'line', 'polygon', 'polar', 'ternary',
                'nonorthpoint', 'nonorthfunc',
                'covariance',
                'scene3d',
                'graph3d', 'function3d', 'point3d', 'axis3d',
                'surface3d', 'volume3d'
        ):

            wc = document.thefactory.getWidgetClass(widgettype)
            slot = slotklass(wc)
            self.addslots[wc] = slot

            actionname = 'add.' + widgettype
            actions[actionname] = utils.makeAction(
                self,
                wc.description, _('Add %s') % widgettype,
                slot,
                icon='button_%s' % widgettype)

        a = utils.makeAction
        actions.update({
            'edit.cut': a(
                self, _('Cut the selected widget'), _('Cu&t'),
                self.slotWidgetCut,
                icon='veusz-edit-cut', key='Ctrl+X'),
            'edit.copy': a(
                self, _('Copy the selected widget'), _('&Copy'),
                self.slotWidgetCopy,
                icon='kde-edit-copy', key='Ctrl+C'),
            'edit.copy_as_image': a(
                self, _('Copy the current page as svg image'), _('&Copy as Image'),
                self.slotWidgetCopyAsImage,
                icon='kde-edit-copy', key='Ctrl+Alt+C'),
            'edit.paste': a(
                self, _('Paste widget from the clipboard'), _('&Paste'),
                self.slotWidgetPaste,
                icon='kde-edit-paste', key='Ctrl+V'),
            'edit.moveup': a(
                self, _('Move the selected widget up'), _('Move &up'),
                lambda: self.slotWidgetMove(-1),
                icon='kde-go-up', key='Ctrl+Shift+PgUp'),
            'edit.movedown': a(
                self, _('Move the selected widget down'), _('Move d&own'),
                lambda: self.slotWidgetMove(1),
                icon='kde-go-down', key='Ctrl+Shift+PgDown'),
            'edit.delete': a(
                self, _('Remove the selected widget'), _('&Delete'),
                self.slotWidgetDelete,
                icon='kde-edit-delete'),
            'edit.rename': a(
                self, _('Renames the selected widget'), _('&Rename'),
                self.slotWidgetRename,
                icon='kde-edit-rename'),
            'edit.show': a(
                self, _('Show selected widget(s)'), _('Show'),
                self.slotWidgetShow,
                key='Ctrl+]'),
            'edit.hide': a(
                self, _('Hide selected widget(s)'), _('Hide'),
                self.slotWidgetHide,
                key='Ctrl+['),

            'add.shapemenu': a(
                self, _('Add a shape to the plot'), _('Shape'),
                self.slotShowShapeMenu,
                icon='veusz-shape-menu'),

            'add.axismenu': a(
                self, _('Add an axis to the plot'), _('Axis'),
                None,
                icon='button_axis'),
        })

        # list of widget-generating actions for menu and toolbar
        widgetactions = (
            'add.page',
            'add.grid',
            'add.graph',
            'add.axismenu',
            'add.axis3d',
            'add.xy',
            'add.nonorthpoint',
            'add.point3d',
            'add.bar',
            'add.histo',
            'add.fit',
            'add.function',
            'add.nonorthfunc',
            'add.function3d',
            'add.boxplot',
            'add.image',
            'add.surface3d',
            'add.contour',
            'add.volume3d',
            'add.vectorfield',
            'add.key',
            'add.label',
            'add.colorbar',
            'add.polar',
            'add.ternary',
            'add.scene3d',
            'add.graph3d',
            'add.covariance',
            'add.shapemenu',
        )

        # separate menus for adding shapes and axis types
        shapemenu = qt.QMenu(self)
        shapemenu.addActions( [actions[act] for act in (
            'add.rect',
            'add.ellipse',
            'add.line',
            'add.imagefile',
            'add.svgfile',
            'add.polygon',
        )])
        actions['add.shapemenu'].setMenu(shapemenu)

        axismenu = qt.QMenu(self)
        axismenu.addActions( [actions[act] for act in (
            'add.axis',
            'add.axis-broken',
            'add.axis-function',
        )])
        actions['add.axismenu'].setMenu(axismenu)
        actions['add.axismenu'].triggered.connect(actions['add.axis'].trigger)

        menuitems = (
            ('insert', '', widgetactions),
            ('edit', '', (
                'edit.cut',
                'edit.copy',
                'edit.copy_as_image',
                'edit.paste',
                'edit.delete',
                'edit.rename',
                'edit.moveup',
                'edit.movedown',
                'edit.show',
                'edit.hide',
            )),
        )
        utils.constructMenus(
            self.parentwin.menuBar(),
            self.parentwin.menus,
            menuitems,
            actions
        )

        # add actions to toolbar to create widgets
        utils.addToolbarActions(self.addtoolbar, actions, widgetactions)

        # add action to toolbar for editing
        utils.addToolbarActions(
            self.edittoolbar,  actions,
            (
                'edit.cut', 'edit.copy', 'edit.paste',
                'edit.moveup', 'edit.movedown',
                'edit.delete', 'edit.rename',
            )
        )

        self.parentwin.menus['edit.select'].aboutToShow.connect(
            self.updateSelectMenu)

    def slotMakeWidgetButton(self, wc):
        """User clicks button to make widget."""
        self.makeWidget(wc.typename)

    def slotShowShapeMenu(self):
        a = self.vzactions['add.shapemenu']
        a.menu().popup( qt.QCursor.pos() )

    def makeWidget(self, widgettype, autoadd=True, name=None):
        """Called when an add widget button is clicked.
        widgettype is the type of widget
        autoadd specifies whether to add default children
        if name is set this name is used if possible (ie no other children have it)
        """

        # if no widget selected, bomb out
        if not self.selwidgets:
            return
        parent = document.getSuitableParent(widgettype, self.selwidgets[0])

        assert parent is not None

        if name in parent.childnames:
            name = None

        # make the new widget and update the document
        w = self.document.applyOperation(
            document.OperationWidgetAdd(
                parent, widgettype, autoadd=autoadd, name=name) )

        # select the widget
        self.selectWidget(w)

        # bump the feedback statistics
        utils.feedback.widgetcts[widgettype] += 1

    def slotWidgetCut(self):
        """Cut the selected widget"""
        self.slotWidgetCopy()
        self.slotWidgetDelete()

    def slotWidgetCopy(self):
        """Copy selected widget to the clipboard."""

        if self.selwidgets:
            mimedata = document.generateWidgetsMime(self.selwidgets)
            clipboard = qt.QApplication.clipboard()
            clipboard.setMimeData(mimedata)

    def slotWidgetCopyAsImage(self):
        """Copy current page to the clipboard as image files."""
        # general image extentions and their mime types exportable from document
        exts = {'svg': ['image/svg+xml'],
                'png': ['image/png', 'PNG'],
                'bmp': ['image/x-bmp'],
                }

        def checkDone(export, exts):
            """Check whether exporting pages as images has finished."""
            if not export.haveDone():
                return
            try:
                export.finish()
                clipboard = qt.QApplication.clipboard()
                data = qt.QMimeData()

                # set images to QMimeData
                for ext, mimes in exts.items():
                    try:
                        fname = os.path.join(tmpdir.name, 'tmp.{}'.format(ext))
                        with open(fname, 'rb') as fo:
                            bs = fo.read()
                        for mime in mimes:
                            data.setData(mime, bs)
                        os.remove(fname)
                    except:
                        pass

                # set QMimeData to clipboard
                clipboard.setMimeData(data)
                tmpdir.cleanup()

            except:
                qt.QMessageBox.critical(self, _("Error - Veusz"), _("Error while exporting images"))
            self.checktimer.stop()

        # read settings from settingdb
        setdb = setting.settingdb
        export = document.AsyncExport(
            self.document,
            bitmapdpi=setdb['export_DPI'],
            pdfdpi=setdb['export_DPI_PDF'],
            antialias=setdb['export_antialias'],
            color=setdb['export_color'],
            quality=setdb['export_quality'],
            backcolor=setdb['export_background'],
            svgtextastext=setdb['export_SVG_text_as_text'],
            svgdpi=setdb['export_DPI_SVG'],
        )

        tmpdir = tempfile.TemporaryDirectory()
        pages = [self.parentwin.plot.getPageNumber()]

        # export images to temporary files
        for ext in exts:
            fname = os.path.join(tmpdir.name, 'tmp.{}'.format(ext))
            export.add(fname, pages)

        # copy images to clipboard after finishing export
        self.checktimer = qt.QTimer(self)
        self.checktimer.timeout.connect(lambda: checkDone(export, exts))
        self.checktimer.start(20)

    def updatePasteButton(self):
        """Is the data on the clipboard a valid paste at the currently
        selected widget? If so, enable paste button"""

        data = document.getClipboardWidgetMime()
        if len(self.selwidgets) == 0:
            show = False
        else:
            show = document.isWidgetMimePastable(self.selwidgets[0], data)
        self.vzactions['edit.paste'].setEnabled(show)

    def doInitialWidgetSelect(self):
        """Select a sensible initial widget."""
        w = self.document.basewidget
        for i in range(2):
            try:
                c = w.children[0]
            except IndexError:
                break
            if c:
                w = c
        self.selectWidget(w)

    def slotWidgetPaste(self):
        """Paste something from the clipboard"""

        data = document.getClipboardWidgetMime()
        if data:
            op = document.OperationWidgetPaste(self.selwidgets[0], data)
            widgets = self.document.applyOperation(op)
            if widgets:
                self.selectWidget(widgets[0])

    def slotWidgetDelete(self):
        """Delete the widget selected."""

        widgets = self.selwidgets
        # if no item selected, leave
        if not widgets:
            return

        # get list of widgets in order
        widgetlist = []
        self.document.basewidget.buildFlatWidgetList(widgetlist)

        # find indices of widgets to be deleted - find one to select after
        indexes = [widgetlist.index(w) for w in widgets]
        if -1 in indexes:
            raise RuntimeError("Invalid widget in list of selected widgets")
        minindex = min(indexes)

        # delete selected widget
        self.document.applyOperation(
            document.OperationWidgetsDelete(widgets))

        # rebuild list
        widgetlist = []
        self.document.basewidget.buildFlatWidgetList(widgetlist)

        # find next to select
        if minindex < len(widgetlist):
            nextwidget = widgetlist[minindex]
        else:
            nextwidget = widgetlist[-1]

        # select the next widget (we have to select root first!)
        self.selectWidget(self.document.basewidget)
        self.selectWidget(nextwidget)

    def slotWidgetRename(self):
        """Allows the user to rename the selected widget."""

        selected = self.treeview.selectedIndexes()
        if len(selected) != 0:
            self.treeview.edit(selected[0])

    def selectWidget(self, widget, mode='new'):
        """Select the associated listviewitem for the widget w in the
        listview.

        mode:
         'new': new selection
         'add': add to selection
         'toggle': toggle selection
        """

        index = self.treemodel.getWidgetIndex(widget)

        if index is not None:
            self.treeview.scrollTo(index)

            flags = qt.QItemSelectionModel.SelectionFlag.Rows | {
                'new':  (
                    qt.QItemSelectionModel.SelectionFlag.ClearAndSelect |
                    qt.QItemSelectionModel.SelectionFlag.Current),
                'add': qt.QItemSelectionModel.SelectionFlag.Select,
                'toggle': qt.QItemSelectionModel.SelectionFlag.Toggle,
            }[mode]

            self.treeview.selectionModel().select(index, flags)

    def slotWidgetMove(self, direction):
        """Move the selected widget up/down in the hierarchy.

        a is the action (unused)
        direction is -1 for 'up' and +1 for 'down'
        """

        if not self.selwidgets:
            return
        # widget to move
        w = self.selwidgets[0]

        # actually move the widget
        self.document.applyOperation(
            document.OperationWidgetMoveUpDown(w, direction) )

        # re-highlight moved widget
        self.selectWidget(w)

    def slotWidgetHide(self):
        """Hide selected widgets."""
        ops = [
            document.OperationSettingSet(w.settings.get('hide'), True)
            for w in self.selwidgets
            if 'hide' in w.settings
        ]
        self.document.applyOperation(
            document.OperationMultiple(ops, descr=_('hide widget(s)')))
        # update state of action after hiding
        self._enableCorrectButtons()

    def slotWidgetShow(self):
        """Hide selected widgets."""
        ops = [
            document.OperationSettingSet(w.settings.get('hide'), False)
            for w in self.selwidgets
            if 'hide' in w.settings
        ]
        self.document.applyOperation(
            document.OperationMultiple(ops, descr=_('show widget(s)')))
        # update state of action after hiding
        self._enableCorrectButtons()

    def checkWidgetSelected(self):
        """Check widget is selected."""
        if len(self.treeview.selectionModel().selectedRows()) == 0:
            self.selectWidget(self.document.basewidget)

    def _selectWidgetsTypeAndOrName(self, wtype, wname, root=None):
        """Select widgets with type or name given.
        Give None if you don't care for either."""
        def selectwidget(path, w):
            """Select widget if of type or name given."""
            if ( (wtype is None or w.typename == wtype) and
                 (wname is None or w.name == wname) ):
                idx = self.treemodel.getWidgetIndex(w)
                self.treeview.selectionModel().select(
                    idx,
                    qt.QItemSelectionModel.SelectionFlag.Select |
                    qt.QItemSelectionModel.SelectionFlag.Rows)

        self.document.walkNodes(selectwidget, nodetypes=('widget',), root=root)

    def _selectWidgetSiblings(self, w, wtype):
        """Select siblings of widget given with type."""

        if w.parent is None:
            return

        for c in w.parent.children:
            if c is not w and c.typename == wtype:
                idx = self.treemodel.getWidgetIndex(c)
                self.treeview.selectionModel().select(
                    idx,
                    qt.QItemSelectionModel.SelectionFlag.Select |
                    qt.QItemSelectionModel.SelectionFlag.Rows)

    def updateSelectMenu(self):
        """Update edit.select menu."""
        menu = self.parentwin.menus['edit.select']
        menu.clear()

        if len(self.selwidgets) == 0:
            return

        widget = self.selwidgets[0]
        wtype = widget.typename
        name = widget.name

        # get page widget for selecting on page
        page = widget
        while page is not None and page.typename != 'page':
            page = page.parent

        menu.addAction(
            _("All '%s' widgets") % wtype,
            lambda: self._selectWidgetsTypeAndOrName(wtype, None))
        menu.addAction(
            _("Siblings of '%s' with type '%s'") % (name, wtype),
            lambda: self._selectWidgetSiblings(widget, wtype))
        menu.addAction(
            _("All '%s' widgets called '%s'") % (wtype, name),
            lambda: self._selectWidgetsTypeAndOrName(wtype, name))
        menu.addAction(
            _("All widgets called '%s'") % name,
            lambda: self._selectWidgetsTypeAndOrName(None, name))
        if page and page is not widget:
            menu.addAction(
                _("All widgets called '%s' on page '%s'") % (name, page.name),
                lambda: self._selectWidgetsTypeAndOrName(
                    None, name, root=page))

class SettingLabel(qt.QWidget):
    """A label to describe a setting.

    This widget shows the name, a tooltip description, and gives
    access to the context menu
    """

    # this is emitted when widget is clicked
    signalClicked = qt.pyqtSignal(qt.QPoint)

    def __init__(self, document, setting, setnsproxy):
        """Initialise button, passing document, setting, and parent widget."""

        qt.QWidget.__init__(self)
        self.setFocusPolicy(qt.Qt.FocusPolicy.StrongFocus)

        self.document = document
        document.signalModified.connect(self.slotDocModified)

        self.setting = setting
        self.setnsproxy = setnsproxy

        self.layout = qt.QHBoxLayout(self)
        self.layout.setContentsMargins(2,2,2,2)

        if setting.usertext:
            text = setting.usertext
        else:
            text = setting.name
        self.labelicon = qt.QLabel(text)
        self.layout.addWidget(self.labelicon)

        self.iconlabel = qt.QLabel()
        self.layout.addWidget(self.iconlabel)

        self.signalClicked.connect(self.settingMenu)

        self.infocus = False
        self.inmouse = False
        self.inmenu = False

        # initialise settings
        self.slotDocModified(True)

    def mouseReleaseEvent(self, event):
        """Emit signalClicked(pos) on mouse release."""
        self.signalClicked.emit( self.mapToGlobal(event.pos()) )
        return qt.QWidget.mouseReleaseEvent(self, event)

    def keyReleaseEvent(self, event):
        """Emit signalClicked(pos) on key release."""
        if event.key() == qt.Qt.Key.Key_Space:
            self.signalClicked.emit(
                self.mapToGlobal(self.iconlabel.pos()) )
            event.accept()
        else:
            return qt.QWidget.keyReleaseEvent(self, event)

    # Mark as a qt slot. This fixes a bug where you get C/C++ object
    # deleted messages when the document emits signalModified but this
    # widget has been deleted. This can be reproduced by dragging a
    # widget between two windows, then undoing.
    @qt.pyqtSlot(int)
    def slotDocModified(self, ismodified):
        """If the document has been modified."""

        # update pixmap (e.g. link added/removed)
        self.updateHighlight()

        # update tooltip
        tooltip = self.setting.descr
        if self.setting.isReference():
            paths = self.setting.getReference().getPaths()
            tooltip += _('\nLinked to: %s') % ', '.join(paths)
        self.setToolTip(tooltip)

        # if not default, make label bold
        f = qt.QFont(self.labelicon.font())
        multivalued = self.setnsproxy.multivalued(self.setting.name)
        f.setBold( (not self.setting.isDefault()) or multivalued )
        f.setItalic( multivalued )
        self.labelicon.setFont(f)

    def updateHighlight(self):
        """Show drop down arrow if item has focus."""
        if self.inmouse or self.infocus or self.inmenu:
            pixmap = 'downarrow.png'
        else:
            if self.setting.isReference() and not self.setting.isDefault():
                pixmap = 'link.png'
            else:
                pixmap = 'downarrow_blank.png'
        self.iconlabel.setPixmap(utils.getPixmap(pixmap))

    def enterEvent(self, event):
        """Focus on mouse enter."""
        self.inmouse = True
        self.updateHighlight()
        return qt.QWidget.enterEvent(self, event)

    def leaveEvent(self, event):
        """Clear focus on mouse leaving."""
        self.inmouse = False
        self.updateHighlight()
        return qt.QWidget.leaveEvent(self, event)

    def focusInEvent(self, event):
        """Focus if widgets gets focus."""
        self.infocus = True
        self.updateHighlight()
        return qt.QWidget.focusInEvent(self, event)

    def focusOutEvent(self, event):
        """Lose focus if widget loses focus."""
        self.infocus = False
        self.updateHighlight()
        return qt.QWidget.focusOutEvent(self, event)

    def addCopyToWidgets(self, menu):
        """Make a menu with list of other widgets in it."""

        def getWidgetsOfType(widget, widgettype, widgets=[]):
            """Recursively build up a list of widgets of the type given."""
            for w in widget.children:
                if w.typename == widgettype:
                    widgets.append(w)
                getWidgetsOfType(w, widgettype, widgets)

        # get list of widget paths to copy setting to
        # this is all widgets of same type
        widgets = []
        setwidget = self.setting.getWidget()
        if setwidget is None:
            return
        getWidgetsOfType(
            self.document.basewidget, setwidget.typename, widgets)
        widgets = [w.path for w in widgets if w != setwidget]
        widgets.sort()

        # chop off widget part of setting path
        # this is so we can add on a different widget path
        # note setpath needs to include Settings part of path too
        setpath = self.setting.path
        wpath = self.setting.getWidget().path
        setpath = setpath[len(wpath):]  # includes /

        def modifyfn(widget):
            def modify():
                """Modify the setting for the widget given."""
                wpath = widget + setpath
                self.document.applyOperation(
                    document.OperationSettingSet(wpath, self.setting.get()))
            return modify

        for widget in widgets:
            action = menu.addAction(widget)
            action.triggered.connect(modifyfn(widget))

    @qt.pyqtSlot(qt.QPoint)
    def settingMenu(self, pos):
        """Pop up menu for each setting."""

        # forces settings to be updated
        self.parentWidget().setFocus()
        # get it back straight away
        self.setFocus()

        # get widget, with its type and name
        widget = self.setting.parent
        while widget is not None and not isinstance(widget, widgets.Widget):
            widget = widget.parent
        if widget is None:
            return
        self._clickwidget = widget

        wtype = widget.typename
        name = widget.name

        popup = qt.QMenu(self)
        popup.addAction(
            _('Reset to default'),
            self.actionResetDefault)

        if self.setting.path[:12] != '/StyleSheet/':
            # settings not relevant for style sheet items

            copyto = popup.addMenu(_('Copy to'))
            copyto.addAction(
                _("all '%s' widgets") % wtype,
                self.actionCopyTypedWidgets)
            copyto.addAction(
                _("'%s' siblings") % wtype,
                self.actionCopyTypedSiblings)
            copyto.addAction(
                _("'%s' widgets called '%s'") % (wtype, name),
                self.actionCopyTypedNamedWidgets)
            copyto.addSeparator()
            self.addCopyToWidgets(copyto)

            popup.addAction(
                _('Use as default style'),
                self.actionSetStyleSheet)

        # special actions for references
        if self.setting.isReference():
            popup.addSeparator()
            popup.addAction(
                _('Unlink setting'),
                self.actionUnlinkSetting)

        self.inmenu = True
        self.updateHighlight()
        popup.exec(pos)
        self.inmenu = False
        self.updateHighlight()

    def actionResetDefault(self):
        """Reset setting to default."""
        self.setnsproxy.resetToDefault(self.setting.name)

    def actionCopyTypedWidgets(self):
        """Copy setting to widgets of same type."""
        self.document.applyOperation(
            document.OperationSettingPropagate(self.setting) )

    def actionCopyTypedSiblings(self):
        """Copy setting to siblings of the same type."""
        self.document.applyOperation(
            document.OperationSettingPropagate(
                self.setting,
                root=self._clickwidget.parent,
                maxlevels=1) )

    def actionCopyTypedNamedWidgets(self):
        """Copy setting to widgets with the same name and type."""
        self.document.applyOperation(
            document.OperationSettingPropagate(
                self.setting,
                widgetname=
                self._clickwidget.name) )

    def actionUnlinkSetting(self):
        """Unlink the setting if it is a reference."""
        self.document.applyOperation(
            document.OperationSettingSet(self.setting, self.setting.get()) )

    def actionSetStyleSheet(self):
        """Use the setting as the default in the stylesheet."""

        # get name of stylesheet setting
        sslink = self.setting.getStylesheetLink()
        # apply operation to change it
        self.document.applyOperation(
            document.OperationMultiple(
                [
                    document.OperationSettingSet(sslink, self.setting.get()),
                    document.OperationSettingSet(
                        self.setting, self.setting.default)
                ],
                descr=_("make default style"))
        )
