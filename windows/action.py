
import qt
import os.path

icondir = os.path.dirname(__file__) + '/icons'

class Action(qt.QObject):
    """A QAction-like object for associating actions with buttons,
    menu items, and so on."""

    def __init__(self, parent, onaction, iconfilename = None, menutext = None,
                 tooltiptext = None, statusbartext = None):
        qt.QObject.__init__(self)

        self.parent = parent
        self.onaction = onaction
        self.menutext = menutext
        self.tooltiptext = tooltiptext
        self.statusbartext = statusbartext
        self.items = []

        if self.statusbartext == None:
            if self.menutext != None:
                # if there is no text, use the status bar text (removing ...)
                self.statusbartext = self.menutext.replace('...', '')
                self.statusbartext = self.statusbartext.replace('&', '')
            elif self.tooltiptext != None:
                # else try the tooltiptext
                self.statusbartext = self.tooltiptext
            else:
                # leave it blank
                self.statusbartext = ''

        # makes life easier late
        if self.tooltiptext == None:
            # use the menu text if there is no tooltip
            if self.menutext != None:
                self.tooltiptext = self.menutext.replace('...', '')
                self.tooltiptext = self.tooltiptext.replace('&', '')
            else:
                self.tooptiptext = ''
        if self.menutext == None:
            self.menutext = ''

        # find status bar
        mainwin = parent
        while not isinstance(mainwin, qt.QMainWindow) and mainwin != None:
            mainwin = mainwin.parentWidget()

        if mainwin == None:
            self.tipgroup = None
            self.statusbar = None
        else:
            self.tipgroup = mainwin.toolTipGroup()
            self.statusbar = mainwin.statusBar()

        # make icon set
        if iconfilename != None:
            filename = "%s/%s" % (icondir, iconfilename)
            self.iconset = qt.QIconSet( qt.QPixmap(qt.QPixmap(filename)) )
        else:
            self.iconset = None

    def addTo(self, widget):
        """Add the action to the given widget.

        widget can be an instance of QToolBar, QPopupMenu, a Q[HV]Box..."""

        if isinstance(widget, qt.QPopupMenu):
            if self.iconset != None:
                num = widget.insertItem(self.iconset, self.menutext)
            else:
                num = widget.insertItem(self.menutext)
            widget.connectItem(num, self.slotAction)
            self.items.append( (widget, num) )

            self.connect(widget, qt.SIGNAL('highlighted(int)'),
                         self.slotMenuHighlighted)
            self.connect(widget, qt.SIGNAL('aboutToHide()'),
                         self.slotMenuAboutToHide)

            return num

        else:
            b = qt.QToolButton(widget)
            if self.iconset != None:
                b.setIconSet(self.iconset)
            if self.tipgroup == None:
                qt.QToolTip.add(b, self.tooptiptext)
            else:
                qt.QToolTip.add(b, self.tooltiptext,
                                self.tipgroup, self.statusbartext)
            self.connect(b, qt.SIGNAL('pressed()'),
                         self.slotAction)

            self.items.append( (b,) )

            return b

    def enable(self, enabled = True):
        """Enabled the item in the connected objects."""

        for i in self.items:
            if isinstance(i[0], qt.QButton):
                if enabled:
                    i[0].show()
                else:
                    i[0].hide()
                #i[0].setEnabled(enabled)
            elif isinstance(i[0], qt.QPopupMenu):
                i[0].setItemVisible(i[1], enabled)
            else:
                assert False

    def slotAction(self):
        """Called when the action is activated."""
        self.onaction( self )

    def slotMenuHighlighted(self, id):
        """Called when the user hovers over a menu item."""

        # popup menu which called us
        widget = self.sender()
        if (widget, id) in self.items and self.statusbar != None:
            self.statusbar.message(self.statusbartext)

    def slotMenuAboutToHide(self):
        """Menu about to go, so we hide the status bar text."""
        self.statusbar.clear()
