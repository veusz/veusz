#    Copyright (C) 2011 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This file is part of Veusz.
#
#    Veusz is free software: you can redistribute it and/or modify it
#    under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 2 of the License, or
#    (at your option) any later version.
#
#    Veusz is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#    General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Veusz. If not, see <https://www.gnu.org/licenses/>.
#
##############################################################################

"""Wrapper for D-Bus module it."""

import sys
import os

# interface name veusz interfaces appear on
sessionbus = busname = None

try:
    import dbus
    from dbus.service import method, Object
    from dbus.mainloop.qt import DBusQtMainLoop

    def setup():
        """Initialise dbus."""

        global sessionbus
        global busname

        try:
            DBusQtMainLoop(set_as_default=True)

            sessionbus = dbus.SessionBus()
            busname = dbus.service.BusName(
                'org.veusz.pid%i' % os.getpid(), sessionbus)

        except dbus.exceptions.DBusException:
            sys.stderr.write('Exception when connecting to DBus')
            sessionbus = None
            busname = None

except ImportError:
    # no DBus, so we try to make the interface do nothing

    def setup():
        pass

    def method(**argsv):
        def donothing(m):
            return m
        return donothing

    class Object:
        def __init__(self, *args):
            pass
