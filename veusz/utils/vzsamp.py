#    Copyright (C) 2012 Science and Technology Facilities Council.
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

from __future__ import division, print_function
import atexit

from ..windows.mainwindow import MainWindow
from ..document.commandinterpreter import CommandInterpreter
from ..utils import resourceDirectory

# samp client object
sampcl = None

try:
    try:
        # astropy 2.0+
        from astropy import samp
    except ImportError:
        from astropy.io import samp

except ImportError:
    def setup():
        print('SAMP: sampy module not available')

else:
    def setup():
        _setup()

def _setup():
    global sampcl

    try:
        icon = 'file:///' + '/'.join(
            [resourceDirectory, 'icons', 'veusz_16.png'])

        sampcl = samp.SAMPIntegratedClient(
            metadata={'samp.name': 'Veusz', 'samp.icon.url': icon})
        sampcl.connect()

        atexit.register(close)
        try:
            sampcl.bindReceiveCall('table.load.votable', load_votable)
        except AttributeError:
            sampcl.bind_receive_call('table.load.votable', load_votable)

    except samp.SAMPHubError:
        print('SAMP: could not connect to hub')

def close():
    global sampcl

    if sampcl is not None:
        sampcl.disconnect()
        sampcl = None

def load_votable(private_key, sender_id, msg_id, mtype, params, extra):
    global sampcl

    try:
        url = params['url']
        name = params['name']
        #table_id = params['table-id']

        # For now, load into the first window which is still open.
        ci = None
        for window in MainWindow.windows:
            if window.isVisible():
                ci = CommandInterpreter(window.document).interface
                break

        if ci is not None:
            ci.ImportFilePlugin('VO table import', name, url=url)

        sampcl.ereply(msg_id, samp.SAMP_STATUS_OK, result={})

    except KeyError:
        print('SAMP: parameter missing from table.load.votable call')
        sampcl.ereply(
            msg_id, samp.SAMP_STATUS_ERROR, result={},
            error={'samp.errortxt': 'Missing parameter'})
