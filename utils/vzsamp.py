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

from __future__ import division
import atexit

from veusz.windows.mainwindow import MainWindow
from veusz.document.commandinterpreter import CommandInterpreter
from veusz.utils import veuszDirectory

samp = None

try:
    from sampy import SAMPIntegratedClient, SAMPHubError, \
                      SAMP_STATUS_OK, SAMP_STATUS_ERROR

except ImportError:
    def setup():
        print('SAMP: sampy module not available')

else:
    def load_votable(private_key, sender_id, msg_id, mtype, params, extra):
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

            samp.ereply(msg_id, SAMP_STATUS_OK, result={})

        except KeyError:
            print('SAMP: parameter missing from table.load.votable call')
            samp.ereply(msg_id, SAMP_STATUS_ERROR, result={},
                        error={'samp.errortxt': 'Missing parameter'})

    def close():
        global samp

        if samp is not None:
            samp.disconnect()
            samp = None

    def setup():
        global samp

        try:
            icon = 'file:///' + '/'.join([veuszDirectory,
                                          'windows', 'icons', 'veusz.png'])

            samp = SAMPIntegratedClient(metadata={'samp.name': 'Veusz',
                                                  'samp.icon.url': icon})
            samp.connect()

            atexit.register(close)

            samp.bindReceiveCall('table.load.votable', load_votable)

        except SAMPHubError:
            print('SAMP: could not connect to hub')
