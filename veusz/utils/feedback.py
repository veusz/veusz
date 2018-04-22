#    Copyright (C) 2018 Jeremy S. Sanders
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

from __future__ import division, absolute_import, print_function
from collections import defaultdict
import datetime
import sys
import atexit
import platform

import sip
import numpy as N
from .. import qtall as qt

from .utilfuncs import rrepr
from .version import version
from ..compat import citems, curlrequest, curlencode

"""Feedback module for providing information about usage.

Note: careful not to send a unique identifier and to reset counts to
ensure lack of traceability.
"""

# patch this to disable any feedback
disableFeedback=False

# for QSettings
_org='veusz.org'
_app='veusz-feedback'
_url='https://barmag.net/veusz-feedback/'

# min send interval in days
_mininterval = 7
# min interval to try sending in days
_minattemptinterval = 1

class Feedback:
    """Keep track of number of activities."""

    def __init__(self):
        # counts of widget creation
        self.widgetcts = defaultdict(int)
        # counts of data import
        self.importcts = defaultdict(int)
        # counts of data export
        self.exportcts = defaultdict(int)

# singleton
feedback = Feedback()

@atexit.register
def updatects():
    """Add saved counts with values from app."""
    #print("running updates")
    setn = qt.QSettings(_org, _app)

    # get statistics and reset in config file
    widgetcts = eval(setn.value('counts/widget', '{}'))
    importcts = eval(setn.value('counts/import', '{}'))
    exportcts = eval(setn.value('counts/export', '{}'))

    # add existing counts
    for k, v in citems(feedback.widgetcts):
        widgetcts[k] = widgetcts.get(k, 0) + v
    for k, v in citems(feedback.importcts):
        importcts[k] = importcts.get(k, 0) + v
    for k, v in citems(feedback.exportcts):
        exportcts[k] = exportcts.get(k, 0) + v

    setn.setValue('counts/widget', rrepr(widgetcts))
    setn.setValue('counts/import', rrepr(importcts))
    setn.setValue('counts/export', rrepr(exportcts))

class FeedbackCheckThread(qt.QThread):
    """Async thread to send feedback."""

    def run(self):
        from ..setting import settingdb

        # exit if disabled
        if (settingdb['feedback_disabled'] or
            disableFeedback or
            not settingdb['feedback_asked_user']):
            #print('disabled')
            return

        setn = qt.QSettings(_org, _app)

        # keep track of when we successfully sent the data (lastsent)
        # and when we last tried (lastattempt), so we don't send too
        # often

        today = datetime.date.today()
        today_tpl = (today.year, today.month, today.day)

        # don't try to send too often
        lastattempt = setn.value('last-attempt', '(2000,1,1)')
        lastattempt = datetime.date(*eval(lastattempt))
        delta_attempt = (today-lastattempt).days
        if delta_attempt<_minattemptinterval:
            #print("too soon 1")
            return

        lastsent = setn.value('last-sent')
        if not lastsent:
            delta_sent = -1
        else:
            lastsent = datetime.date(*eval(lastsent))
            delta_sent = (today-lastsent).days

            # are we within the send period
            if delta_sent<_mininterval:
                #print("too soon 2")
                return

        # avoid accessing url too often by updating date first
        setn.setValue('last-attempt', repr(today_tpl))

        # get statistics and reset in config file
        widgetcts = setn.value('counts/widget', '{}')
        importcts = setn.value('counts/import', '{}')
        exportcts = setn.value('counts/export', '{}')

        try:
            winver = str(sys.getwindowsversion())
        except Exception:
            winver = 'N/A'

        # construct post message - these are the data sent to the
        # remote server
        args = {
            'interval': str(delta_sent),
            'veusz-version': version(),
            'python-version': sys.version,
            'python-version_info': repr(tuple(sys.version_info)),
            'python-platform': sys.platform,
            'platform-machine': platform.machine(),
            'windows-version': winver,
            'numpy-version': N.__version__,
            'qt-version': qt.qVersion(),
            'pyqt-version': qt.PYQT_VERSION_STR,
            'sip-version': sip.SIP_VERSION_STR,
            'locale': qt.QLocale().name(),
            'widgetcts': widgetcts,
            'importcts': importcts,
            'exportcts': exportcts,
        }
        postdata = curlencode(args).encode('utf8')

        # now post the data
        try:
            f = curlrequest.urlopen(_url, postdata)
            retn = f.readline().decode('utf8').strip()
            f.close()

            if retn == 'ok':
                #print("success")
                # reset in stats file and set date last done
                setn.setValue('counts/widget', '{}')
                setn.setValue('counts/import', '{}')
                setn.setValue('counts/export', '{}')
                setn.setValue('last-sent', repr(today_tpl))

        except Exception as e:
            #print("failure",e)
            pass
