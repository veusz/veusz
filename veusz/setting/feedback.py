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

import sip
import numpy as N
from .. import qtall as qt

from .. import utils
from .settingdb import settingdb
from ..compat import citems, curlrequest

"""Feedback module for providing information about usage.

Note: careful not to send a unique identifier and to reset counts to
ensure lack of traceability.
"""

# patch this to disable any feedback
disable_feedback=False

# 'empty' dictionary to make sure QVariant is using a text hash map
_dummy = {'dummy': 0}

# for QSettings
org='veusz.org'
app='veusz-feedback'

class SendThread(qt.QThread):
    # check interval in days
    mininterval = 7

    def run(self):
        setn = qt.QSettings(org, app)

        lastcheck = setn.value("last-check")
        if not lastcheck.year:
            delta_days = -1
        else:
            delta_days = (datetime.date.today()-lastcheck).days
            if delta_days < mininterval:
                return

        # get statistics and reset in config file
        widgetcts = setn.value('widgetcts', dummy)
        importcts = setn.value('importcts', dummy)
        exportcts = setn.value('exportcts', dummy)

        # construct post message
        args = {
            'interval': str(delta_days),
            'veusz-version': utils.version(),
            'python-version': sys.version,
            'python-platform': sys.platform,
            'numpy-version': N.__version__,
            'qt-version': qt.qVersion(),
            'pyqt-version': qt.PYQT_VERSION_STR,
            'sip-version': sip.SIP_VERSION_STR,
        }
        for k, v in citems(widgetcts):
            args['widgetcts_'+k] = str(v)
        for k, v in citems(importcts):
            args['importcts_'+k] = str(v)
        for k, v in citems(exportcts):
            args['exportcts_'+k] = str(v)

        print(args)

        # reset in stats file
        setn.setValue('last-check', datetime.date.today())
        setn.setValue('widgetcts', dummy)
        setn.setValue('importcts', dummy)
        setn.setValue('exportcts', dummy)

        return args

class Feedback:
    def __init__(self):

        # counts of widget creation
        self.widgetcts = defaultdict(int)
        # counts of data import
        self.importcts = defaultdict(int)
        # counts of data export
        self.exportcts = defaultdict(int)

    def checkLastSend(self, interval=7):
        """Check whether to do a send."""

        if (settingdb['feedback_disabled'] or
            or disable_feedback or not settingdb['feedback_asked_user']):
            return

