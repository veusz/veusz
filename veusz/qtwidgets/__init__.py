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

"""Veusz qtwidgets module."""

import sys
from . import historycombo
from . import historycheck
from . import historyvaluecombo
from . import historygroupbox
from . import historyspinbox
from . import recentfilesbutton
from . import lineeditwithclear
from . import clicklabel

# insert widgets into the list of modules so that it can be found
# by loadUi (any better way to do this?)
sys.modules['historycombo'] = historycombo
sys.modules['historycheck'] = historycheck
sys.modules['historyvaluecombo'] = historyvaluecombo
sys.modules['historygroupbox'] = historygroupbox
sys.modules['historyspinbox'] = historyspinbox
sys.modules['recentfilesbutton'] = recentfilesbutton
sys.modules['lineeditwithclear'] = lineeditwithclear
sys.modules['clicklabel'] = clicklabel
