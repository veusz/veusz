#    Copyright (C) 2008 Jeremy S. Sanders
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
##############################################################################

"""Veusz dialogs module."""

# load custom widgets
from .. import qtwidgets
from .. import document
from .veuszdialog import recreate_register

# lazy loading recreation routines
def _lazy_recreate_1d(*args):
    from .datacreate import recreateDataset
    recreateDataset(*args)
def _lazy_recreate_2d(*args):
    from .datacreate2d import recreateDataset
    recreateDataset(*args)
def _lazy_recreate_histo(*args):
    from .histodata import recreateDataset
    recreateDataset(*args)
def _lazy_recreate_plugin(*args):
    from .plugin import recreateDataset
    recreateDataset(*args)

for kls, fn in (
        (document.DatasetExpression, _lazy_recreate_1d),
        (document.DatasetRange, _lazy_recreate_1d),
        (document.Dataset2DXYZExpression, _lazy_recreate_2d),
        (document.Dataset2DExpression, _lazy_recreate_2d),
        (document.Dataset2DXYFunc, _lazy_recreate_2d),
        (document.DatasetHistoValues, _lazy_recreate_histo),
        (document.DatasetHistoBins, _lazy_recreate_histo),
        (document.Dataset1DPlugin, _lazy_recreate_plugin),
        (document.Dataset2DPlugin, _lazy_recreate_plugin),
        (document.DatasetTextPlugin, _lazy_recreate_plugin),
        ):
    recreate_register[kls] = fn
