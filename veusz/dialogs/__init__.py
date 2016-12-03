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
from .. import datasets
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
def _lazy_recreate_filtered(*args):
    from .filterdialog import recreateDataset
    recreateDataset(*args)
def _lazy_recreate_plugin(*args):
    from .plugin import recreateDataset
    recreateDataset(*args)

for kls, fn in (
        (datasets.DatasetExpression, _lazy_recreate_1d),
        (datasets.DatasetRange, _lazy_recreate_1d),
        (datasets.Dataset2DXYZExpression, _lazy_recreate_2d),
        (datasets.Dataset2DExpression, _lazy_recreate_2d),
        (datasets.Dataset2DXYFunc, _lazy_recreate_2d),
        (datasets.DatasetHistoValues, _lazy_recreate_histo),
        (datasets.DatasetHistoBins, _lazy_recreate_histo),
        (datasets.DatasetFiltered, _lazy_recreate_filtered),
        (datasets.Dataset1DPlugin, _lazy_recreate_plugin),
        (datasets.Dataset2DPlugin, _lazy_recreate_plugin),
        (datasets.DatasetNDPlugin, _lazy_recreate_plugin),
        (datasets.DatasetTextPlugin, _lazy_recreate_plugin),
        (datasets.DatasetDateTimePlugin, _lazy_recreate_plugin),
        ):
    recreate_register[kls] = fn
