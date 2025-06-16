#    Copyright (C) 2010 Jeremy S. Sanders
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

from . import field
from . import datasetplugin
from . import importplugin
from . import toolsplugin
from . import votable

from .field import *
from .datasetplugin import *
from .importplugin import *
from .toolsplugin import *
from .votable import *

# backward compatibility
ImportDataset1D = Dataset1D
ImportDataset2D = Dataset2D
ImportDatasetText = DatasetText
ImportField = Field
ImportFieldCheck = FieldBool
ImportFieldText = FieldText
ImportFieldFloat = FieldFloat
ImportFieldInt = FieldInt
ImportFieldCombo = FieldCombo
