# __init__.py file for utils
  
#    Copyright (C) 2004 Jeremy S. Sanders
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

from .version import version
from .textrender import Renderer, FontMetrics, latexEscape
from .safe_eval import compileChecked, SafeEvalException
from .fitlm import fitLM

from .utilfuncs import *
from .points import *
from .action import *
from .dates import *
from .formatting import *
from .colormap import *
from .extbrushfilling import *

try:
    from ..helpers.qtloops import addNumpyToPolygonF, plotPathsToPainter, \
        plotLinesToPainter, plotClippedPolyline, polygonClip, \
        plotClippedPolygon, plotBoxesToPainter, addNumpyPolygonToPath, \
        resampleLinearImage, RotatedRectangle, RectangleOverlapTester
except ImportError:
    from .slowfuncs import addNumpyToPolygonF, plotPathsToPainter, \
        plotLinesToPainter, plotClippedPolyline, polygonClip, \
        plotClippedPolygon, plotBoxesToPainter, addNumpyPolygonToPath, \
        resampleLinearImage, RotatedRectangle, RectangleOverlapTester
