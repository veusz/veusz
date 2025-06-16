# __init__.py file for utils

#    Copyright (C) 2004 Jeremy S. Sanders
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

from .version import *
from .textrender import Renderer, FontMetrics, latexEscape
from .safe_eval import compileChecked, SafeEvalException
from .fitlm import fitLM

from .utilfuncs import *
from .points import getPointPainterPath, MarkerCodes, plotMarkers, \
    plotMarker, ArrowCodes, plotLineArrow
from .action import *
from .dates import *
from .formatting import *
from .colormap import *
from .extbrushfilling import *
from .feedback import feedback, FeedbackCheckThread, disableFeedback

from ..helpers.qtloops import addNumpyToPolygonF, plotPathsToPainter, \
    plotLinesToPainter, plotClippedPolyline, polygonClip, \
    plotClippedPolygon, plotBoxesToPainter, addNumpyPolygonToPath, \
    RotatedRectangle, RectangleOverlapTester
