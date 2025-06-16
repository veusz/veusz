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

"""Widgets are defined in this module."""

from .widget import Widget, Action
from .axis import Axis
from .axisbroken import AxisBroken
from .axisfunction import AxisFunction
from .graph import Graph
from .grid import Grid
from .plotters import GenericPlotter, FreePlotter
from .pickable import PickInfo
from .point import PointPlotter
from .function import FunctionPlotter
from .histo import Histo
from .textlabel import TextLabel
from .page import Page
from .root import Root
from .key import Key
from .fit import Fit
from .image import Image
from .contour import Contour
from .colorbar import ColorBar
from .shape import Shape, BoxShape, Rectangle, Ellipse, ImageFile, SVGFile
from .line import Line
from .bar import BarPlotter
from .polygon import Polygon
from .vectorfield import VectorField
from .boxplot import BoxPlot
from .polar import Polar
from .ternary import Ternary
from .nonorthpoint import NonOrthPoint
from .nonorthfunction import NonOrthFunction
from .scene3d import Scene3D
from .graph3d import Graph3D
from .axis3d import Axis3D
from .plotters3d import GenericPlotter3D
from .function3d import Function3D
from .point3d import Point3D
from .surface3d import Surface3D
from .covariance import Covariance
from .volume3d import Volume3D
