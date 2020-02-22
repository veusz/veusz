// -*- mode: C++; -*-

#ifndef QTLOOPS_H
#define QTLOOPS_H

//    Copyright (C) 2009 Jeremy S. Sanders
//    Email: Jeremy Sanders <jeremy@jeremysanders.net>
//
//    This program is free software; you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation; either version 2 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License along
//    with this program; if not, write to the Free Software Foundation, Inc.,
//    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
/////////////////////////////////////////////////////////////////////////////

#include "qtloops_helpers.h"

#include <QPolygonF>
#include <QPainter>
#include <QPainterPath>
#include <QRectF>
#include <QImage>

class QtLoops {
public:
  QtLoops() {};
};

// add sets of points to a QPolygonF
void addNumpyToPolygonF(QPolygonF& poly,
			const Tuple2Ptrs& v);

// add sets of polygon points to a path
void addNumpyPolygonToPath(QPainterPath &path, const Tuple2Ptrs& d,
			   const QRectF* clip = 0);

// plot paths to painter
// x and y locations are given in x and y
// if scaling is not 0, is an array to scale the data points by
// if colorimg is not 0, is a Nx1 image containing color points for path fills
// clip is a clipping rectangle if set
void plotPathsToPainter(QPainter& painter, QPainterPath& path,
			const Numpy1DObj& x, const Numpy1DObj& y,
			const Numpy1DObj* scaling = 0,
			const QRectF* clip = 0,
			const QImage* colorimg = 0,
			bool scaleline = false);

void plotLinesToPainter(QPainter& painter,
			const Numpy1DObj& x1, const Numpy1DObj& y1,
			const Numpy1DObj& x2, const Numpy1DObj& y2,
			const QRectF* clip = 0, bool autoexpand = true);

void plotBoxesToPainter(QPainter& painter,
			const Numpy1DObj& x1, const Numpy1DObj& y1,
			const Numpy1DObj& x2, const Numpy1DObj& y2,
			const QRectF* clip = 0, bool autoexpand = true);

// add polygon to painter path as a cubic
void addCubicsToPainterPath(QPainterPath& path, const QPolygonF& poly);

QImage numpyToQImage(const Numpy2DObj& data, const Numpy2DIntObj &colors,
		     bool forcetrans = false);

void applyImageTransparancy(QImage& img, const Numpy2DObj& data);

QImage resampleNonlinearImage(const QImage& img,
                              int x0, int y0,
                              int x1, int y1,
                              const Numpy1DObj& xedge,
                              const Numpy1DObj& yedge);

// plot image as a set of rectangles
void plotImageAsRects(QPainter& painter, const QRectF& bounds, const QImage& img);

// plot a non linear image as a set of boxes
// the coordinates for each edge are given in xedges/yedges
void plotNonlinearImageAsBoxes(QPainter& painter,
                               const QImage& img,
                               const Numpy1DObj& xedges,
                               const Numpy1DObj& yedges);

#endif
