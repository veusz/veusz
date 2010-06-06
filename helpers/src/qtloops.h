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
#include <vector>
#include <valarray>
#include <QPolygonF>
#include <QPainter>
#include <QPainterPath>
#include <QRectF>

class QtLoops {
public:
  QtLoops() {};
};

void addNumpyToPolygonF(QPolygonF* poly,
			const doublearray_ptr_vec &v);

void plotPathsToPainter(QPainter* painter, QPainterPath* path,
			const doublearray* x, const doublearray* y,
			const QRectF* clip = 0);

void plotLinesToPainter(QPainter* painter,
			const doublearray* x1, const doublearray* y1,
			const doublearray* x2, const doublearray* y2,
			const QRectF* clip = 0, bool autoexpand = true);

#endif
