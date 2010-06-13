// Copyright (C) 2010 Jeremy Sanders
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
////////////////////////////////////////////////////////////////////////

#ifndef POLYLINECLIP_HH
#define POLYLINECLIP_HH

#include <QRectF>
#include <QPainter>
#include <QPolygonF>

// plot a polyline poly on the painter, clipping by the rectangle given
// if autoexpand is true, then the rectangle is expanded by the line width
void plotClippedPolyline(QPainter& painter,
			 QRectF clip,
			 const QPolygonF& poly,
			 bool autoexpand = true);

// clip a line made up of the points given, returning true
// if is in region or false if not
bool clipLine(const QRectF& clip, QPointF& pt1, QPointF& pt2);

#endif
