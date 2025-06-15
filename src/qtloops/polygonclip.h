// -*- mode: C++; -*-

//    Copyright (C) 2010 Jeremy Sanders
//    Email: Jeremy Sanders <jeremy@jeremysanders.net>
//
//    This file is part of Veusz.
//
//    Veusz is free software: you can redistribute it and/or modify it
//    under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 2 of the License, or
//    (at your option) any later version.
//
//    Veusz is distributed in the hope that it will be useful, but
//    WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
//    General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with Veusz. If not, see <https://www.gnu.org/licenses/>.
//
//////////////////////////////////////////////////////////////////////////////

#ifndef POLYGONCLIP_HH
#define POLYGONCLIP_HH

#include <QPointF>
#include <QRectF>
#include <QPolygonF>
#include <QPainter>

// clip input polygon to clipping rectangle, filling outpoly
void polygonClip(const QPolygonF& inpoly,
		 const QRectF& cliprect,
		 QPolygonF& outpoly);

// plot a clipped polygon to painter
void plotClippedPolygon(QPainter& painter,
			QRectF rect,
			const QPolygonF& inpoly,
			bool autoexpand = true);

#endif
