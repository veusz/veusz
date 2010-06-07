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

#include "qtloops.h"

#include <QPointF>
#include <QVector>
#include <QLineF>
#include <QPen>

#include <vector>
#include <algorithm>

#include <polylineclip.h>

void addNumpyToPolygonF(QPolygonF *poly,
			const doublearray_ptr_vec &d)
{
  // get sizes of objects first
  const int numcols = d.size();
  std::vector<int> sizes;
  for(int i = 0; i != numcols; ++i)
    sizes.push_back(d[i]->size());

  // iterate over rows until none left
  for(int row=0 ; ; ++row)
    {
      bool ifany = false;
      // the numcols-1 makes sure we don't get odd numbers of columns
      for(int col=0; col < (numcols-1); col += 2)
	{
	  // add point if point in two columns
	  if( row < sizes[col] && row < sizes[col+1] )
	    {
	      (*poly) << QPointF( (*d[col])[row], (*d[col+1])[row] );
	      ifany = true;
	    }
	 }
      // exit loop if no more columns
      if(! ifany )
	break;
    }
}

void plotPathsToPainter(QPainter* painter, QPainterPath* path,
			const doublearray* x, const doublearray* y,
			const QRectF* clip)
{
  QRectF cliprect( QPointF(-32767,-32767), QPointF(32767,32767) );
  if( clip != 0 )
    {
      qreal x1, y1, x2, y2;
      clip->getCoords(&x1, &y1, &x2, &y2);
      cliprect.setCoords(x1, y1, x2, y2);
    }
  QRectF pathbox = path->boundingRect();
  cliprect.adjust(pathbox.left(), pathbox.top(),
		  pathbox.bottom(), pathbox.right());

  const size_t size = std::min(x->size(), y->size());
  for(size_t i = 0; i != size; ++i)
    {
      const QPointF pt((*x)[i], (*y)[i]);
      if( cliprect.contains(pt) )
	{
	  painter->translate(pt);
	  painter->drawPath(*path);
	  painter->translate(-pt);
	}
    }
}

void plotLinesToPainter(QPainter* painter,
			const doublearray* x1, const doublearray* y1,
			const doublearray* x2, const doublearray* y2,
			const QRectF* clip, bool autoexpand)
{
  const size_t maxsize = std::min( std::min(x1->size(), y1->size()),
				   std::min(x2->size(), y2->size()) );

  // if autoexpand, expand rectangle by line width
  QRectF clipcopy;
  if ( clip != 0 and autoexpand )
    {
      const qreal lw = painter->pen().widthF();
      qreal x1, y1, x2, y2;
      clip->getCoords(&x1, &y1, &x2, &y2);
      clipcopy.setCoords(x1, y1, x2, y2);
      clipcopy.adjust(-lw, -lw, lw, lw);
    }

  if( maxsize != 0 )
    {
      QVector<QLineF> lines;
      for(size_t i = 0; i != maxsize; ++i)
	{
	  QPointF pt1((*x1)[i], (*y1)[i]);
	  QPointF pt2((*x2)[i], (*y2)[i]);
	  if( clip != 0 )
	    {
	      if( clipLine(clipcopy, pt1, pt2) )
		lines << QLineF(pt1, pt2);
	    }
	  else
	    lines << QLineF(pt1, pt2);
	}

      painter->drawLines(lines);
    }
}

void plotBoxesToPainter(QPainter* painter,
			const doublearray* x1, const doublearray* y1,
			const doublearray* x2, const doublearray* y2,
			const QRectF* clip, bool autoexpand)
{
  // if autoexpand, expand rectangle by line width
  QRectF clipcopy(QPointF(-32767,-32767), QPointF(32767,32767));
  if ( clip != 0 and autoexpand )
    {
      const qreal lw = painter->pen().widthF();
      qreal x1, y1, x2, y2;
      clip->getCoords(&x1, &y1, &x2, &y2);
      clipcopy.setCoords(x1, y1, x2, y2);
      clipcopy.adjust(-lw, -lw, lw, lw);
    }

  const size_t maxsize = std::min( std::min(x1->size(), y1->size()),
				   std::min(x2->size(), y2->size()) );

  QVector<QRectF> rects;
  for(size_t i = 0; i != maxsize; ++i)
    {
      const QPointF pt1((*x1)[i], (*y1)[i]);
      const QPointF pt2((*x2)[i], (*y2)[i]);
      const QRectF rect(pt1, pt2);

      if( clipcopy.intersects(rect) )
	{
	  rects << clipcopy.intersected(rect);
	}
    }

  if( ! rects.isEmpty() )
    painter->drawRects(rects);
}
