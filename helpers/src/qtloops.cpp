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

#include <vector>
#include <algorithm>

void addNumpyToPolygonF(QPolygonF *poly,
			const doublearray_ptr_vec &d)
{
  // get sizes of objects first
  const int numcols = d.size();
  std::vector<int> sizes;
  for(int i=0; i<numcols; i++)
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
			const doublearray* x, const doublearray* y)
{
  const double* xi = & x->operator[](0);
  const double* yi = & y->operator[](0);
  const double* const xend = xi + x->size();
  const double* const yend = yi + y->size();

  for( ; xi != xend && yi != yend; ++xi, ++yi)
    {
      painter->translate(*xi, *yi);
      painter->drawPath(*path);
      painter->translate(-*xi, -*yi);
    }

}

void plotLinesToPainter(QPainter* painter,
			const doublearray* x1, const doublearray* y1,
			const doublearray* x2, const doublearray* y2)
{
  const size_t maxsize = std::min( std::min(x1->size(), y1->size()),
				   std::min(x2->size(), y2->size()) );

  if( maxsize != 0 )
    {
      QVector<QLineF> lines;

      const double* x1p = &((*x1)[0]);
      const double* x2p = &((*x2)[0]);
      const double* y1p = &((*y1)[0]);
      const double* y2p = &((*y2)[0]);
      for( size_t i = 0; i != maxsize; ++i)
	{
	  lines << QLineF(*x1p, *y1p, *x2p, *y2p);
	  ++x1p; ++x2p; ++y1p; ++y2p;
	}
      
      painter->drawLines(lines);
    }
}
