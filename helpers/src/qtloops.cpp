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
#include "isnan.h"
#include "polylineclip.h"

#include <math.h>

#include <QPointF>
#include <QVector>
#include <QLineF>
#include <QPen>
#include <QTransform>
#include <QColor>

namespace
{
  // is difference between points very small?
  inline bool smallDelta(const QPointF& pt1, const QPointF& pt2)
  {
    return fabs(pt1.x() - pt2.x()) < 0.01 &&
      fabs(pt1.y()- pt2.y()) < 0.01;
  }

  template <class T> inline T min(T a, T b)
  {
    return (a<b) ? a : b;
  }

  template <class T> inline T min(T a, T b, T c, T d)
  {
    return min( min(a, b), min(c, d) );
  }

  template <class T> inline T clipval(T val, T minv, T maxv)
  {
    if( val < minv ) return minv;
    if( val > maxv ) return maxv;
    return val;
  }
}

void addNumpyToPolygonF(QPolygonF& poly, const Tuple2Ptrs& d)
{
  // iterate over rows until none left
  const int numcols = d.data.size();
  QPointF lastpt(-1e6, -1e6);
  for(int row=0 ; ; ++row)
    {
      bool ifany = false;
      // the numcols-1 makes sure we don't get odd numbers of columns
      for(int col=0; col < (numcols-1); col += 2)
	{
	  // add point if point in two columns
	  if( row < d.dims[col] && row < d.dims[col+1] )
	    {
	      const QPointF pt(d.data[col][row], d.data[col+1][row]);
	      if( ! smallDelta(pt, lastpt) )
		{
		  poly << pt;
		  lastpt = pt;
		}
	      ifany = true;
	    }
	}
      // exit loop if no more columns
      if(! ifany )
	break;
    }
}

void plotPathsToPainter(QPainter& painter, QPainterPath& path,
			const Numpy1DObj& x, const Numpy1DObj& y,
			const Numpy1DObj* scaling,
			const QRectF* clip,
			const QImage* colorimg)
{
  QRectF cliprect( QPointF(-32767,-32767), QPointF(32767,32767) );
  if( clip != 0 )
    {
      qreal x1, y1, x2, y2;
      clip->getCoords(&x1, &y1, &x2, &y2);
      cliprect.setCoords(x1, y1, x2, y2);
    }
  QRectF pathbox = path.boundingRect();
  cliprect.adjust(pathbox.left(), pathbox.top(),
		  pathbox.bottom(), pathbox.right());

  // keep track of duplicate points
  QPointF lastpt(-1e6, -1e6);
  // keep original transformation for restoration after each iteration
  QTransform origtrans(painter.worldTransform());

  // number of iterations
  int size = min(x.dim, y.dim);

  // if few color points, trim down number of paths
  if( colorimg != 0 )
    size = min(size, colorimg->width());
  // too few scaling points
  if( scaling != 0 )
    size = min(size, scaling->dim);

  // draw each path
  for(int i = 0; i < size; ++i)
    {
      const QPointF pt(x(i), y(i));
      if( cliprect.contains(pt) && ! smallDelta(lastpt, pt) )
	{
	  painter.translate(pt);
	  if( scaling != 0 )
	    {
	      // scale point if requested
	      const qreal s = (*scaling)(i);
	      painter.scale(s, s);
	    }
	  if( colorimg != 0 )
	    {
	      // get color from pixel and create a new brush
	      QBrush b( QColor::fromRgba(colorimg->pixel(i, 0)) );
	      painter.setBrush(b);
	    }

	  painter.drawPath(path);
	  painter.setWorldTransform(origtrans);
	  lastpt = pt;
	}
    }
}

void plotLinesToPainter(QPainter& painter,
			const Numpy1DObj& x1, const Numpy1DObj& y1,
			const Numpy1DObj& x2, const Numpy1DObj& y2,
			const QRectF* clip, bool autoexpand)
{
  const int maxsize = min(x1.dim, x2.dim, y1.dim, y2.dim);

  // if autoexpand, expand rectangle by line width
  QRectF clipcopy;
  if ( clip != 0 && autoexpand )
    {
      const qreal lw = painter.pen().widthF();
      qreal x1, y1, x2, y2;
      clip->getCoords(&x1, &y1, &x2, &y2);
      clipcopy.setCoords(x1, y1, x2, y2);
      clipcopy.adjust(-lw, -lw, lw, lw);
    }

  if( maxsize != 0 )
    {
      QVector<QLineF> lines;
      for(int i = 0; i < maxsize; ++i)
	{
	  QPointF pt1(x1(i), y1(i));
	  QPointF pt2(x2(i), y2(i));
	  if( clip != 0 )
	    {
	      if( clipLine(clipcopy, pt1, pt2) )
		lines << QLineF(pt1, pt2);
	    }
	  else
	    lines << QLineF(pt1, pt2);
	}

      painter.drawLines(lines);
    }
}

void plotBoxesToPainter(QPainter& painter,
			const Numpy1DObj& x1, const Numpy1DObj& y1,
			const Numpy1DObj& x2, const Numpy1DObj& y2,
			const QRectF* clip, bool autoexpand)
{
  // if autoexpand, expand rectangle by line width
  QRectF clipcopy(QPointF(-32767,-32767), QPointF(32767,32767));
  if ( clip != 0 && autoexpand )
    {
      const qreal lw = painter.pen().widthF();
      qreal x1, y1, x2, y2;
      clip->getCoords(&x1, &y1, &x2, &y2);
      clipcopy.setCoords(x1, y1, x2, y2);
      clipcopy.adjust(-lw, -lw, lw, lw);
    }

  const int maxsize = min(x1.dim, x2.dim, y1.dim, y2.dim);

  QVector<QRectF> rects;
  for(int i = 0; i < maxsize; ++i)
    {
      QPointF pt1(x1(i), y1(i));
      QPointF pt2(x2(i), y2(i));
      const QRectF rect(pt1, pt2);

      if( clipcopy.intersects(rect) )
	{
	  rects << clipcopy.intersected(rect);
	}
    }

  if( ! rects.isEmpty() )
    painter.drawRects(rects);
}

QImage numpyToQImage(const Numpy2DObj& imgdata, const Numpy2DIntObj &colors,
		     bool forcetrans)
{
  // make format use alpha transparency if required
  const int numcolors = colors.dims[0];
  if ( colors.dims[1] != 4 )
    throw "4 columns required in colors array";
  const int numbands = numcolors-1;
  const int xw = imgdata.dims[1];
  const int yw = imgdata.dims[0];

  QImage::Format format = QImage::Format_RGB32;
  if( forcetrans )
    format = QImage::Format_ARGB32;
  else
    {
      for(int i = 0; i < numcolors; ++i)
	{
	  if( colors(i, 3) != 255 )
	    format = QImage::Format_ARGB32;
	}
    }

  // make image
  QImage img(xw, yw, format);

  // iterate over input pixels
  for(int y=0; y<yw; ++y)
    {
      // direction of images is different for qt and numpy image
      QRgb* scanline = reinterpret_cast<QRgb*>(img.scanLine(yw-y-1));
      for(int x=0; x<xw; ++x)
	{
	  double val = imgdata(x, y);
	  if( ! isFinite(val) )
	    {
	      // transparent
	      *(scanline+x) = qRgba(0, 0, 0, 0);
	    }
	  else
	    {
	      // do linear interpolation between bands
	      // make sure between 0 and 1
	      val = clipval(val, 0., 1.);
	      const int band = clipval(int(val*numbands), 0, numbands-1);
	      const double delta = val*numbands - band;

	      // ensure we don't read beyond where we should
	      const int band2 = min(band + 1, numbands);
	      const double delta1 = 1.-delta;

	      const int b = int(delta1*colors(0, band) +
				delta *colors(0, band2));
	      const int g = int(delta1*colors(1, band) +
				delta *colors(1, band2));
	      const int r = int(delta1*colors(2, band) +
				delta *colors(2, band2));
	      const int a = int(delta1*colors(3, band) +
				delta *colors(3, band2));
	      
	      *(scanline+x) = qRgba(r, g, b, a);
	    }
	}
    }
  return img;
}

void applyImageTransparancy(QImage& img, const Numpy2DObj& data)
{
  const int xw = min(data.dims[0], img.width());
  const int yw = min(data.dims[1], img.height());
  
  for(int y=0; y<yw; ++y)
    {
      // direction of images is different for qt and numpy image
      QRgb* scanline = reinterpret_cast<QRgb*>(img.scanLine(yw-y-1));
      for(int x=0; x<xw; ++x)
	{
	  const double val = clipval(data(x, y), 0., 1.);
	  const QRgb col = *(scanline+x);

	  // update pixel alpha component
	  QRgb newcol = qRgba( qRed(col), qGreen(col), qBlue(col),
			       int(qAlpha(col)*val) );
	  *(scanline+x) = newcol;
	}
    }
}
