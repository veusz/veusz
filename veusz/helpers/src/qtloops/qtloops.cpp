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

#include <algorithm>

#include "qtloops.h"
#include "isnan.h"
#include "polylineclip.h"
#include "polygonclip.h"

#include <QBrush>
#include <QColor>
#include <QLineF>
#include <QPen>
#include <QPointF>
#include <QTransform>
#include <QVector>

namespace
{
  // is difference between points very small?
  inline bool smallDelta(const QPointF& pt1, const QPointF& pt2)
  {
    return fabs(pt1.x() - pt2.x()) < 0.01 &&
      fabs(pt1.y()- pt2.y()) < 0.01;
  }

  template <class T> inline T min(T a, T b, T c, T d)
  {
    return std::min( std::min(a, b), std::min(c, d) );
  }

  template <class T> inline T clipval(T val, T minv, T maxv)
  {
    if( val < minv ) return minv;
    if( val > maxv ) return maxv;
    return val;
  }

  template <class T> void putinorder(T& a, T& b)
  {
    if(a>b) std::swap(a, b);
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

void addNumpyPolygonToPath(QPainterPath &path, const Tuple2Ptrs& d,
			   const QRectF* clip)
{
  const int numcols = d.data.size();
  for(int row=0 ; ; ++row)
    {
      bool ifany = false;
      // output polygon
      QPolygonF poly;

      // the numcols-1 makes sure we don't get odd numbers of columns
      for(int col=0; col < (numcols-1); col += 2)
	{
	  // add point if point in two columns
	  if( row < d.dims[col] && row < d.dims[col+1] )
	    {
	      const QPointF pt(d.data[col][row], d.data[col+1][row]);
	      poly << pt;
	      ifany = true;
	    }
	}

      if( ifany )
	{
	  if( clip != 0 )
	    {
	      QPolygonF clippedpoly;
	      polygonClip(poly, *clip, clippedpoly);
	      path.addPolygon(clippedpoly);
	    }
	  else
	    {
	      path.addPolygon(poly);
	    }
	  path.closeSubpath();
	}
      else
	{
	  // exit loop if no more columns
	  break;
	}
    }
}

namespace
{

  // Scale path by scale given. Puts output in out.
  void scalePath(const QPainterPath& path, qreal scale, QPainterPath& out)
  {
    const int count = path.elementCount();
    for(int i=0; i<count; ++i)
      {
	const QPainterPath::Element& el = path.elementAt(i);
	if(el.isMoveTo())
	  {
	    out.moveTo(el*scale);
	  }
	else if(el.isLineTo())
	  {
	    out.lineTo(el*scale);
	  }
	else if(el.isCurveTo())
	  {
	    out.cubicTo(el*scale,
			path.elementAt(i+1)*scale,
			path.elementAt(i+2)*scale);
	    i += 2;
	  }
      }
  }

} // namespace

void plotPathsToPainter(QPainter& painter, QPainterPath& path,
			const Numpy1DObj& x, const Numpy1DObj& y,
			const Numpy1DObj* scaling,
			const QRectF* clip,
			const QImage* colorimg,
			bool scaleline)
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
  int size = std::min(x.dim, y.dim);

  // if few color points, trim down number of paths
  if( colorimg != 0 )
    size = std::min(size, colorimg->width());
  // too few scaling points
  if( scaling != 0 )
    size = std::min(size, scaling->dim);

  // draw each path
  for(int i = 0; i < size; ++i)
    {
      const QPointF pt(x(i), y(i));
      if( cliprect.contains(pt) && ! smallDelta(lastpt, pt) )
	{
	  painter.translate(pt);

	  if( colorimg != 0 )
	    {
	      // get color from pixel and create a new brush
	      QBrush b( QColor::fromRgba(colorimg->pixel(i, 0)) );
	      painter.setBrush(b);
	    }

	  if( scaling == 0 )
	    {
	      painter.drawPath(path);
	    }
	  else
	    {
	      // scale point if requested
	      const qreal s = (*scaling)(i);
	      if( scaleline )
		{
		  painter.scale(s, s);
		  painter.drawPath(path);
		}
	      else
		{
		  QPainterPath scaled;
		  scalePath(path, s, scaled);
		  painter.drawPath(scaled);
		}
	    }

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

void addCubicsToPainterPath(QPainterPath& path, const QPolygonF& poly)
{
  QPointF lastpt(-999999, -999999);
  for(int i=0; i<poly.size()-3; i+=4)
    {
      if(lastpt != poly[i])
        path.moveTo(poly[i]);
      path.cubicTo(poly[i+1], poly[i+2], poly[i+3]);
      lastpt = poly[i+3];
    }
}

QImage numpyToQImage(const Numpy2DObj& imgdata, const Numpy2DIntObj &colors,
		     bool forcetrans)
{
  // make format use alpha transparency if required
  const int numcolors = colors.dims[0];
  if ( colors.dims[1] != 4 )
    throw "4 columns required in colors array";
  if ( numcolors < 1 )
    throw "at least 1 color required";
  const int numbands = numcolors-1;
  const int xw = imgdata.dims[1];
  const int yw = imgdata.dims[0];

  // if the first value in the color is -1 then switch to jumping mode
  const bool jumps = colors(0,0) == -1;

  // make image
  QImage img(xw, yw, QImage::Format_ARGB32);

  // does the image use alpha values?
  bool hasalpha = false;

  // iterate over input pixels
  for(int y=0; y<yw; ++y)
    {
      // direction of images is different for qt and numpy image
      QRgb* scanline = reinterpret_cast<QRgb*>(img.scanLine(yw-y-1));
      for(int x=0; x<xw; ++x)
	{
	  double val = imgdata(x, y);

	  // output color
	  int b, g, r, a;

	  if( ! isFinite(val) )
	    {
	      // transparent
	      b = g = r = a = 0;
	    }
	  else
	    {
	      val = clipval(val, 0., 1.);

	      if( jumps )
		{
		  // jumps between colours in discrete mode
		  // (ignores 1st color, which signals this mode)
		  const int band = clipval(int(val*(numcolors-1))+1, 1,
					   numcolors-1);

		  b = colors(0, band);
		  g = colors(1, band);
		  r = colors(2, band);
		  a = colors(3, band);
		}
	      else
		{
		  // do linear interpolation between bands
		  // make sure between 0 and 1

		  const int band = clipval(int(val*numbands), 0, numbands-1);
		  const double delta = val*numbands - band;

		  // ensure we don't read beyond where we should
		  const int band2 = std::min(band + 1, numbands);
		  const double delta1 = 1.-delta;

                  // we add 0.5 before truncating to round to nearest int
		  b = int(delta1*colors(0, band) +
			  delta *colors(0, band2) + 0.5);
		  g = int(delta1*colors(1, band) +
			  delta *colors(1, band2) + 0.5);
		  r = int(delta1*colors(2, band) +
			  delta *colors(2, band2) + 0.5);
		  a = int(delta1*colors(3, band) +
			  delta *colors(3, band2) + 0.5);
	      
		}
	    }

          if(a != 255)
            hasalpha = true;

	  *(scanline+x) = qRgba(r, g, b, a);
	}
    }


  if(!hasalpha)
    {
      // return image without transparency for speed / space improvements
#if QT_VERSION >= QT_VERSION_CHECK(5, 9, 0)
      // recent qt version
      // just change the format to the non-transparent version
      img.reinterpretAsFormat(QImage::Format_RGB32);
#else
      // do slower conversion of data
      return img.convertToFormat(QImage::Format_RGB32);
#endif
    }

  return img;
}

void applyImageTransparancy(QImage& img, const Numpy2DObj& data)
{
  const int xw = std::min(data.dims[1], img.width());
  const int yw = std::min(data.dims[0], img.height());
  
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

QImage resampleNonlinearImage(const QImage& img,
                              int x0, int y0,
                              int x1, int y1,
                              const Numpy1DObj& xedge,
                              const Numpy1DObj& yedge)
{
  putinorder(x0, x1);
  putinorder(y0, y1);

  const int xw = x1-x0;
  const int yw = y1-y0;

  QImage outimg(xw, yw, img.format());

  int iy=0;
  for(int oy=0; oy<yw; ++oy)
    {
      while( yedge(yedge.dim-2-iy)<=oy+y0+0.5 && iy<yedge.dim-1 )
        ++iy;

      QRgb* oscanline = reinterpret_cast<QRgb*>(outimg.scanLine(oy));
      const QRgb* iscanline = reinterpret_cast<const QRgb*>(img.scanLine(iy));

      int ix=0;
      for(int ox=0; ox<xw; ++ox)
        {
          while( xedge(ix+1)<=ox+x0+0.5 && ix<xedge.dim-1 )
            ++ix;

          oscanline[ox] = iscanline[ix];
        }
    }

  return outimg;
}

void plotImageAsRects(QPainter& painter, const QRectF& bounds, const QImage& img)
{
  const int width=img.width();
  const int height=img.height();
  if(width<=0 || height<=0)
    return;

  const qreal dx = bounds.width() / width;
  const qreal dy = bounds.height() / height;
  const qreal x0 = bounds.left();
  const qreal y0 = bounds.top();

  const QRectF cliprect = painter.clipBoundingRect();
  const bool clipped = ! cliprect.isEmpty();

  painter.save();
  for(int y=0; y<height; ++y)
    for(int x=0; x<width; ++x)
      {
        QRectF r(x0+x*dx, y0+y*dy, dx, dy);
        if(clipped)
          r &= cliprect;

        if(r.isValid())
          {
            const QColor col(img.pixelColor(x, y));
            const int alpha = col.alpha();
            if(alpha == 0)
              {
                // transparent
              }
            else if(alpha == 255)
              {
                // opaque, so draw line to avoid antialiasing gaps round
                // boxes
                painter.setPen(QPen(QBrush(col), 0.));
                painter.setBrush(QBrush(col));
                painter.drawRect(r);
              }
            else
              {
                painter.fillRect(r, col);
              }
          }
      }
  painter.restore();
}

void plotNonlinearImageAsBoxes(QPainter& painter,
                               const QImage& img,
                               const Numpy1DObj& xedges,
                               const Numpy1DObj& yedges)
{
  const int width=img.width();
  const int height=img.height();

  // safety
  if( xedges.dim != width+1 || yedges.dim != height+1 )
    throw "Number of edges did not match image size";

  const QRectF cliprect = painter.clipBoundingRect();
  const bool clipped = ! cliprect.isEmpty();

  painter.save();
  for(int y=0; y<height; ++y)
    for(int x=0; x<width; ++x)
      {
        const qreal x0 = std::min(xedges(x), xedges(x+1));
        const qreal x1 = std::max(xedges(x), xedges(x+1));
        const qreal y0 = std::min(yedges(y), yedges(y+1));
        const qreal y1 = std::max(yedges(y), yedges(y+1));

        QRectF r(x0, y0, x1-x0, y1-y0);
        if(clipped)
          r &= cliprect;

        if(r.isValid())
          {
            // note: axis coordinates are reversed wrt to image
            const QColor col(img.pixelColor(x, height-1-y));

            const int alpha = col.alpha();
            if(alpha == 0)
              {
                // transparent
              }
            else if(alpha == 255)
              {
                // opaque, so draw line to avoid antialiasing gaps round
                // boxes
                painter.setPen(QPen(QBrush(col), 0.));
                painter.setBrush(QBrush(col));
                painter.drawRect(r);
              }
            else
              {
                painter.fillRect(r, col);
              }
          }
      }
  painter.restore();
}
