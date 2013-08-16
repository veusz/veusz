// Copyright (C) 2010 Jeremy Sanders

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
// 02110-1301, USA.
////////////////////////////////////////////////////////////////////

#include <QPointF>
#include <QPen>
#include <cmath>

#include <polylineclip.h>

using std::fabs;

// Cohen-Sutherland line clipping algorithm

// codes used classify endpoints are combinations of these
#define LEFT 1
#define RIGHT 2
#define TOP 4
#define BOTTOM 8

namespace
{
  // compute intersection with horizontal line
  inline QPointF horzIntersection(qreal yval, const QPointF& pt1,
				  const QPointF& pt2)
  {
    if( pt1.y() == pt2.y() )
      // line is vertical
      return QPointF( pt1.x(), yval );
    else
      return QPointF( pt1.x() + (yval-pt1.y()) *
		      (pt2.x()-pt1.x()) / (pt2.y()-pt1.y()),
		      yval );
  }

  // compute intersection with vertical line
  inline QPointF vertIntersection(qreal xval, const QPointF& pt1,
				  const QPointF& pt2)
  {
    if( pt1.x() == pt2.x() )
      // line is horizontal
      return QPointF( xval, pt1.y() );
    else
      return QPointF( xval,
		      pt1.y() + (xval-pt1.x()) *
		      (pt2.y()-pt1.y()) / (pt2.x()-pt1.x()) );
  }

  class Clipper
  {
  public:
    Clipper(const QRectF& cliprect)
      : clip(cliprect)
    {
    }

    // compute the Cohen-Sutherland code
    unsigned computeCode(const QPointF& pt) const
    {
      unsigned code = 0;
      if (pt.x() < clip.left())
	code |= LEFT;
      else if (pt.x() > clip.right())
	code |= RIGHT;
      if (pt.y() < clip.top())
	code |= TOP;
      else if (pt.y() > clip.bottom())
	code |= BOTTOM;
      return code;
    }

    // get consistent clipping on different platforms by making line
    // edges meet clipping box if close
    void fixPt(QPointF& pt) const
    {
      if( fabs(pt.x() - clip.left()) < 1e-4 )
	pt.setX(clip.left());
      if( fabs(pt.x() - clip.right()) < 1e-4 )
	pt.setX(clip.right());
      if( fabs(pt.y() - clip.top()) < 1e-4 )
	pt.setY(clip.top());
      if( fabs(pt.y() - clip.bottom()) < 1e-4 )
	pt.setY(clip.bottom());
    }

    // modifies points, returning true if okay to accept
    bool clipLine(QPointF& pt1, QPointF& pt2) const
    {
      // fixup ends to meet clip box if close
      fixPt(pt1);
      fixPt(pt2);

      unsigned code1 = computeCode(pt1);
      unsigned code2 = computeCode(pt2);

      // repeat until points are at edge of box
      // bail out if we repeat too many times (avoid numerical issues)
      for(unsigned count = 0 ; count < 16 ; count++ )
	{
	  if( (code1 | code2) == 0 )
	    {
	      // no more clipping required - inside
	      return true;
	    }
	  else if( (code1 & code2) != 0 )
	    {
	      // line should not be drawn - outside
	      return false;
	    }
	  else
	    {
	      // compute intersection

	      // which point to compute for?
	      const unsigned code = (code1 != 0) ? code1 : code2;

	      // get intersection new point and new code for line
	      QPointF pt;
	      if( code & LEFT )
		pt = vertIntersection(clip.left(), pt1, pt2);
	      else if( code & RIGHT )
		pt = vertIntersection(clip.right(), pt1, pt2);
	      else if( code & TOP )
		pt = horzIntersection(clip.top(), pt1, pt2);
	      else if ( code & BOTTOM )
		pt = horzIntersection(clip.bottom(), pt1, pt2);

	      // update point as intersection
	      if( code == code1 )
		{
		  // changed first point
		  pt1 = pt;
		  code1 = computeCode(pt1);
		}
	      else
		{
		  // changed second point
		  pt2 = pt;
		  code2 = computeCode(pt2);
		}
	    }
	} // loop
      return false;
    }

  private:
    QRectF clip;
  };

  // is difference between points very small?
  inline bool smallDelta(const QPointF& pt1, const QPointF& pt2)
  {
    return fabs(pt1.x() - pt2.x()) < 0.01 &&
      fabs(pt1.y()- pt2.y()) < 0.01;
  }
}

bool clipLine(const QRectF& clip, QPointF& pt1, QPointF& pt2)
{
  Clipper clipper(clip);
  return clipper.clipLine(pt1, pt2);
}

void plotClippedPolyline(QPainter& painter,
			 QRectF clip,
			 const QPolygonF& poly,
			 bool autoexpand)
{
  // exit if fewer than 2 points in polygon
  if ( poly.size() < 2 )
    return;

  // if autoexpand, expand rectangle by line width
  if ( autoexpand )
    {
      const qreal lw = painter.pen().widthF();
      clip.adjust(-lw, -lw, lw, lw);
    }

  // work is done by the clipping object
  Clipper clipper(clip);

  // output goes here
  QPolygonF pout;

  QPolygonF::const_iterator i = poly.begin();
  QPointF lastpt = *i;
  i++;

  for( ; i != poly.end(); ++i )
    {
      QPointF p1 = lastpt;
      QPointF p2 = *i;

      bool plotline = clipper.clipLine(p1, p2);
      if( plotline )
	{
	  if ( pout.isEmpty() )
	    {
	      // add first line
	      pout << p1;
	      if( ! smallDelta(p1, p2) )
		pout << p2;
	    }
	  else
	    {
	      if( p1 == pout.last() )
		{
		  if( ! smallDelta(p1, p2) )
		    // extend polyline
		    pout << p2;
		}
	      else
		{
		  // paint existing line
		  if( pout.size() >= 2 )
		    painter.drawPolyline(pout);

		  // start new line
		  pout.clear();
		  pout << p1;
		  if( ! smallDelta(p1, p2) )
		    pout << p2;
		}
	    }
	}
      else
	{
	  // line isn't in region, so ignore results from clip function

	  // paint existing line
	  if( pout.size() >= 2 )
	    painter.drawPolyline(pout);

	  // cleanup
	  pout.clear();
	}


      lastpt = *i;
    }

  if( pout.size() >= 2 )
    painter.drawPolyline(pout);
}
