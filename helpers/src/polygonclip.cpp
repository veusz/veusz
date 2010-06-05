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

// This code uses the Sutherland Hodgman algorithm to clip a polygon
// It is inspired by another version by Sjaak Priester
// see http://www.codeguru.com/Cpp/misc/misc/graphics/article.php/c8965/

#include "polygonclip.h"

// macro to clip point against edge
//   - edge: name of edge for clipping
//   - isinside: test whether point is inside edge
//   - intercept: function to calculate coordinate where line
//     crosses edge
//   - next: function to call next to clip point
#define CLIPEDGE(edge, isinside, intercept, next)		\
  void edge##ClipPoint(const QPointF& pt)			\
  {								\
    QPointF& lastpt = edge##last;				\
    if( edge##is1st )						\
      {								\
	/* do nothing */					\
	edge##1st = pt;						\
	edge##is1st = false;					\
      }								\
    else							\
      {								\
	if( isinside(pt) )					\
	  {							\
	    if( ! isinside(lastpt) )				\
	      /* this point inside and last point outside */	\
	      next( intercept(clip.edge(), pt, lastpt) );	\
	    next(pt);						\
	  }							\
	else							\
	  {							\
	    if( isinside(lastpt) )				\
	      /* this point outside and last point inside */	\
	      next( intercept(clip.edge(), pt, lastpt) );	\
	    /* else do nothing if both outside */		\
	  }							\
      }								\
								\
    lastpt = pt;						\
  }

namespace
{
  inline QPointF interceptVert(qreal horzval,
			       const QPointF& pt1, const QPointF& pt2)
  {
    const qreal gradient = (pt2.y()-pt1.y()) / (pt2.x()-pt1.x());
    return QPointF(horzval, (horzval - pt1.x())*gradient + pt1.y());
  }
  inline QPointF interceptHorz(qreal vertval,
			       const QPointF& pt1, const QPointF& pt2)
  {
    const qreal gradient = (pt2.x()-pt1.x()) / (pt2.y()-pt1.y());
    return QPointF((vertval - pt1.y())*gradient + pt1.x(), vertval);
  }

  struct State
  {
    State(const QRectF& rect, QPolygonF& out)
      : clip(rect), output(out),
	leftis1st(true), rightis1st(true), topis1st(true), bottomis1st(true)
    {
    }

    // tests for whether point is inside of outside of each side
    inline bool insideBottom(const QPointF& pt) const
    {
      return pt.y() <= clip.bottom();
    }
    inline bool insideTop(const QPointF& pt) const
    {
      return pt.y() >= clip.top();
    }
    inline bool insideRight(const QPointF& pt) const
    {
      return pt.x() <= clip.right();
    }
    inline bool insideLeft(const QPointF& pt) const
    {
      return pt.x() >= clip.left();
    }

    // add functions for clipping to each edge
    CLIPEDGE(bottom, insideBottom, interceptHorz, writeClipPoint)
    CLIPEDGE(top, insideTop, interceptHorz, bottomClipPoint)
    CLIPEDGE(right, insideRight, interceptVert, topClipPoint)
    CLIPEDGE(left, insideLeft, interceptVert, rightClipPoint)

    // finally writes to output
    void writeClipPoint(const QPointF& pt)
    {
      output << pt;
    }

    /* location of corners of clip rectangle */
    QRectF clip;
 
    /* output points are added here */
    QPolygonF &output;

    /* last points added */
    QPointF leftlast, rightlast, toplast, bottomlast;

    /* first point for each stage */
    QPointF left1st, right1st, top1st, bottom1st;

    /* whether this is the 1st point through */
    bool leftis1st, rightis1st, topis1st, bottomis1st;
  };
}

void polygonClip(const QPolygonF& inpoly,
		 const QRectF& cliprect,
		 QPolygonF& out)
{
  // construct initial state
  State state(cliprect, out);

  // do the clipping
  for(QPolygonF::const_iterator pt = inpoly.begin(); pt != inpoly.end(); ++pt)
    {
      state.leftClipPoint(*pt);
    }

  // complete
  state.leftClipPoint(state.left1st);
  state.rightClipPoint(state.right1st);
  state.topClipPoint(state.top1st);
  state.bottomClipPoint(state.bottom1st);
}

// #include <stdio.h>
// int main()
// {
//   QPolygonF in;
//   in << QPointF(100, 100) << QPointF(200, 100)
//      << QPointF(200, 200) << QPointF(100, 200);
//   QPolygonF out;
 
//   polygonClip(in, QRectF(QPointF(0,0),QPointF(1000,1000)), out);

//   for(QPolygonF::const_iterator i = out.begin(); i != out.end(); ++i)
//     printf("%g %g\n", i->x(), i->y());


//   return 0;
// }
