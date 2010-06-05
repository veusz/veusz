/*
    Copyright (C) 2010 Jeremy Sanders

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
 This code uses the Sutherland Hodgman algorithm to clip a polygon
 It is based on algorithm of a C++ version by Sjaak Priester
 see http://www.codeguru.com/Cpp/misc/misc/graphics/article.php/c8965/
*/

#include "polygonclip.h"

/* makes calculations of intercept of line and edge easier */
/* vara = x and varb = y if calculating a x value, and vice versa */
#define INTERCEPT(pt, lastpt, edgeval, vara, varb)	\
  (edgeval - pt.vara()) * (lastpt.varb() - pt.varb()) /	\
  (lastpt.vara() - pt.vara()) + pt.varb()

/* macro to clip point against edge
   - edge: name of edge for clipping
   - isinside: f(pt) to return whether point is inside edge
   - interceptx: value of x for new point when line intercepts edge
   - intercepty: value of y for new point when line intercepts edge
   - next: function to call next to clip point
*/
#define CLIPEDGE(edge, isinside, interceptx, intercepty, next)	\
  void edge##ClipPoint(const QPointF& pt)			\
  {								\
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
	    if( ! isinside(edge##last) )			\
	      {							\
		/* this point inside and last point outside */	\
		next(QPointF(interceptx, intercepty));		\
	      }							\
	    next(pt);						\
	  }							\
	else							\
	  {							\
	    if( isinside(edge##last) )				\
	      {							\
		/* this point outside and last point inside */	\
		next(QPointF(interceptx, intercepty));		\
	      }							\
	    /* else do nothing if both outside */		\
	  }							\
      }								\
								\
    edge##last = pt;						\
  }

// tests to see whether point is inside particular side
#define INSIDEBOTTOM(pt) (pt.y() <= clip.bottom())
#define INSIDETOP(pt) (pt.y() >= clip.top())
#define INSIDERIGHT(pt) (pt.x() <= clip.right())
#define INSIDELEFT(pt) (pt.x() >= clip.left())

namespace
{
  struct State
  {
    State(const QRectF& rect, QPolygonF& out)
      : clip(rect), output(out),
	leftis1st(true), rightis1st(true), topis1st(true), bottomis1st(true)
    {
    }

    // add functions for clipping to each edge
    CLIPEDGE(bottom, INSIDEBOTTOM,
	     INTERCEPT(pt, bottomlast, clip.bottom(), y, x),
	     clip.bottom(),
	     writeClipPoint)
    CLIPEDGE(top, INSIDETOP,
	     INTERCEPT(pt, toplast, clip.top(), y, x),
	     clip.top(),
	     bottomClipPoint)
    CLIPEDGE(right, INSIDERIGHT,
	     clip.right(),
	     INTERCEPT(pt, rightlast, clip.right(), x, y),
	     topClipPoint)
    CLIPEDGE(left, INSIDELEFT,
	     clip.left(),
	     INTERCEPT(pt, leftlast, clip.left(), x, y),
	     rightClipPoint)

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
 
//   polygonClip(in, QRectF(QPointF(0,0),QPointF(150,150)), out);

//   for(QPolygonF::const_iterator i = out.begin(); i != out.end(); ++i)
//     printf("%g %g\n", i->x(), i->y());


//   return 0;
// }
