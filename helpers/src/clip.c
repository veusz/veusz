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

#include <stdlib.h>
#include <stdio.h>

typedef struct
{
  float x, y;
} Point;

typedef struct
{
  /* location of corners of clip rectangle */
  float clipleft, clipright, cliptop, clipbottom;

  /* last points added */
  Point leftlast, rightlast, toplast, bottomlast;

  /* first point for each stage */
  Point left1st, right1st, top1st, bottom1st;

  /* whether this is the 1st point through */
  int leftis1st, rightis1st, topis1st, bottomis1st;

  /* output points are added here */
  float* output;
  int outindex;
} State;

/* makes calculations of intercept of line and edge easier */
/* vara = x and varb = y if calculating a x value, and vice versa */
#define INTERCEPT(pt, lastpt, edgeval, vara, varb) \
  (edgeval - pt.vara) * (lastpt.varb - pt.varb) / \
  (lastpt.vara - pt.vara) + pt.varb

/* macro to clip point against edge
   - edge: name of edge for clipping
   - isinside: f(pt) to return whether point is inside edge
   - interceptx: value of x for new point when line intercepts edge
   - intercepty: value of y for new point when line intercepts edge
   - next: function to call next to clip point
*/
#define CLIPEDGE(edge, isinside, interceptx, intercepty, next)		\
  static void edge##ClipPoint(Point pt, State* state)			\
  {									\
    if( state->edge##is1st )						\
      {									\
	/* do nothing */						\
	state->edge##1st = pt;						\
	state->edge##is1st = 0;						\
      }									\
    else								\
      {									\
	if( isinside(pt) )						\
	  {								\
	    if( ! isinside(state->edge##last) )				\
	      {								\
		/* this point inside and last point outside */		\
		Point newpt = {interceptx, intercepty};			\
		next(newpt, state);					\
	      }								\
	    next(pt, state);						\
	  }								\
	else								\
	  {								\
	    if( isinside(state->edge##last) )				\
	      {								\
		/* this point outside and last point inside */		\
		Point newpt = {interceptx, intercepty};			\
		next(newpt, state);					\
	      }								\
	    /* else do nothing if both outside */			\
	  }								\
      }									\
    									\
    state->edge##last = pt;						\
  }

/* add a point to output */
static void writeClipPoint(Point pt, State* state)
{
  state->output[state->outindex*2] = pt.x;
  state->output[state->outindex*2 + 1] = pt.y;
  state->outindex++;
}

#define INSIDEBOTTOM(pt) (pt.y <= state->clipbottom)
CLIPEDGE(bottom, INSIDEBOTTOM,
	 INTERCEPT(pt, state->bottomlast, state->clipbottom, y, x),
	 state->clipbottom,
	 writeClipPoint)

#define INSIDETOP(pt) (pt.y >= state->cliptop)
CLIPEDGE(top, INSIDETOP,
	 INTERCEPT(pt, state->toplast, state->cliptop, y, x),
	 state->cliptop,
	 bottomClipPoint)

#define INSIDERIGHT(pt) (pt.x <= state->clipright)
CLIPEDGE(right, INSIDERIGHT,
	 state->clipright,
	 INTERCEPT(pt, state->rightlast, state->clipright, x, y),
	 topClipPoint)

#define INSIDELEFT(pt) (pt.x >= state->clipleft)
CLIPEDGE(left, INSIDELEFT,
	 state->clipleft,
	 INTERCEPT(pt, state->leftlast, state->clipleft, x, y),
	 rightClipPoint)

static void doClipping(const float* pts, const int numpts,
		       const float x1, const float y1,
		       const float x2, const float y2,
		       float *output, int* numoutput)
{
  int i;

  /* construct initial state */
  State state;
  state.clipleft = x1; state.clipright = x2;
  state.cliptop = y1; state.clipbottom = y2;
  state.leftis1st = state.rightis1st = state.topis1st = state.bottomis1st = 1;
  state.output = output;
  state.outindex = 0;

  /* do the clipping */
  for(i = 0; i < numpts; ++i)
    {
      Point pt = {pts[i*2], pts[i*2+1]};
      leftClipPoint(pt, &state);
    }
  leftClipPoint(state.left1st, &state);
  rightClipPoint(state.right1st, &state);
  topClipPoint(state.top1st, &state);
  bottomClipPoint(state.bottom1st, &state);

  /* return number of points */
  *numoutput = state.outindex;
}

int main()
{
  int i;

  float pts[8] = { 250, 0, 0, 250, 250, 500, 500, 250 };

  float out[100];
  int numout;

  doClipping(pts, 4, 100, 100, 400, 400, out, &numout);
  printf("%i\n", numout);

  for(i=0; i<numout; ++i)
    {
      printf("%g %g\n", out[i*2], out[i*2+1]);
    }

  return 0;
}
