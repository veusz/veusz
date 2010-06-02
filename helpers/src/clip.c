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

#define DEBUG 0

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

static void clipLeftPoint(Point pt, State* state);
static void clipRightPoint(Point pt, State* state);
static void clipTopPoint(Point pt, State* state);
static void clipBottomPoint(Point pt, State* state);
static void clipWritePoint(Point pt, State* state);

/* makes calculations of intercept of line and edge easier */
/* vara = x and varb = y if calculating a x value, and vice versa */
#define INTERCEPT(pt, lastpt, edgeval, vara, varb) \
  (edgeval - pt.vara) * (lastpt.varb - pt.varb) / \
  (lastpt.vara - pt.vara) + pt.varb

static void clipLeftPoint(Point pt, State* state)
{
  #if DEBUG
  printf(" left: %.1f %.1f\n", pt.x, pt.y);
  #endif

  if( state->leftis1st )
    {
      /* do nothing */
      state->left1st = pt;
      state->leftis1st = 0;
    }
  else
    {
      if( pt.x >= state->clipleft )
	{
	  if( state->leftlast.x < state->clipleft )
	    {
	      /* this point inside and last point outside */
	      Point newpt = {
		state->clipleft,
		INTERCEPT(pt, state->leftlast, state->clipleft, x, y)
	      };
	      clipRightPoint(newpt, state);
	    }
	  clipRightPoint(pt, state);
	}
      else
	{
	  if( state->leftlast.x >= state->clipleft )
	    {
	      /* this point outside and last point inside */
	      Point newpt = {
		state->clipleft,
		INTERCEPT(pt, state->leftlast, state->clipleft, x, y)
	      };
	      clipRightPoint(newpt, state);
	    }
	  /* else do nothing if both outside */
	}
    }

  state->leftlast = pt;
  #if DEBUG
  printf(" returning\n");
  #endif
}

static void clipRightPoint(Point pt, State* state)
{
  #if DEBUG
  printf(" right: %.1f %.1f\n", pt.x, pt.y);
  #endif

  if( state->rightis1st )
    {
      /* do nothing */
      state->right1st = pt;
      state->rightis1st = 0;
    }
  else
    {
      if( pt.x <= state->clipright )
	{
	  if( state->rightlast.x > state->clipright )
	    {
	      /* this point inside and last point outside */
	      Point newpt = {
		state->clipright,
		INTERCEPT(pt, state->rightlast, state->clipright, x, y)
	      };
	      clipTopPoint(newpt, state);
	    }
	  clipTopPoint(pt, state);
	}
      else
	{
	  /* this point outside and last point inside */
	  if( state->rightlast.x <= state->clipright )
	    {
	      Point newpt = {
		state->clipright,
		INTERCEPT(pt, state->rightlast, state->clipright, x, y)
	      };
	      clipTopPoint(newpt, state);
	    }
	  /* else do nothing if both outside */
	}
    }

  state->rightlast = pt;
  #if DEBUG
  printf(" returning\n");
  #endif
}

static void clipTopPoint(Point pt, State* state)
{
  #if DEBUG
  printf(" top: %.1f %.1f\n", pt.x, pt.y);
  #endif

  if( state->topis1st )
    {
      /* do nothing */
      state->top1st = pt;
      state->topis1st = 0;
    }
  else
    {
      if( pt.y >= state->cliptop )
	{
	  if( state->toplast.y < state->cliptop )
	    {
	      /* this point inside and last point outside */
	      Point newpt = {
		INTERCEPT(pt, state->toplast, state->cliptop, y, x),
		state->cliptop
	      };
	      clipBottomPoint(newpt, state);
	    }
	  clipBottomPoint(pt, state);
	}
      else
	{
	  /* this point outside */
	  if( state->toplast.y >= state->cliptop )
	    {
	      /* & last point inside */
	      Point newpt = {
		INTERCEPT(pt, state->toplast, state->cliptop, y, x),
		state->cliptop
	      };
	      clipBottomPoint(newpt, state);
	    }
	  /* else do nothing if both outside */
	}
    }

  state->toplast = pt;
  #if DEBUG
  printf(" returning\n");
  #endif
}

static void clipBottomPoint(Point pt, State* state)
{
  #if DEBUG
  printf(" bottom: %.1f %.1f\n", pt.x, pt.y);
  #endif

  if( state->bottomis1st )
    {
      /* do nothing */
      state->bottom1st = pt;
      state->bottomis1st = 0;
    }
  else
    {
      if( pt.y <= state->clipbottom )
	{
	  if( state->bottomlast.y > state->clipbottom )
	    {
	      /* this point inside and last point outside */
	      Point newpt = {
		INTERCEPT(pt, state->bottomlast, state->clipbottom, y, x),
		state->clipbottom
	      };
	      clipWritePoint(newpt, state);
	    }
	  clipWritePoint(pt, state);
	}
      else
	{
	  if( state->bottomlast.y <= state->clipbottom )
	    {
	      /* this point outside and last point inside */
	      Point newpt = {
		INTERCEPT(pt, state->bottomlast, state->clipbottom, y, x),
		state->clipbottom
	      };
	      clipWritePoint(newpt, state);
	    }
	  /* else do nothing if both outside */
	}
    }

  state->bottomlast = pt;
  #if DEBUG
  printf(" returning\n");
  #endif
}

/* add a point to output */
static void clipWritePoint(Point pt, State* state)
{
  #if DEBUG
  printf("Writing: %.1f %.1f\n", pt.x, pt.y);
  #endif

  state->output[state->outindex*2] = pt.x;
  state->output[state->outindex*2 + 1] = pt.y;
  state->outindex++;
}

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
      clipLeftPoint(pt, &state);
    }
  clipLeftPoint(state.left1st, &state);
  clipRightPoint(state.right1st, &state);
  clipTopPoint(state.top1st, &state);
  clipBottomPoint(state.bottom1st, &state);

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
