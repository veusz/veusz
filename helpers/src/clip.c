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

#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdlib.h>

#define MIN(a,b)  ((a) < (b) ? (a) : (b))
#define MAX(a,b)  ((a) > (b) ? (a) : (b))

typedef double FloatData;

typedef struct
{
  FloatData x, y;
} Point;

typedef struct
{
  /* location of corners of clip rectangle */
  FloatData clipleft, clipright, cliptop, clipbottom;

  /* last points added */
  Point leftlast, rightlast, toplast, bottomlast;

  /* first point for each stage */
  Point left1st, right1st, top1st, bottom1st;

  /* whether this is the 1st point through */
  int leftis1st, rightis1st, topis1st, bottomis1st;

  /* output points are added here */
  FloatData* output;
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

static void doClipping(const FloatData* xdata,
		       const FloatData* ydata,
		       const int numpts,
		       const FloatData x1, const FloatData y1,
		       const FloatData x2, const FloatData y2,
		       FloatData *output, int* numoutput)
{
  int i;

  /* construct initial state */
  State state;
  state.clipleft = MIN(x1, x2);
  state.clipright = MAX(x1, x2);
  state.cliptop = MIN(y1, y2);
  state.clipbottom = MAX(y1, y2);
  state.leftis1st = state.rightis1st = state.topis1st = state.bottomis1st = 1;
  state.output = output;
  state.outindex = 0;

  /* do the clipping */
  for(i = 0; i < numpts; ++i)
    {
      Point pt = {xdata[i], ydata[i]};
      leftClipPoint(pt, &state);
    }
  leftClipPoint(state.left1st, &state);
  rightClipPoint(state.right1st, &state);
  topClipPoint(state.top1st, &state);
  bottomClipPoint(state.bottom1st, &state);

  /* return number of points */
  *numoutput = state.outindex;
}

static PyObject *
python_clip(PyObject *self, PyObject *args)
{
  double x1, y1, x2, y2;
  npy_intp dimsx, dimsy, retdims[2];

  int numitems;
  PyObject *xarray, *yarray;
  PyArrayObject *retarray;
  PyArray_Dims dims;

  double *xdata, *ydata, *retdata;

  if (!PyArg_ParseTuple(args, "ddddOO", &x1, &y1, &x2, &y2,
			&xarray, &yarray))
    return NULL;

  if( PyArray_AsCArray(&xarray, &xdata, &dimsx, 1,
		       PyArray_DescrFromType(NPY_DOUBLE)) )
    {
      PyErr_SetString(PyExc_TypeError, "Cannot convert X data to C array");
      return NULL;
    }

  if( PyArray_AsCArray(&yarray, &ydata, &dimsy, 1,
		       PyArray_DescrFromType(NPY_DOUBLE)) )
    {
      PyArray_Free(xarray, xdata);
      PyErr_SetString(PyExc_TypeError, "Cannot convert Y data to C array");
      return NULL;
    }
  
  retdims[0] = MIN(dimsx, dimsy) * 2;
  retdims[1] = 2;

  retarray = (PyArrayObject*) PyArray_SimpleNew(2, retdims, NPY_DOUBLE);
  retdata = (double*)(retarray->data);
   
  doClipping(xdata, ydata, MIN(dimsx, dimsy),
	     x1, y1, x2, y2, retdata, &numitems);

  retdims[0] = numitems;
  dims.ptr = retdims;
  dims.len = 2;
  PyArray_Resize(retarray, &dims, 1, NPY_CORDER);

  PyArray_Free(xarray, xdata);
  PyArray_Free(yarray, ydata);

  return PyArray_Return(retarray);
}

static PyMethodDef ClipMethods[] =
  {
    {"clippolygon",  python_clip, METH_VARARGS,
     "Clip a polygon to a box."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
  };

PyMODINIT_FUNC
initclip(void)
{
  PyObject *m;
  
  m = Py_InitModule("clip", ClipMethods);
  if (m == NULL)
    return;

  import_array();
}
