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

#include "Python.h"
#include "numpy/arrayobject.h"

#include "qtloops_helpers.h"

void do_numpy_init_package()
{
  import_array();
}

Tuple2Ptrs::Tuple2Ptrs(PyObject* tuple)
{
  const size_t numitems = PyTuple_Size(tuple);

  for(size_t i=0; i != numitems; ++i)
    {
      // access python tuple item
      PyObject* obj = PyTuple_GetItem(tuple, i);

      // convert to C array (stored in objdata)
      PyArrayObject *array = (PyArrayObject*)
	PyArray_ContiguousFromObject(obj, PyArray_DOUBLE, 1, 1);
      if( array == NULL )
	{
	  throw "Cannot covert parameter to 1D numpy array";
	}
      data.push_back( (double*)(array->data) );
      dims.push_back( array->dimensions[0] );
      _arrays.push_back( (PyObject*)array );
    }
}

Tuple2Ptrs::~Tuple2Ptrs()
{
  // delete array objects
  for(int i=0; i < _arrays.size(); ++i)
    {
      Py_DECREF(_arrays[i]);
      _arrays[i] = 0;
      data[i] = 0;
    }
}

Numpy1DObj::Numpy1DObj(PyObject* array)
  : data(0), _array(0)
{
  PyArrayObject *arrayobj = (PyArrayObject*)
    PyArray_ContiguousFromObject(array, PyArray_DOUBLE, 1, 1);
  if( arrayobj == NULL )
    {
      throw "Cannot covert item to 1D numpy array";
    }

  data = (double*)(arrayobj->data);
  dim = arrayobj->dimensions[0];
  _array = (PyObject*)arrayobj;
}

Numpy1DObj::~Numpy1DObj()
{
  Py_XDECREF(_array);
  _array = 0;
  data = 0;
}

Numpy2DObj::Numpy2DObj(PyObject* array)
  : data(0), _array(0)
{
  PyArrayObject *arrayobj = (PyArrayObject*)
    PyArray_ContiguousFromObject(array, PyArray_DOUBLE, 2, 2);

  if( arrayobj == NULL )
    {
      throw "Cannot convert to 2D numpy array";
    }

  data = (double*)(arrayobj->data);
  dims[0] = arrayobj->dimensions[0];
  dims[1] = arrayobj->dimensions[1];
  _array = (PyObject*)arrayobj;
}

Numpy2DObj::~Numpy2DObj()
{
  Py_XDECREF(_array);
  _array = 0;
  data = 0;
}

Numpy2DIntObj::Numpy2DIntObj(PyObject* array)
  : data(0), _array(0)
{
  PyArrayObject *arrayobj = (PyArrayObject*)
    PyArray_ContiguousFromObject(array, PyArray_INT, 2, 2);

  if( arrayobj == NULL )
    {
      throw "Cannot convert to 2D numpy integer array. "
	"Requires numpy.intc argument.";
    }

  data = (int*)(arrayobj->data);
  dims[0] = arrayobj->dimensions[0];
  dims[1] = arrayobj->dimensions[1];
  _array = (PyObject*)arrayobj;
}

Numpy2DIntObj::~Numpy2DIntObj()
{
  Py_XDECREF(_array);
  _array = 0;
  data = 0;
}

PyObject* doubleArrayToNumpy(const double* d, int len)
{
  npy_int dims[1];
  dims[0] = len;
  PyObject* n = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

  double* pydata = (double*) ((PyArrayObject*)(n))->data;
  for(int i = 0; i < len; ++i)
    pydata[i] = d[i];

  return n;
}
