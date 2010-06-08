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

TupleInValarray::TupleInValarray(PyObject* tuple)
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
	  throw "Cannot covert item";
	}
      data.push_back( new doublearray( (double*)(array->data),
				       array->dimensions[0]) );
      _convitems.push_back( (PyObject*)array);
    }
}

TupleInValarray::~TupleInValarray()
{
  // delete constructed valarrays
  for(size_t i=0; i != data.size(); ++i)
    {
      delete data[i];
    }

  // delete array objects
  for(size_t i=0; i != _convitems.size(); ++i)
    {
      Py_DECREF(_convitems[i]);
    }
}

NumpyInValarray::NumpyInValarray(PyObject* array)
  : data(0), _convitem(0)
{
  PyArrayObject *arrayobj = (PyArrayObject*)
    PyArray_ContiguousFromObject(array, PyArray_DOUBLE, 1, 1);
  if( arrayobj == NULL )
    {
      throw "Cannot covert item";
    }

  data = new doublearray((double*)(arrayobj->data), arrayobj->dimensions[0]);

  _convitem = (PyObject*)array;
}

NumpyInValarray::~NumpyInValarray()
{
  delete data;
  if( _convitem )
    {
      Py_DECREF(_convitem);
    }
}


NumpyIn2DValarray::NumpyIn2DValarray(PyObject* array)
  : data(0), _convitem(0)
{
  PyArrayObject *arrayobj = (PyArrayObject*)
    PyArray_ContiguousFromObject(array, PyArray_DOUBLE, 2, 2);

  if( arrayobj == NULL )
    {
      throw "Cannot convert to floating point data";
    }

  dims[0] = arrayobj->dimensions[0];
  dims[1] = arrayobj->dimensions[1];

  data = new doublearray((double*)(arrayobj->data), dims[0]*dims[1]);
  _convitem = (PyObject*)arrayobj;
}

NumpyIn2DValarray::~NumpyIn2DValarray()
{
  delete data;
  if( _convitem )
    {
      Py_DECREF(_convitem);
    }
}


NumpyIn2DIntValarray::NumpyIn2DIntValarray(PyObject* array)
  : data(0), _convitem(0)
{
  PyArrayObject *arrayobj = (PyArrayObject*)
    PyArray_ContiguousFromObject(array, PyArray_INT, 2, 2);

  if( arrayobj == NULL )
    {
      throw "Cannot convert to to an integer array";
    }

  dims[0] = arrayobj->dimensions[0];
  dims[1] = arrayobj->dimensions[1];

  data = new intarray((int*)(arrayobj->data), dims[0]*dims[1]);
  _convitem = (PyObject*)arrayobj;
}

NumpyIn2DIntValarray::~NumpyIn2DIntValarray()
{
  delete data;
  if( _convitem )
    {
      Py_DECREF(_convitem);
    }
}

