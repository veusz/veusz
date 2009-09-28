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
      npy_intp dims = -1;
      double* objdata = 0;

      // convert to C array (stored in objdata)
      if( PyArray_AsCArray(&obj, &objdata, &dims, 1,
			   PyArray_DescrFromType(NPY_DOUBLE)) )
	{
	  throw "Cannot convert items to floating point data";
	}
      
      // store corresponding valarray
      data.push_back(new doublearray(objdata, dims));
      _convitems.push_back(obj);
      _convdata.push_back(objdata);
    }
}

TupleInValarray::~TupleInValarray()
{
  // delete constructed valarrays
  for(size_t i=0; i != data.size(); ++i)
    {
      delete data[i];
    }

  // delete any space used in PyArray_AsCArray
  for(size_t i=0; i != _convitems.size(); ++i)
    {
      PyArray_Free(_convitems[i], _convdata[i]);
    }
}
