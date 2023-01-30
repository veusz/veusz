//    Copyright (C) 2015 Jeremy S. Sanders
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

#include "numpy_helpers.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

namespace
{
  // python3 numpy import_array is a macro with a return (how stupid),
  // so we have to wrap it up to get it to portably compile
#if PY_MAJOR_VERSION >= 3
  void* doImport()
  {
    import_array();
    return 0;
  }
#else
  void doImport()
  {
    import_array();
  }
#endif
}

void doNumpyInitPackage()
{
  doImport();
}

ValVector numpyToValVector(PyObject* obj)
{
  PyArrayObject *arrayobj = (PyArrayObject*)
    PyArray_ContiguousFromAny(obj, NPY_DOUBLE, 1, 1);
  if(arrayobj == NULL)
    {
      throw "Cannot covert item to 1D numpy array";
    }

  const double* data = (double*)PyArray_DATA(arrayobj);
  unsigned dim = PyArray_DIMS(arrayobj)[0];

  ValVector out;
  out.reserve(dim);
  for(unsigned i=0; i<dim; ++i)
    out.push_back(data[i]);

  Py_DECREF((PyObject*)arrayobj);

  return out;
}
