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
    PyArray_ContiguousFromObject(obj, NPY_DOUBLE, 1, 1);
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
