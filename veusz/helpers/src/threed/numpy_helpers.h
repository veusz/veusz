// -*-c++-*-

#ifndef NUMPY_HELPERS_H
#define NUMPY_HELPERS_H

#include "Python.h"
#include "mmaths.h"

void doNumpyInitPackage();
ValVector numpyToValVector(PyObject* obj);

#endif
