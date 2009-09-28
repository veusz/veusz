#ifndef QTLOOPS_HELPERS_H
#define QTLOOPS_HELPERS_H

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

#include <valarray>
#include <vector>

typedef std::valarray<double> doublearray;
typedef std::vector<const doublearray*> doublearray_ptr_vec;

void do_numpy_init_package();

// class for converting tuples of numpy arrays to a vector of valarrays
// throws const char* if conversion failed
class TupleInValarray
{
public:
  TupleInValarray(PyObject* tuple);
  ~TupleInValarray();

  // data in tuple are stored here
  doublearray_ptr_vec data;

private:
  // these are the python objects made by PyArray_AsCArray
  std::vector<PyObject*> _convitems;
  // corresponding pointers to data
  std::vector<double*> _convdata;
};

// class for converting numpy array to a valarray
class NumpyInValarray
{
 public:
  NumpyInValarray(PyObject* array);
  ~NumpyInValarray();

  doublearray* data;

 private:
  PyObject* _convitem;
  double* _convdata;
};

#endif
