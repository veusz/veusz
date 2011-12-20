// -*- mode: C++; -*-

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

#include <QVector>

#define DEBUG false

void do_numpy_init_package();

// class for converting tuples to objects which clean themselves up
// throws const char* if conversion failed
class Tuple2Ptrs
{
public:
  Tuple2Ptrs(PyObject* tuple);
  ~Tuple2Ptrs();

  // data in tuple are stored here
  QVector<const double*> data;
  QVector<int> dims;

private:
  // these are the python objects made by PyArray_AsCArray
  QVector<PyObject*> _arrays;
};

// class for converting numpy array to an array
class Numpy1DObj
{
 public:
  Numpy1DObj(PyObject* array);
  ~Numpy1DObj();

  const double* data;
  int dim;

  inline double operator()(const int x) const
  {
    if( DEBUG && (x < 0 || x >= dim) )
	throw "Invalid index in array";
    return data[x];
  }

 private:
  PyObject* _array;
};

// class for converting a 2D numpy array to an array
class Numpy2DObj
{
 public:
  Numpy2DObj(PyObject* array);
  ~Numpy2DObj();

  const double* data;
  int dims[2];

  inline double operator()(const int x, const int y) const
  {
    if( DEBUG && (x < 0 || x >= dims[0] || y < 0 || y >= dims[1]) )
      throw "Invalid index in array";
    return data[x+y*dims[1]];
  }

 private:
  PyObject* _array;
};

// class for converting a 2D numpy array to an integer array
class Numpy2DIntObj
{
 public:
  Numpy2DIntObj(PyObject* array);
  ~Numpy2DIntObj();

  const int* data;
  int dims[2];

  inline int operator()(const int x, const int y) const
  {
    if( DEBUG && (x < 0 || x >= dims[0] || y < 0 || y >= dims[1]) )
      throw "Invalid index in array";
    return data[x+y*dims[1]];
  }

 private:
  PyObject* _array;
};

PyObject* doubleArrayToNumpy(const double* d, int len);

#endif
