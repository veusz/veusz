// -*-c++-*-

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

#ifndef FIXEDVECTOR_H
#define FIXEDVECTOR_H

// fixed sized vector with no dynamic allocations
// warning: no checks for invalid indices!
// class T needs a default constructor
template<typename T, unsigned short N> class FixedVector
{
 public:
  typedef T* iterator;
  typedef  const T* const_iterator;

  FixedVector()
    : _size(0)
    {}

  void push_back(const T& v)
  {
    _data[_size++] = v;
  }
  unsigned short size() const { return _size; }
  bool empty() const { return _size==0; }
  unsigned short max_size() const { return N; }

  const T& operator[](unsigned short idx) const { return _data[idx]; }
  T& operator[](unsigned short idx) { return _data[idx]; }

  iterator begin() { return &_data[0]; }
  const_iterator begin() const { return &_data[0]; }
  const_iterator cbegin() const { return &_data[0]; }
  iterator end() { return &_data[_size]; }
  const_iterator end() const { return &_data[_size]; }
  const_iterator cend() const { return &_data[_size]; }

  T& front() { return _data[0]; }
  const T& front() const { return _data[0]; }
  T& back() { return _data[_size-1]; }
  const T& back() const { return _data[_size-1]; }

 private:
  unsigned short _size;
  T _data[N];
};

#endif
