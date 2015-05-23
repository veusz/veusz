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

#ifndef SHAPES_H
#define SHAPES_H

#include <QtGui/QPainterPath>
#include <algorithm>
#include <vector>

#include "mmaths.h"
#include "fragment.h"
#include "properties.h"

class Object
{
 public:
  virtual ~Object();

  virtual void getFragments(const Mat4& outerM, FragmentVector& v);
};

class Triangle : public Object
{
 public:
  Triangle()
    : Object(), surfaceprop(0)
  {
  }
  Triangle(const Vec4& a, const Vec4& b, const Vec4& c,
	   const SurfaceProp* prop=0)
    : surfaceprop(prop)
  {
    points[0] = a; points[1] = b; points[2] = c;
  }

  void getFragments(const Mat4& outerM, FragmentVector& v);

 public:
  Vec4 points[3];
  PropSmartPtr<const SurfaceProp> surfaceprop;
};

class PolyLine : public Object
{
 public:
  PolyLine(const LineProp* prop=0)
    : Object(), lineprop(prop)
  {
  }

  void addPoint(const Vec4& v)
  {
    points.push_back(v);
  }

  void addPoints(const ValVector& x, const ValVector& y, const ValVector& z);

  void getFragments(const Mat4& outerM, FragmentVector& v);

 public:
  Vec4Vector points;
  PropSmartPtr<const LineProp> lineprop;
};

// a grid of height values on a regular mesh with grid points given
// heights has M*N elements where M and N are the length of pos1 & pos2
class Mesh : public Object
{
public:
  // X_DIRN: heights is X, a is Y, b is Z
  // Y_DIRN: heights is Y, a is Z. b is X
  // Z_DIRN: heights is Z, a is X, b is Y
  enum Direction {X_DIRN, Y_DIRN, Z_DIRN};

public:
  Mesh(const ValVector& _pos1, const ValVector& _pos2,
       const ValVector& _heights,
       Direction _dirn,
       const LineProp* lprop=0, const SurfaceProp* sprop=0)
    : pos1(_pos1), pos2(_pos2), heights(_heights),
      dirn(_dirn), lineprop(lprop), surfaceprop(sprop)
  {
  }

  void getFragments(const Mat4& outerM, FragmentVector& v);

private:
  void getSurfaceFragments(const Mat4& outerM, FragmentVector& v);
  void getLineFragments(const Mat4& outerM, FragmentVector& v);

  void getVecIdxs(unsigned &vidx_h, unsigned &vidx_1, unsigned &vidx_2) const;

public:
  ValVector pos1, pos2, heights;
  Direction dirn;
  PropSmartPtr<const LineProp> lineprop;
  PropSmartPtr<const SurfaceProp> surfaceprop;
};

// a set of points to plot
class Points : public Object
{
public:
  Points(const ValVector& px, const ValVector& py, const ValVector& pz,
         QPainterPath pp,
         const LineProp* pointedge=0, const SurfaceProp* pointfill=0)
    : x(px), y(py), z(pz),
      path(pp),
      scaleedges(1),
      lineedge(pointedge), surfacefill(pointfill)
  {
  }

  void setSizes(const ValVector& _sizes) { sizes = _sizes; }

  void getFragments(const Mat4& outerM, FragmentVector& v);

private:
  FragmentPathParameters fragparams;

public:
  ValVector x, y, z;
  ValVector sizes;
  QPainterPath path;
  bool scaleedges;
  PropSmartPtr<const LineProp> lineedge;
  PropSmartPtr<const SurfaceProp> surfacefill;
};


// container of objects with transformation matrix of children
// Note: object pointers passed to object will be deleted when this
// container is deleted
class ObjectContainer : public Object
{
 public:
  ObjectContainer()
    : objM(identityM4())
    {}

  ~ObjectContainer();
  void getFragments(const Mat4& outerM, FragmentVector& v);

  void addObject(Object* obj)
  {
    objects.push_back(obj);
  }

 public:
  Mat4 objM;
  std::vector<Object*> objects;
};

#endif
