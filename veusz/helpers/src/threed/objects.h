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
#include <QtGui/QPainter>
#include <algorithm>
#include <vector>

#include "mmaths.h"
#include "fragment.h"
#include "properties.h"

class Object
{
 public:
  Object() : widgetid(0) {}

  virtual ~Object();

  virtual void getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);

  // recursive set id of child objects
  virtual void assignWidgetId(unsigned long id);

  // id of widget which generated object
  unsigned long widgetid;
};

class Triangle : public Object
{
 public:
  Triangle()
    : Object(), surfaceprop(0)
  {
  }
  Triangle(const Vec3& a, const Vec3& b, const Vec3& c,
	   const SurfaceProp* prop=0)
    : surfaceprop(prop)
  {
    points[0] = a; points[1] = b; points[2] = c;
  }

  void getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);

 public:
  Vec3 points[3];
  PropSmartPtr<const SurfaceProp> surfaceprop;
};

class PolyLine : public Object
{
public:
  PolyLine(const LineProp* prop=0)
    : Object(), lineprop(prop)
  {
  }

  PolyLine(const ValVector& x, const ValVector& y, const ValVector& z,
           const LineProp* prop=0)
    : Object(), lineprop(prop)
  {
    addPoints(x, y, z);
  }

  void addPoint(const Vec3& v)
  {
    points.push_back(v);
  }

  void addPoints(const ValVector& x, const ValVector& y, const ValVector& z);

  void getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);

public:
  Vec3Vector points;
  PropSmartPtr<const LineProp> lineprop;
};

class LineSegments : public Object
{
public:
  LineSegments(const ValVector& x1, const ValVector& y1, const ValVector& z1,
               const ValVector& x2, const ValVector& y2, const ValVector& z2,
               const LineProp* prop);
  LineSegments(const ValVector& pts1, const ValVector& pts2,
               const LineProp* prop);

  void getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);

public:
  Vec3Vector points;
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
       const LineProp* lprop=0, const SurfaceProp* sprop=0,
       bool _hidehorzline=0, bool _hidevertline=0)
    : pos1(_pos1), pos2(_pos2), heights(_heights),
      dirn(_dirn), lineprop(lprop), surfaceprop(sprop),
      hidehorzline(_hidehorzline), hidevertline(_hidevertline)
  {
  }

  void getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);

private:
  void getSurfaceFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);
  void getLineFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);

  void getVecIdxs(unsigned &vidx_h, unsigned &vidx_1, unsigned &vidx_2) const;

public:
  ValVector pos1, pos2, heights;
  Direction dirn;
  PropSmartPtr<const LineProp> lineprop;
  PropSmartPtr<const SurfaceProp> surfaceprop;
  bool hidehorzline, hidevertline;
};

// Grid of data values, where the centres of the bins are specified.
// There should be 1 more values along edges than values in array.
// idxval, edge1, edge2 give the index of the axis (x=0,y=1,z=2) for
// that direction.
class DataMesh : public Object
{
public:
  DataMesh(const ValVector& _edges1, const ValVector& _edges2,
           const ValVector& _vals,
           unsigned _idxval, unsigned _idxedge1, unsigned _idxedge2,
           bool _highres,
           const LineProp* lprop=0, const SurfaceProp* sprop=0,
           bool _hidehorzline=0, bool _hidevertline=0)
    : edges1(_edges1), edges2(_edges2), vals(_vals),
      idxval(_idxval), idxedge1(_idxedge1), idxedge2(_idxedge2),
      highres(_highres),
      lineprop(lprop), surfaceprop(sprop),
      hidehorzline(_hidehorzline), hidevertline(_hidevertline)
  {
  }

  void getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);

public:
  ValVector edges1, edges2, vals;
  unsigned idxval, idxedge1, idxedge2;
  bool highres;

  PropSmartPtr<const LineProp> lineprop;
  PropSmartPtr<const SurfaceProp> surfaceprop;
  bool hidehorzline, hidevertline;
};

// multiple cuboids
class MultiCuboid : public Object
{
public:
  MultiCuboid(const ValVector& _xmin, const ValVector& _xmax,
              const ValVector& _ymin, const ValVector& _ymax,
              const ValVector& _zmin, const ValVector& _zmax,
              const LineProp* lprop=0, const SurfaceProp* sprop=0)
    : xmin(_xmin), xmax(_xmax),
      ymin(_ymin), ymax(_ymax),
      zmin(_zmin), zmax(_zmax),
      lineprop(lprop), surfaceprop(sprop)
  {
  }

  void getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);

public:
  ValVector xmin, xmax, ymin, ymax, zmin, zmax;

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
      scaleline(true),
      scalepersp(true),
      lineedge(pointedge), surfacefill(pointfill)
  {
  }

  void setSizes(const ValVector& _sizes) { sizes = _sizes; }

  void getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);

private:
  FragmentPathParameters fragparams;

public:
  ValVector x, y, z;
  ValVector sizes;
  QPainterPath path;
  bool scaleline, scalepersp;
  PropSmartPtr<const LineProp> lineedge;
  PropSmartPtr<const SurfaceProp> surfacefill;
};

// a "text" class which calls back draw() when drawing is requested
class Text : public Object
{
public:
  // pos1 and pos2 contain a list of x,y,z values
  Text(const ValVector& _pos1, const ValVector& _pos2);

  void getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);

  virtual void draw(QPainter* painter,
                    QPointF pt1, QPointF pt2, QPointF pt3,
                    unsigned index, double scale, double linescale);

private:
  class TextPathParameters : public FragmentPathParameters
  {
  public:
    void callback(QPainter* painter, QPointF pt1, QPointF pt2, QPointF pt3,
                  int index,  double scale, double linescale);
    Text* text;
  };

  TextPathParameters fragparams;

public:
  ValVector pos1, pos2;
};

// A triangle only visible if its norm (translated to viewing space) is +ve
class TriangleFacing : public Triangle
{
public:
  TriangleFacing(const Vec3& a, const Vec3& b, const Vec3& c,
	   const SurfaceProp* prop=0)
    : Triangle(a, b, c, prop)
  {}

  void getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);
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
  void getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);

  void addObject(Object* obj)
  {
    objects.push_back(obj);
  }

  // recursive set id of child objects
  void assignWidgetId(unsigned long id);

 public:
  Mat4 objM;
  std::vector<Object*> objects;
};

// container which only draws contents if the norm is pointing in +ve
// z direction

class FacingContainer : public ObjectContainer
{
public:
  FacingContainer(Vec3 _norm)
    : ObjectContainer(), norm(_norm)
  {
  }
  void getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);

public:
  Vec3 norm;
};


// This class draws tick labels with correct choice of axis

class AxisLabels : public Object
{
public:
  // cube defined to be between these corners
  AxisLabels(const Vec3& _box1, const Vec3& _box2,
             const ValVector& _tickfracs,
             double _labelfrac);

  void addAxisChoice(const Vec3& start, const Vec3& end);

  // override this: draw reqested label at origin, with alignment
  // given
  // (if index==-1, then draw axis label)
  virtual void drawLabel(QPainter* painter, int index,
                         QPointF pt,
                         QPointF ax1, QPointF ax2,
                         double axangle);

  void getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);

private:
  Vec3 box1, box2;
  ValVector tickfracs;
  double labelfrac;
  std::vector<Vec3> starts, ends;

private:
  struct PathParameters : public FragmentPathParameters
  {
    void callback(QPainter* painter, QPointF pt, QPointF ax1, QPointF ax2,
                  int index, double scale, double linescale);
    AxisLabels* tl;
    double axangle;
  };

  PathParameters fragparams;
};

#endif
