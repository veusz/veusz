// -*-c++-*-

#ifndef SHAPES_H
#define SHAPES_H

#include <algorithm>
#include <vector>

#include "camera.h"
#include "mmaths.h"
#include "fragment.h"
#include "properties.h"

class Object
{
 public:
  virtual ~Object();

  virtual void getFragments(const Mat4& outerM, const Camera& cam,
			    FragmentVector& v) const;
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
    surfaceprop = prop;
  }

  void getFragments(const Mat4& outerM, const Camera& cam,
		    FragmentVector& v) const;

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

  void getFragments(const Mat4& outerM, const Camera& cam,
		    FragmentVector& v) const;

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

  void getFragments(const Mat4& outerM, const Camera& cam,
                    FragmentVector& v) const;

private:
  void getSurfaceFragments(const Mat4& outerM, const Camera& cam,
                           FragmentVector& v) const;
  void getLineFragments(const Mat4& outerM, const Camera& cam,
                        FragmentVector& v) const;

  void getVecIdxs(unsigned &vidx_h, unsigned &vidx_1, unsigned &vidx_2) const;

public:
  ValVector pos1, pos2, heights;
  Direction dirn;
  PropSmartPtr<const LineProp> lineprop;
  PropSmartPtr<const SurfaceProp> surfaceprop;
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
  void getFragments(const Mat4& outerM, const Camera& cam,
		    FragmentVector& v) const;

  void addObject(Object* obj)
  {
    objects.push_back(obj);
  }

 public:
  Mat4 objM;
  std::vector<Object*> objects;
};

#endif
