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
