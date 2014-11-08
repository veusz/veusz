#ifndef SHAPES_H
#define SHAPES_H

#include <algorithm>
#include <vector>

#include "camera.h"
#include "mmaths.h"

struct SurfaceProp
{
  SurfaceProp(float _r=0.5f, float _g=0.5f, float _b=0.5f,
	      float _specular=0.5f, float _diffuse=0.5f, float _trans=0,
	      bool _hide=0)
    : r(_r), g(_g), b(_b),
    specular(_specular), diffuse(_diffuse), trans(_trans), hide(_hide)
  {
  }

  float r, g, b;
  float specular, diffuse, trans;
  bool hide;
};

struct LineProp
{
  LineProp(float _r=0, float _g=0, float _b=0,
	   float _specular=0.5f, float _diffuse=0.5f, float _trans=0,
	   float _width=1, bool _hide=0)
    : r(_r), g(_g), b(_b),
      specular(_specular), diffuse(_diffuse), trans(_trans),
    width(_width), hide(_hide)
  {
  }

  float r, g, b;
  float specular, diffuse, trans;
  float width;
  bool hide;
};

// structure returned from object
struct Fragment
{
  enum FragmentType {FR_TRIANGLE, FR_LINESEG, FR_PATH};

  // type of fragment
  FragmentType type;

  // 3D points
  Vec4 points[3];

  // projected points associated with fragment
  Vec3 proj[3];

  // point to object or QPainterPath
  void* object;

  // drawing style
  SurfaceProp const* surfaceprop;
  LineProp const* lineprop;
};;

typedef std::vector<Fragment> FragmentVector;

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
  Triangle(const Vec4& a, const Vec4& b, const Vec4& c)
  {
    points[0] = a; points[1] = b; points[2] = c;
    surfaceprop = 0;
  }

  void getFragments(const Mat4& outerM, const Camera& cam,
		    FragmentVector& v) const;

 public:
  Vec4 points[3];
  const SurfaceProp* surfaceprop;
};

class PolyLine : public Object
{
 public:
  PolyLine()
    : Object(), lineprop(0)
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
  const LineProp* lineprop;
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
