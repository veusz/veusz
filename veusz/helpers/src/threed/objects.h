#ifndef SHAPES_H
#define SHAPES_H

#include <vector>
#include "camera.h"
#include "mmaths.h"

struct SurfaceProp
{
  SurfaceProp(float _r=0.5f, float _g=0.5f, float _b=0.5f,
	      float _specular=0.5f, float _diffuse=0.5f, float _trans=0)
    : r(_r), g(_g), b(_b),
      specular(_specular), diffuse(_diffuse), trans(_trans)
  {
  }

  float r, g, b;
  float specular, diffuse, trans;
};

struct LineProp
{
  LineProp(float _r=0, float _g=0, float _b=0,
	   float _specular=0.5f, float _diffuse=0.5f, float _trans=0,
	   float _width=1)
    : r(_r), g(_g), b(_b),
      specular(_specular), diffuse(_diffuse), trans(_trans),
      width(_width)
  {
  }

  float r, g, b;
  float specular, diffuse, trans;
  float width;
};

// structure returned from object
struct Fragment
{
  enum FragmentType {FR_TRIANGLE, FR_LINESEG, FR_PATH};

  // type of fragment
  FragmentType type;

  // 3D points
  Vec4 points3d[3];

  // projected points associated with fragment
  Vec3 projpoints[3];

  // point to object or QPainterPath
  void* object;

  // drawing style
  SurfaceProp const* surfaceprop;
  LineProp const* lineprop;
};

typedef std::vector<Fragment> FragmentVector;

class Object
{
 public:
  Object()
    : objM(identityM())
    {
    }
  virtual ~Object();

  virtual void getFragments(const Mat4& outerM, const Camera& cam,
			    FragmentVector& v) const = 0;

 public:
  Mat4 objM;
};

class Triangle : public Object
{
 public:
  Triangle()
    : Object(), surfaceprop(0)
  {
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

  void getFragments(const Mat4& outerM, const Camera& cam,
		    FragmentVector& v) const;

 public:
  Vec4Vector points;
  LineProp const* lineprop;
};

#endif
