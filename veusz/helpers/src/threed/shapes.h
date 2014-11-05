#ifndef SHAPES_H
#define SHAPES_H

#include <vector>
#include "camera.h"
#include "mmaths.h"



// structure returned from object
struct Fragment
{
  enum FragmentType {FR_TRIANGLE, FR_LINESEG, FR_PATH};

  // type of fragment
  FragmentType type;

  // points associated with fragment
  Vec3 points[3];

  // point to object or QPainterPath
  void* object;
};


typedef std::vector<Fragment> FragmentVec;

class Object
{
 public:
  Object()
    : objM(identityM())
    {
    }
  virtual ~Object();

  virtual void getFragments(const Mat4& outerM, const Camera& cam,
			    FragmentVec& v) const = 0;

 public:
  Mat4 objM;
};


class Triangle : public Object
{
 public:
  void getFragments(const Mat4& outerM, const Camera& cam,
		    FragmentVec& v) const;

 public:
  Vec4 points[3];
};

class PolyLine : public Object
{
 public:
  void getFragments(const Mat4& outerM, const Camera& cam,
		    FragmentVec& v) const;

 public:
  std::vector<Vec4> points;
};

#endif
