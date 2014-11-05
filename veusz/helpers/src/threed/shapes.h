#ifndef SHAPES_H
#define SHAPES_H

#include <vector>
#include "mmaths.h"

enum FragmentType {FR_TRIANGLE, FR_LINE, FR_PATH};

// structure returned from object
struct Fragment
{
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
    {
      sceneM = identityM();
    }
  virtual ~Object();

  virtual void getFragments(const Mat4& outerM, FragmentVec& v) const = 0;

 public:
  Mat4 sceneM;
};


class Triangle : public Object
{
 public:
  void getFragments(const Mat4& outerM, FragmentVec& v) const;

 public:
  Vec4 points[3];
};


#endif
