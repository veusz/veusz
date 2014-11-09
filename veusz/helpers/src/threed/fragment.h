#ifndef FRAGMENT_H
#define FRAGMENT_H

#include <vector>
#include "properties.h"

// created by drawing Objects to draw to screen
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

  // number of points used by fragment type
  unsigned nPoints() const
  {
    switch(type)
      {
      case FR_TRIANGLE: return 3;
      case FR_LINESEG: return 2;
      case FR_PATH: return 1;
      default: return 0;
      }
  }
};

typedef std::vector<Fragment> FragmentVector;

#endif
