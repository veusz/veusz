#ifndef FRAGMENT_H
#define FRAGMENT_H

#include <limits>
#include <vector>
#include "properties.h"

#define LINE_DELTA_DEPTH 1e-4

struct FragBounds
{
  float minx, miny, minz, maxx, maxy, maxz;
};

// created by drawing Objects to draw to screen
struct Fragment
{
  enum FragmentType {FR_TRIANGLE, FR_LINESEG, FR_PATH};

  // type of fragment
  FragmentType type;

  // 3D points
  Vec3 points[3];

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

  float minDepth() const
  {
    switch(type)
      {
      case FR_TRIANGLE:
	return std::min(proj[0](2), std::min(proj[1](2), proj[2](2)));
      case FR_LINESEG:
	return std::min(proj[0](2), proj[1](2)) - LINE_DELTA_DEPTH;
      case FR_PATH:
	return proj[0](2);
      default:
	return std::numeric_limits<float>::infinity();
      }
  }
  float maxDepth() const
  {
    switch(type)
      {
      case FR_TRIANGLE:
	return std::max(proj[0](2), std::max(proj[1](2), proj[2](2)));
      case FR_LINESEG:
	return std::max(proj[0](2), proj[1](2)) - LINE_DELTA_DEPTH;
      case FR_PATH:
	return proj[0](2);
      default:
	return std::numeric_limits<float>::infinity();
      }
  }
  float meanDepth() const
  {
    switch(type)
      {
      case FR_TRIANGLE:
	return (proj[0](2) + proj[1](2) + proj[2](2))*(1/3.f);
      case FR_LINESEG:
	return (proj[0](2) + proj[1](2))*0.5f - LINE_DELTA_DEPTH;
      case FR_PATH:
	return proj[0](2);
      default:
	return std::numeric_limits<float>::infinity();
      }
  }

  // get bounds of fragment
  FragBounds bounds() const
  {
    FragBounds b;
    b.minx = b.miny = b.minz = std::numeric_limits<float>::infinity();
    b.maxx = b.maxy = b.maxz = -std::numeric_limits<float>::infinity();
    for(unsigned i=0, s=nPoints(); i<s; ++i)
      {
	const Vec3& pt = points[i];
	b.minx=std::min(b.minx, pt(0)); b.maxx=std::max(b.maxx, pt(0));
	b.miny=std::min(b.miny, pt(1)); b.maxy=std::max(b.maxy, pt(1));
	b.minz=std::min(b.minz, pt(2)); b.maxz=std::max(b.maxz, pt(2));
      }
    return b;
  }

  // does this overlap with other fragment
  bool overlaps(const FragBounds& b) const
  {
    // see whether other one is inside this region
    for(unsigned i=0, s=nPoints(); i<s; ++i)
      {
	const Vec3& pt = points[i];
	if(pt(0) > b.minx && pt(0) < b.maxx &&
	   pt(1) > b.miny && pt(1) < b.maxy &&
	   pt(2) > b.minz && pt(2) < b.maxz)
	  return 1;
      }
    return 0;
  }

};

typedef std::vector<Fragment> FragmentVector;

#endif
