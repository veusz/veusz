#ifndef FRAGMENT_H
#define FRAGMENT_H

#include <limits>
#include <vector>
#include "properties.h"
#include "mmaths.h"

#define LINE_DELTA_DEPTH 1e-3

// created by drawing Objects to draw to screen
struct Fragment
{
  enum FragmentType {FR_NONE, FR_TRIANGLE, FR_LINESEG, FR_PATH};

  // type of fragment
  FragmentType type;

  // point to object or QPainterPath
  void* object;

  // drawing style
  SurfaceProp const* surfaceprop;
  LineProp const* lineprop;

  // 3D points
  Vec3 points[3];

  // projected points associated with fragment
  Vec3 proj[3];

  // number of times this has been split
  unsigned splitcount;

  // for debugging
  unsigned index;

  static unsigned _count;

  // zero on creation
  Fragment()
  : type(FR_NONE),
    object(0),
    surfaceprop(0),
    lineprop(0),
    //meandepth(-1), maxdepth(-1), meandepth(-1),
    splitcount(0)
  {
    index=_count;
    ++_count;
  }

  void bumpIndex()
  {
    index=_count;
    ++_count;
  }

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

  double minDepth() const
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
	return std::numeric_limits<double>::infinity();
      }
  }
  double maxDepth() const
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
	return std::numeric_limits<double>::infinity();
      }
  }
  double meanDepth() const
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
	return std::numeric_limits<double>::infinity();
      }
  }

  // recalculate projected coordinates
  void updateProjCoords(const Mat4& projM)
  {
    for(unsigned i=0, n=nPoints(); i<n; ++i)
      proj[i] = calcProjVec(projM, points[i]);
  }

};

typedef std::vector<Fragment> FragmentVector;

// #include <cassert>
// class FragmentVector
// {
//  public:
//   //typedef const Fragment* const_iterator;

//   class const_iterator
//   {
//   public:
//     const_iterator(const FragmentVector& _v, unsigned _i)
//       : v(_v), i(_i);

//     const Fragment& operator->() const
//     {
//       assert(i<v.size());
//       return v[i];
//     }

//     const FragmentVector& v;
//     unsigned i;
//   };


//   FragmentVector():num(0) {};
//   void clear() { num=0;}

//   void push_back(const Fragment& f) { vecs[num++]=f; }

//   Fragment& operator[](unsigned i) {
//     assert(i<num);
//     return vecs[i];
//   }
//   const Fragment& operator[](unsigned i) const {
//     assert(i<num);
//     return vecs[i];
//   }
//   unsigned size() const { return num; }

//   Fragment* begin() { return &vecs[0]; }
//   Fragment* end() { return &vecs[num]; }
//   const Fragment* begin() const { return &vecs[0]; }
//   const Fragment* end() const { return &vecs[num]; }

//   unsigned num;
//   Fragment vecs[1000];
// };


// try to split fragments into pieces if they overlap
// returns number of pieces for each part after fragmentation
void splitFragments(const Fragment& f1, const Fragment& f2,
		    FragmentVector& v,
		    unsigned* num1, unsigned* num2);

// get average depths of intersection in 2D
void overlapDepth(const Fragment& f1, const Fragment& f2,
		  double* d1, double* d2);

#endif
