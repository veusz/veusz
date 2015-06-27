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

#ifndef FRAGMENT_H
#define FRAGMENT_H

#include <QtGui/QPainter>
#include <QtGui/QPainterPath>
#include <limits>
#include <vector>
#include "properties.h"
#include "mmaths.h"

#define LINE_DELTA_DEPTH 1e-3

// from objects.h
class Object;

// this is passed to the renderer to get the parameters for painting
// the path

struct FragmentParameters
{
  virtual ~FragmentParameters();
};

struct FragmentPathParameters : public FragmentParameters
{
  QPainterPath* path;
  bool scaleedges;
  bool runcallback;
  // optional callback function if runcallback is set
  virtual void callback(QPainter* painter, QPointF pt1, QPointF pt2,
                        unsigned index,  double scale, double linescale);
};

// created by drawing Objects to draw to screen
struct Fragment
{
  enum FragmentType {FR_NONE, FR_TRIANGLE, FR_LINESEG, FR_PATH};

  // type of fragment
  FragmentType type;

  // pointer to object, to avoid self-comparison.
  Object* object;
  // optional pointer to a paramters object
  FragmentParameters* params;

  // drawing style
  SurfaceProp const* surfaceprop;
  LineProp const* lineprop;

  // 3D points
  Vec3 points[3];

  // projected points associated with fragment
  Vec3 proj[3];

  // for path
  float pathsize;

  // calculated lighting norm (updated by Scene)
  float lighting;

  // number of times this has been split
  unsigned splitcount;

  // passed to path plotting or as index to color bar
  unsigned index;

  // zero on creation
  Fragment()
  : type(FR_NONE),
    object(0),
    params(0),
    surfaceprop(0),
    lineprop(0),
    pathsize(0),
    lighting(1),
    splitcount(0),
    index(0)
  {
  }

  // number of (visible) points used by fragment type
  unsigned nPointsVisible() const
  {
    switch(type)
      {
      case FR_TRIANGLE: return 3;
      case FR_LINESEG: return 2;
      case FR_PATH: return 1;
      default: return 0;
      }
  }

  // number of points used by fragment, including hidden ones
  // FR_PATH has an optional 2nd point for keeping track of a baseline
  unsigned nPointsTotal() const
  {
    switch(type)
      {
      case FR_TRIANGLE: return 3;
      case FR_LINESEG: return 2;
      case FR_PATH: return 2;
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
	return proj[0](2) - 2*LINE_DELTA_DEPTH;
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
	return proj[0](2) - 2*LINE_DELTA_DEPTH;
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
	return proj[0](2) - 2*LINE_DELTA_DEPTH;
      default:
	return std::numeric_limits<double>::infinity();
      }
  }

  // recalculate projected coordinates
  void updateProjCoords(const Mat4& projM)
  {
    unsigned n=nPointsTotal();
    for(unsigned i=0; i<n; ++i)
      proj[i] = calcProjVec(projM, points[i]);
  }

};

typedef std::vector<Fragment> FragmentVector;


#endif
