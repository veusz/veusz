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

#include <cmath>
#include <algorithm>
#include "twod.h"

#define EPS 1e-8

ISect twodLineIntersect(Vec2 p1, Vec2 p2, Vec2 q1, Vec2 q2,
                        Vec2* posn, Vec2* posn2)
{
  Vec2 dp = p2-p1;
  Vec2 dq = q2-q1;
  Vec2 dpq = p1-q1;
  double denom = cross(dp, dq);

  // parallel vectors or points below
  if(std::abs(denom) < EPS)
    {
      if( std::abs(cross(dp, dpq)) > EPS || std::abs(cross(dq, dpq)) > EPS )
        return LINE_NOOVERLAP;

      // colinear segments - do they overlap?
      double u0, u1;
      Vec2 dpq2 = p2-q1;
      if(std::abs(dq(0)) > std::abs(dq(1)))
        {
          u0 = dpq(0)*(1/dq(0));
          u1 = dpq2(0)*(1/dq(0));
        }
      else
        {
          u0 = dpq(1)*(1/dq(1));
          u1 = dpq2(1)*(1/dq(1));
        }

      if(u0 > u1)
        std::swap(u0, u1);

      if( u0>(1+EPS) || u1<-EPS )
        return LINE_NOOVERLAP;

      u0 = std::max(u0, 0.);
      u1 = std::min(u1, 1.);
      if(posn != 0)
        *posn = q1 + dq*u0;
      if( std::abs(u0-u1) < EPS )
        return LINE_CROSS;
      if(posn2 != 0)
        *posn2 = q1 + dq*u1;
      return LINE_OVERLAP;
    }

  double s = cross(dq, dpq)*(1/denom);
  if(s < -EPS || s > (1+EPS))
    return LINE_NOOVERLAP;
  double t = cross(dp, dpq)*(1/denom);
  if(t < -EPS || t > (1+EPS))
    return LINE_NOOVERLAP;

  if(posn != 0)
    *posn = p1 + dp * std::max(std::min(s, 1.), 0.);

  return LINE_CROSS;
}


namespace
{
  // is a to the left of p1->p2?
  inline bool ptInside(Vec2 p, Vec2 cp0, Vec2 cp1)
  {
    return (cp1(0)-cp0(0))*(p(1)-cp0(1)) > (cp1(1)-cp0(1))*(p(0)-cp0(0));
  };

  // version of above with tolerence of points on line
  // 0: on line, -1: outside, +1: inside
  inline int ptInsideTol(Vec2 p, Vec2 cp0, Vec2 cp1)
  {
    double del = (cp1(0)-cp0(0))*(p(1)-cp0(1)) - (cp1(1)-cp0(1))*(p(0)-cp0(0));
    return del>EPS ? 1 : del<-EPS ? -1 : 0;
  }

  // return whether lines intersect and return intersection point
  bool SHlineIntersection(Vec2 a0, Vec2 a1, Vec2 b0, Vec2 b1, Vec2* res)
  {
    Vec2 da = a0-a1;
    Vec2 db = b0-b1;
    double n1 = cross(a0, a1);
    double n2 = cross(b0, b1);
    double denom = cross(da, db);
    if(denom == 0)
      return 0;
    double idenom = 1/denom;
    *res = db*(n1*idenom) - da*(n2*idenom);
    return 1;
  }
}

// Sutherland–Hodgman algorithm for clipping polygon against
// 2nd polygon. Requires clockwise orientation of points.
Vec2Vector twodPolyEdgeClip(Vec2Vector inPoly, const Vec2Vector& clipPoly)
{
  if(clipPoly.empty())
    return inPoly;
  Vec2 cp1 = clipPoly[clipPoly.size()-1];
  for(unsigned ci=0; ci != clipPoly.size() && !inPoly.empty(); ++ci)
    {
      Vec2 cp2 = clipPoly[ci];

      Vec2Vector outPoly;
      Vec2 S = inPoly[inPoly.size()-1];
      for(unsigned si=0; si != inPoly.size(); ++si)
        {
          Vec2 E = inPoly[si];
          if(ptInside(E, cp1, cp2))
            {
              if(!ptInside(S, cp1, cp2))
                {
                  Vec2 isect;
                  if(SHlineIntersection(S, E, cp1, cp2, &isect))
                    outPoly.push_back(isect);
                }
              outPoly.push_back(E);
            }
          else if(ptInside(S, cp1, cp2))
            {
              Vec2 isect;
              if(SHlineIntersection(S, E, cp1, cp2, &isect))
                outPoly.push_back(isect);
            }
          S = E;
        }
      inPoly = outPoly;
      cp1 = cp2;
    }
  return inPoly;
}

double twodPolyArea(const Vec2Vector& poly)
{
  const unsigned s=poly.size();
  double tot=0;
  for(unsigned i=0; i<s; ++i)
    tot += poly[i](0)*poly[(i+1)%s](1) - poly[(i+1)%s](0)*poly[i](1);
  return 0.5*tot;
};

void twodPolyMakeClockwise(Vec2Vector* poly)
{
  if( twodPolyArea(*poly) < 0 )
    std::reverse(poly->begin(), poly->end());
}

bool twodLineIntersectPolygon(Vec2 p1, Vec2 p2, const Vec2Vector& poly)
{
  const unsigned s=poly.size();
  bool inside1=1;
  bool inside2=1;

  for(unsigned i=0; i<s; ++i)
    {
      Vec2 e1 = poly[i];
      Vec2 e2 = poly[(i+1)%s];

      // are any of the points inside?
      int insidep1 = ptInsideTol(p1, e1, e2);
      if(insidep1 != 1)
        inside1=0;
      int insidep2 = ptInsideTol(p2, e1, e2);
      if(insidep2 != 1)
        inside2=0;

      // check for line intersection if one of the edges doesn't touch
      // an edge
      if( insidep1 != 0 && insidep2 != 0 )
        if( twodLineIntersect(p1, p2, e1, e2, 0, 0) == LINE_CROSS )
          return 1;
    }

  // one of the points is inside
  return inside1 || inside2;
}
