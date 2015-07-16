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
