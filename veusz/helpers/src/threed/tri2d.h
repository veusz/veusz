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

#ifndef TRI2D_H
#define TRI2D_H

#include <cmath>
#include <vector>
#include "mmaths.h"

struct Triangle2D
{
  Triangle2D()
  {}
  Triangle2D(const Vec2 &a, const Vec2 &b, const Vec2 &c)
  {
    pts[0] = a; pts[1] = b; pts[2] = c;
  }
  Vec2& operator[](unsigned i) { return pts[i]; }
  Vec2 operator[](unsigned i) const { return pts[i]; }
  Vec2 pts[3];
};

// take two triangles, and split into separate triangles, writing
// those in both, those in number 1 and those in number 2
// return true if ok
bool clip_triangles_2d(const Triangle2D& tri1, const Triangle2D& tri2,
                       std::vector<Triangle2D>& tris_both,
                       std::vector<Triangle2D>& tris1,
                       std::vector<Triangle2D>& tris2);


// compute area of triangle
inline double triangle_area(const Triangle2D& tri)
{
  return 0.5*std::abs(cross(tri[0],tri[1]) + cross(tri[1],tri[2]) +
                      cross(tri[2],tri[0]));
}

#endif
