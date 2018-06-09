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
#include "mmaths.h"

Mat4 rotateM4(double angle, Vec3 vec)
{
  double c = std::cos(angle);
  double s = std::sin(angle);

  Vec3 a(vec);
  a.normalise();
  Vec3 t(a*(1-c));

  Mat4 m;
  m(0,0) = c+t(0)*a(0);
  m(0,1) = 0+t(1)*a(0)-s*a(2);
  m(0,2) = 0+t(2)*a(0)+s*a(1);
  m(0,3) = 0;

  m(1,0) = 0+t(0)*a(1)+s*a(2);
  m(1,1) = c+t(1)*a(1);
  m(1,2) = 0+t(2)*a(1)-s*a(0);
  m(1,3) = 0;

  m(2,0) = 0+t(0)*a(2)-s*a(1);
  m(2,1) = 0+t(1)*a(2)+s*a(0);
  m(2,2) = c+t(2)*a(2);
  m(2,3) = 0;

  m(3,0) = 0;
  m(3,1) = 0;
  m(3,2) = 0;
  m(3,3) = 1;

  return m;
}

Mat4 translationM4(Vec3 vec)
{
  Mat4 m;

  m(0,0) = 1;
  m(0,3) = vec(0);
  m(1,1) = 1;
  m(1,3) = vec(1);
  m(2,2) = 1;
  m(2,3) = vec(2);
  m(3,3) = 1;

  return m;
}
