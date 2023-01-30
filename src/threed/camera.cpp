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
#include "camera.h"

Camera::Camera()
{
  setPointing(Vec3(0,0,0), Vec3(0,0,1), Vec3(0,1,0));
  setPerspective();
}

void Camera::setPointing(const Vec3 &_eye, const Vec3 &target, const Vec3 &up)
{
  // is it this one or the one below?
  // http://3dgep.com/?p=1700

  eye = _eye;

  Vec3 f = target - eye;
  f.normalise();
  Vec3 u = up;
  u.normalise();
  Vec3 s = cross(f, u);
  s.normalise();
  u = cross(s, f);

  viewM(0,0) = s(0);
  viewM(0,1) = s(1);
  viewM(0,2) = s(2);
  viewM(0,3) = -dot(s, eye);

  viewM(1,0) = u(0);
  viewM(1,1) = u(1);
  viewM(1,2) = u(2);
  viewM(1,3) = -dot(u, eye);

  viewM(2,0) = -f(0);
  viewM(2,1) = -f(1);
  viewM(2,2) = -f(2);
  viewM(2,3) = dot(f, eye);

  viewM(3,0) = 0;
  viewM(3,1) = 0;
  viewM(3,2) = 0;
  viewM(3,3) = 1;

  combM = perspM * viewM;
}

void Camera::setPerspective(double fov_degrees,
			    double znear, double zfar)
{
  // matrix from Scratchapixel 2
  // https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix

  double scale = 1/std::tan(fov_degrees*(PI/180/2));

  perspM(0,0) = scale;
  perspM(1,0) = 0;
  perspM(2,0) = 0;
  perspM(3,0) = 0;

  perspM(0,1) = 0;
  perspM(1,1) = scale;
  perspM(2,1) = 0;
  perspM(3,1) = 0;

  perspM(0,2) = 0;
  perspM(1,2) = 0;
  perspM(2,2) = -zfar/(zfar-znear);
  perspM(3,2) = -1;

  perspM(0,3) = 0;
  perspM(1,3) = 0;
  perspM(2,3) = -zfar*znear/(zfar-znear);
  perspM(3,3) = 0;

  combM = perspM * viewM;
}
