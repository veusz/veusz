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

#ifndef CAMERA_H
#define CAMERA_H

#include "mmaths.h"

class Camera
{
 public:
  Camera();

  // Look at target position from eye, given up vector.
  // See glm code lookAt
  void setPointing(const Vec3 &eye, const Vec3 &target, const Vec3 &up);

  // fovy_degrees: total field of view in degrees
  // znear: clip things nearer than this (should be as big as
  //        possible for precision)
  // zfar: far clipping plane.
  void setPerspective(double fov_degrees=90,
		      double znear=0.1, double zfar=100);

 public:
  Mat4 viewM;   // view matrix
  Mat4 perspM;  // perspective matrix
  Mat4 combM;   // combined matrix
  Vec3 eye;     // location of eye
};

#endif
