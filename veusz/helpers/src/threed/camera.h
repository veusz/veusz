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

  // fovy_degrees: field of view in y direction in degrees
  // aspect: aspect ratio
  // znear: clip things nearer than this (should be as big as
  //        possible for precision)
  //zfar: far clipping plane.
  void setPerspective(float fovy_degrees=45, float aspect=1,
		      float znear=0.1, float zfar=100.);

 public:
  Mat4 viewM;   // view matrix
  Mat4 perspM;  // perspective matrix
  Mat4 combM;   // combined matrix
  Vec3 eye;     // location of eye
};

#endif
