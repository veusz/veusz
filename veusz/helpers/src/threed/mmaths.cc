#include <cmath>
#include "mmaths.h"

Mat4 rotateM(float angle, Vec3 vec)
{
  float c = std::cos(angle);
  float s = std::sin(angle);

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

Mat4 translationM(Vec3 vec)
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

