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

#include <array>
#include <cmath>
#include <limits>
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

// wrap degrees from 0..360 between -180..180
inline double wrap180(double x)
{
  return x<=180 ? x : x-360;
}

// find rotation which makes 3d vector close to screen position given
void solveRotation(const Vec2& screenPos, const Mat4& perspViewM,
                   const Mat3& screenM, const Vec4& inVec,
                   double thetax, double thetay, double thetaz,
                   double* thetaxp, double* thetayp)
{
  constexpr int step = 1; // degree steps to check
  constexpr int nstep = 360 / step; // number of steps
  constexpr double maxdist = 0.1;
  constexpr double maxdist2 = maxdist*maxdist;

  // precalculate sin and cos table
  std::array<double, nstep> c, s;
  for(int i=0;i<nstep;++i)
    {
      const double t = i*step*DEG2RAD;
      s[i] = std::sin(t);
      c[i] = std::cos(t);
    }

  // original rotation matrix
  const Mat4 baserotM = rotate3M4(thetax*DEG2RAD, thetay*DEG2RAD, thetaz*DEG2RAD);
  const Mat4 origrotM = baserotM * rotate3M4(*thetaxp*DEG2RAD, *thetayp*DEG2RAD, 0);

  // basis vectors
  const Vec4 vx(1,0,0,1);
  const Vec4 vy(0,1,0,1);
  const Vec4 vz(0,0,1,1);

  // and those rotated by the original rotation
  const Vec3 origvx = vec4to3(origrotM * vx);
  const Vec3 origvy = vec4to3(origrotM * vy);
  const Vec3 origvz = vec4to3(origrotM * vz);

  double mindist2s = std::numeric_limits<double>::infinity();
  int bestix=-1, bestiy=-1;

  for(int ix=0;ix<nstep;++ix)
    for(int iy=0;iy<nstep;++iy)
      {
        // rotatate vector for original and prime angles
        const Mat4 rotM = baserotM * rotate3M4_cs(s[ix], c[ix], s[iy], c[iy], 0, 1);

        // rotate basis vectors
        const double dist2x = (vec4to3(rotM*vx) - origvx).rad2();
        const double dist2y = (vec4to3(rotM*vy) - origvy).rad2();
        const double dist2z = (vec4to3(rotM*vz) - origvz).rad2();

        // only allow vectors to shift in 3D by a certain amount
        // (avoiding flips back and forward)
        if( dist2x<maxdist2 && dist2y<maxdist2 && dist2z<maxdist2 )
          {
            const Vec4 rotV = rotM * inVec;
            const Vec3 proj = vec4to3(perspViewM * rotV);
            const Vec2 screen = projVecToScreen(screenM, proj);

            // distance from this point on screen and last one
            const double dist2s = (screenPos-screen).rad2();

            if( dist2s < mindist2s )
              {
                mindist2s = dist2s;
                bestix=ix; bestiy=iy;
              }
          }
      }

  // return best values between -180 and 180
  *thetaxp = wrap180(bestix*step);
  *thetayp = wrap180(bestiy*step);
}
