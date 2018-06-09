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

#ifndef MATHS_H
#define MATHS_H

#include <vector>
#include <cmath>

#define PI 3.14159265358979323846

//////////////////////////////////////////////////////////////////////////////
// 4-vector
struct Vec4
{
  Vec4()
  {
    v[0] = v[1] = v[2] = v[3] = 0;
  }
  Vec4(double a, double b, double c, double d=1)
  {
    v[0] = a; v[1] = b; v[2] = c; v[3] = d;
  }

  inline double& operator()(unsigned i) { return v[i]; }
  inline double operator()(unsigned i) const { return v[i]; }

  inline void operator*=(double f)
  {
    v[0] *= f; v[1] *= f; v[2] *= f; v[3] *= f;
  }
  inline Vec4 operator+(const Vec4& o) const
  {
    return Vec4(v[0]+o.v[0], v[1]+o.v[1], v[2]+o.v[2], v[3]+o.v[3]);
  }
  inline Vec4 operator-(const Vec4& o) const
  {
    return Vec4(v[0]-o.v[0], v[1]-o.v[1], v[2]-o.v[2], v[3]-o.v[3]);
  }
  inline Vec4 operator*(double f) const
  {
    return Vec4(v[0]*f, v[1]*f, v[2]*f, v[3]*f);
  }
  Vec4& operator+=(const Vec4& o)
  {
    v[0]+=o.v[0]; v[1]+=o.v[1]; v[2]+=o.v[2]; v[3]+=o.v[3]; return *this;
  }
  Vec4& operator-=(const Vec4& o)
  {
    v[0]-=o.v[0]; v[1]-=o.v[1]; v[2]-=o.v[2]; v[3]-=o.v[3]; return *this;
  }
  inline bool operator==(const Vec4& o) const
  {
    return v[0]==o.v[0] && v[1]==o.v[1] && v[2]==o.v[2] && v[3]==o.v[3];
  }
  inline bool operator!=(const Vec4& o) const
  {
    return !(operator==(o));
  }

  // radius
  inline double rad2() const
  {
    return v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+v[3]*v[3];
  }
  inline double rad() const { return std::sqrt(rad2()); }

  inline void normalise() { operator*=(1/rad()); }

  inline bool isfinite() const { return std::isfinite(v[0]+v[1]+v[2]+v[3]); }

private:
  double v[4];
};

//////////////////////////////////////////////////////////////////////////////
// 3-vector

struct Vec3
{
  Vec3()
  {
    v[0] = v[1] = v[2] = 0;
  }
  Vec3(double a, double b, double c)
  {
    v[0] = a; v[1] = b; v[2] = c;
  }

  inline double& operator()(unsigned i) { return v[i]; }
  inline double operator()(unsigned i) const { return v[i]; }

  inline void operator*=(double f)
  {
    v[0] *= f; v[1] *= f; v[2] *= f;
  }
  inline Vec3 operator+(const Vec3& o) const
  {
    return Vec3(v[0]+o.v[0], v[1]+o.v[1], v[2]+o.v[2]);
  }
  inline Vec3 operator-() const
  {
    return Vec3(-v[0], -v[1], -v[2]);
  }
  inline Vec3 operator-(const Vec3& o) const
  {
    return Vec3(v[0]-o.v[0], v[1]-o.v[1], v[2]-o.v[2]);
  }
  inline Vec3 operator*(double f) const
  {
    return Vec3(v[0]*f, v[1]*f, v[2]*f);
  }
  Vec3& operator+=(const Vec3& o)
  {
    v[0]+=o.v[0]; v[1]+=o.v[1]; v[2]+=o.v[2]; return *this;
  }
  Vec3& operator-=(const Vec3& o)
  {
    v[0]-=o.v[0]; v[1]-=o.v[1]; v[2]-=o.v[2]; return *this;
  }

  inline bool operator==(const Vec3& o) const
  {
    return v[0]==o.v[0] && v[1]==o.v[1] && v[2]==o.v[2];
  }
  inline bool operator!=(const Vec3& o) const
  {
    return !(operator==(o));
  }

  // radius
  inline double rad2() const
  {
    return v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
  }
  inline double rad() const { return std::sqrt(rad2()); }

  inline void normalise() { operator*=(1/rad()); }

  inline bool isfinite() const { return std::isfinite(v[0]+v[1]+v[2]); }

private:
  double v[3];
};

inline Vec3 cross(const Vec3& a, const Vec3& b)
{
  return Vec3(a(1)*b(2)-a(2)*b(1),
	      a(2)*b(0)-a(0)*b(2),
	      a(0)*b(1)-a(1)*b(0));
}

inline double dot(const Vec3& a, const Vec3& b)
{
  return a(0)*b(0)+a(1)*b(1)+a(2)*b(2);
}

//////////////////////////////////////////////////////////////////////////////
// 4x4 matrix

struct Mat4
{
  Mat4(bool zero=true)
  {
    if(zero)
      for(unsigned y=0; y<4; ++y)
	for(unsigned x=0; x<4; ++x)
	  m[y][x] = 0;
  }

  inline double& operator()(unsigned y, unsigned x) { return m[y][x]; }
  inline double operator()(unsigned y, unsigned x) const { return m[y][x]; }

  // matrix multiply
  inline Mat4 operator*(const Mat4& o) const
  {
    Mat4 ret(false);
    for(unsigned y=0; y<4; ++y)
      for(unsigned x=0; x<4; ++x)
	ret.m[y][x] = m[y][0]*o.m[0][x] + m[y][1]*o.m[1][x] +
	  m[y][2]*o.m[2][x] + m[y][3]*o.m[3][x];
    return ret;
  }

  inline Vec4 operator*(const Vec4& v) const
  {
    return Vec4(v(0)*m[0][0]+v(1)*m[0][1]+v(2)*m[0][2]+v(3)*m[0][3],
		v(0)*m[1][0]+v(1)*m[1][1]+v(2)*m[1][2]+v(3)*m[1][3],
		v(0)*m[2][0]+v(1)*m[2][1]+v(2)*m[2][2]+v(3)*m[2][3],
		v(0)*m[3][0]+v(1)*m[3][1]+v(2)*m[3][2]+v(3)*m[3][3]);
  }


  inline Mat4 transpose() const
  {
    Mat4 r(false);
    for(unsigned y=0; y<4; ++y)
      for(unsigned x=0; y<x; ++x)
	r.m[y][x] = m[x][y];
    return r;
  }

private:
  double m[4][4];
};

// multiply matrix by vector
inline Vec4 operator*(const Vec4& v, const Mat4& m)
{
  return Vec4(v(0)*m(0,0)+v(1)*m(1,0)+v(2)*m(2,0)+v(3)*m(3,0),
	      v(0)*m(0,1)+v(1)*m(1,1)+v(2)*m(2,1)+v(3)*m(3,1),
	      v(0)*m(0,2)+v(1)*m(1,2)+v(2)*m(2,2)+v(3)*m(3,2),
	      v(0)*m(0,3)+v(1)*m(1,3)+v(2)*m(2,3)+v(3)*m(3,3));
}

// identity matrix
inline Mat4 identityM4()
{
  Mat4 m(false);
  m(0,0)=1; m(0,1)=0; m(0,2)=0; m(0,3)=0;
  m(1,0)=0; m(1,1)=1; m(1,2)=0; m(1,3)=0;
  m(2,0)=0; m(2,1)=0; m(2,2)=1; m(2,3)=0;
  m(3,0)=0; m(3,1)=0; m(3,2)=0; m(3,3)=1;
  return m;
}

// create a rotation matrix
Mat4 rotateM4(double angle, Vec3 vec);

// create a translation matrix
Mat4 translationM4(Vec3 vec);

// create a scaling matrix
inline Mat4 scaleM4(Vec3 s)
{
  Mat4 m(false);
  m(0,0)=s(0); m(0,1)=0;    m(0,2)=0;    m(0,3)=0;
  m(1,0)=0;    m(1,1)=s(1); m(1,2)=0;    m(1,3)=0;
  m(2,0)=0;    m(2,1)=0;    m(2,2)=s(2); m(2,3)=0;
  m(3,0)=0;    m(3,1)=0;    m(3,2)=0;    m(3,3)=1;
  return m;
}

// rotation matrix in terms of sin and cos of three angles in x,y,z
// doing z rotation, then y, then x
inline Mat4 rotate3M4_cs(double sx, double cx, double sy, double cy,
                         double sz, double cz)
{
  Mat4 m(false);
  m(0,0)=cy*cz; m(0,1)=cz*sx*sy-cx*sz; m(0,2)=cx*cz*sy+sx*sz; m(0,3)=0;
  m(1,0)=cy*sz; m(1,1)=cx*cz+sx*sy*sz; m(1,2)=cx*sy*sz-cz*sx; m(1,3)=0;
  m(2,0)=-sy;   m(2,1)=cy*sx;          m(2,2)=cx*cy;          m(2,3)=0;
  m(3,0)=0;     m(3,1)=0;              m(3,2)=0;              m(3,3)=1;
  return m;
}

inline Mat4 rotate3M4(double ax, double ay, double az)
{
  return rotate3M4_cs(std::sin(ax), std::cos(ax),
                      std::sin(ay), std::cos(ay),
                      std::sin(az), std::cos(az));
}

///////////////////////////////////////////////////////////////////////
// 3-Matrix

struct Mat3
{
  Mat3(bool zero=true)
  {
    if(zero)
      for(unsigned y=0; y<3; ++y)
	for(unsigned x=0; x<3; ++x)
	  m[y][x] = 0;
  }

  Mat3(double a, double b, double c,
       double d, double e, double f,
       double g, double h, double i)
    : m{{a,b,c},{d,e,f},{g,h,i}}
  {}

  inline double& operator()(unsigned y, unsigned x) { return m[y][x]; }
  inline double operator()(unsigned y, unsigned x) const { return m[y][x]; }

  inline Mat3 operator*(const Mat3& o) const
  {
    Mat3 ret(false);
    for(unsigned y=0; y<3; ++y)
      for(unsigned x=0; x<3; ++x)
	ret.m[y][x] = m[y][0]*o.m[0][x] + m[y][1]*o.m[1][x] +
	  m[y][2]*o.m[2][x];
    return ret;
  }
  inline Vec3 operator*(const Vec3& v) const
  {
    return Vec3(v(0)*m[0][0]+v(1)*m[0][1]+v(2)*m[0][2],
		v(0)*m[1][0]+v(1)*m[1][1]+v(2)*m[1][2],
		v(0)*m[2][0]+v(1)*m[2][1]+v(2)*m[2][2]);
  }
  inline Mat3 transpose() const
  {
    Mat3 r(false);
    for(unsigned y=0; y<4; ++y)
      for(unsigned x=0; y<x; ++x)
	r.m[y][x] = m[x][y];
    return r;
  }

private:
  double m[3][3];
};

inline Vec3 operator*(const Vec3& v, const Mat3& m)
{
  return Vec3(v(0)*m(0,0)+v(1)*m(1,0)+v(2)*m(2,0),
	      v(0)*m(0,1)+v(1)*m(1,1)+v(2)*m(2,1),
	      v(0)*m(0,2)+v(1)*m(1,2)+v(2)*m(2,2));
}

// identity matrix
inline Mat3 identityM3()
{
  Mat3 m(false);
  m(0,0)=1; m(0,1)=0; m(0,2)=0;
  m(1,0)=0; m(1,1)=1; m(1,2)=0;
  m(2,0)=0; m(2,1)=0; m(2,2)=1;
  return m;
}

inline Mat3 scaleM3(double s)
{
  Mat3 m(false);
  m(0,0)=s; m(0,1)=0; m(0,2)=0;
  m(1,0)=0; m(1,1)=s; m(1,2)=0;
  m(2,0)=0; m(2,1)=0; m(2,2)=1;
  return m;
}

inline Mat3 translateM3(double dx, double dy)
{
  Mat3 m(false);
  m(0,0)=1; m(0,1)=0; m(0,2)=dx;
  m(1,0)=0; m(1,1)=1; m(1,2)=dy;
  m(2,0)=0; m(2,1)=0; m(2,2)=1;
  return m;
}

// determinant
inline double det(const Mat3& m)
{
  return
    m(0,0)*m(1,1)*m(2,2) - m(2,0)*m(1,1)*m(0,2) +
    m(1,0)*m(2,1)*m(0,2) - m(1,0)*m(0,1)*m(2,2) +
    m(2,0)*m(0,1)*m(1,2) - m(0,0)*m(2,1)*m(1,2);
}

//////////////////////////////////////////////////////////////////////////////
// Two item vector

struct Vec2
{
  Vec2()
  {
    v[0] = v[1] = 0;
  }
  Vec2(double a, double b)
  {
    v[0] = a; v[1] = b;
  }

  inline double& operator()(unsigned i) { return v[i]; }
  inline double operator()(unsigned i) const { return v[i]; }

  inline void operator*=(double f)
  {
    v[0] *= f; v[1] *= f;
  }
  inline Vec2 operator+(const Vec2& o) const
  {
    return Vec2(v[0]+o.v[0], v[1]+o.v[1]);
  }
  inline Vec2 operator-() const
  {
    return Vec2(-v[0], -v[1]);
  }
  inline Vec2 operator-(const Vec2& o) const
  {
    return Vec2(v[0]-o.v[0], v[1]-o.v[1]);
  }
  inline Vec2 operator*(double f) const
  {
    return Vec2(v[0]*f, v[1]*f);
  }
  Vec2& operator+=(const Vec2& o)
  {
    v[0]+=o.v[0]; v[1]+=o.v[1]; return *this;
  }
  Vec2& operator-=(const Vec2& o)
  {
    v[0]-=o.v[0]; v[1]-=o.v[1]; return *this;
  }
  inline bool operator==(const Vec2& o) const
  {
    return v[0]==o.v[0] && v[1]==o.v[1];
  }
  inline bool operator!=(const Vec2& o) const
  {
    return !(operator==(o));
  }

  // radius
  inline double rad2() const
  {
    return v[0]*v[0]+v[1]*v[1];
  }
  inline double rad() const { return std::sqrt(rad2()); }

  inline void normalise() { operator*=(1/rad()); }

  inline bool isfinite() const { return std::isfinite(v[0]+v[1]); }

private:
  double v[2];
};

inline double cross(const Vec2& a, const Vec2& b)
{
  return a(0)*b(1)-a(1)*b(0);
}

inline double dot(const Vec2& a, const Vec2& b)
{
  return a(0)*b(0)+a(1)*b(1);
}

//////////////////////////////////////////////////////////////////////////////
// Helper functions

inline Vec3 vec4to3(const Vec4& v)
{
  double inv = 1/v(3);
  return Vec3(v(0)*inv, v(1)*inv, v(2)*inv);
}

inline Vec4 vec3to4(const Vec3& v)
{
  return Vec4(v(0), v(1), v(2), 1);
}

inline Vec2 vec3to2(const Vec3& v)
{
  return Vec2(v(0), v(1));
}

// do projection, getting x,y coordinate and depth
inline Vec3 calcProjVec(const Mat4& projM, const Vec4& v)
{
  Vec4 nv(projM*v);
  double inv = 1/nv(3);
  return Vec3(nv(0)*inv, nv(1)*inv, nv(2)*inv);
}

inline Vec3 calcProjVec(const Mat4& projM, const Vec3& v)
{
  Vec4 nv(projM*vec3to4(v));
  double inv = 1/nv(3);
  return Vec3(nv(0)*inv, nv(1)*inv, nv(2)*inv);
}

// convert projected coordinates to screen coordinates using screen matrix
// makes (x,y,depth) -> screen coordinates
inline Vec2 projVecToScreen(const Mat3& screenM, const Vec3& vec)
{
  Vec3 mult(screenM*Vec3(vec(0), vec(1), 1));
  double inv = 1/mult(2);
  return Vec2(mult(0)*inv, mult(1)*inv);
}

// do 2d lines overlap?
inline bool line2DOverlap(Vec2 A1, Vec2 A2, Vec2 B1, Vec2 B2)
{
  double d = cross(A2-A1, B2-B1);
  double u = cross(B2-B1, A1-B1);
  double v = cross(A2-A1, A1-B1);

  if(d>=0)
    return 0<=u && u<=d && 0<=v && v<=d;
  else
    return 0>=u && u>=d && 0>=v && v>=d;
}

// drop dimension of vector
inline Vec2 dropDim(const Vec3 v)
{
  return Vec2(v(0), v(1));
}

//////////////////////////////////////////////////////////////////////////////
// Helper types

typedef std::vector<Vec2> Vec2Vector;
typedef std::vector<Vec3> Vec3Vector;
typedef std::vector<Vec4> Vec4Vector;
typedef std::vector<double> ValVector;

//////////////////////////////////////////////////////////////////////////////

#if 0
// debug:
#include <cstdio>
using std::printf;

inline void print(const Mat4& m)
{
  printf("% 7.3f % 7.3f % 7.3f % 7.3f\n", m(0,0), m(0,1), m(0,2), m(0,3));
  printf("% 7.3f % 7.3f % 7.3f % 7.3f\n", m(1,0), m(1,1), m(1,2), m(1,3));
  printf("% 7.3f % 7.3f % 7.3f % 7.3f\n", m(2,0), m(2,1), m(2,2), m(2,3));
  printf("% 7.3f % 7.3f % 7.3f % 7.3f\n", m(3,0), m(3,1), m(3,2), m(3,3));
}

inline void print(const Mat3& m)
{
  printf("% 7.3f % 7.3f % 7.3f\n", m(0,0), m(0,1), m(0,2));
  printf("% 7.3f % 7.3f % 7.3f\n", m(1,0), m(1,1), m(1,2));
  printf("% 7.3f % 7.3f % 7.3f\n", m(2,0), m(2,1), m(2,2));
}

inline void print(const Vec4& v)
{
  printf("% 7.3f % 7.3f % 7.3f % 7.3f\n", v(0), v(1), v(2), v(3));
}

inline void print(const Vec3& v)
{
  printf("% 7.3f % 7.3f % 7.3f\n", v(0), v(1), v(2));
}

#endif

#endif
