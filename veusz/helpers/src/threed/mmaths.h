#ifndef MATHS_H
#define MATHS_H

#include <vector>
#include <cmath>

//////////////////////////////////////////////////////////////////////////////
// 4-vector
struct Vec4
{
  Vec4()
  {
    v[0] = v[1] = v[2] = v[3] = 0;
  }
  Vec4(float a, float b, float c, float d=1)
  {
    v[0] = a; v[1] = b; v[2] = c; v[3] = d;
  }

  inline float& operator()(unsigned i) { return v[i]; }
  inline float operator()(unsigned i) const { return v[i]; }

  inline void set(unsigned i, float f) { v[i] = f; }
  inline float get(unsigned i) const { return v[i]; }

  inline void operator*=(float f)
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
  inline Vec4 operator*(float f) const
  {
    return Vec4(v[0]*f, v[1]*f, v[2]*f, v[3]*f);
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
  inline float rad2() const
  {
    return v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+v[3]*v[3];
  }
  inline float rad() const { return std::sqrt(rad2()); }

  inline void normalise() { operator*=(1/rad()); }

private:
  float v[4];
};

//////////////////////////////////////////////////////////////////////////////
// 3-vector

struct Vec3
{
  Vec3()
  {
    v[0] = v[1] = v[2] = 0;
  }
  Vec3(float a, float b, float c)
  {
    v[0] = a; v[1] = b; v[2] = c;
  }

  inline float& operator()(unsigned i) { return v[i]; }
  inline float operator()(unsigned i) const { return v[i]; }
  inline void set(unsigned i, float f) { v[i] = f; }
  inline float get(unsigned i) const { return v[i]; }

  inline void operator*=(float f)
  {
    v[0] *= f; v[1] *= f; v[2] *= f;
  }
  inline Vec3 operator+(const Vec3& o) const
  {
    return Vec3(v[0]+o.v[0], v[1]+o.v[1], v[2]+o.v[2]);
  }
  inline Vec3 operator-(const Vec3& o) const
  {
    return Vec3(v[0]-o.v[0], v[1]-o.v[1], v[2]-o.v[2]);
  }
  inline Vec3 operator*(float f) const
  {
    return Vec3(v[0]*f, v[1]*f, v[2]*f);
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
  inline float rad2() const
  {
    return v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
  }
  inline float rad() const { return std::sqrt(rad2()); }

  inline void normalise() { operator*=(1/rad()); }

private:
  float v[3];
};

inline Vec3 cross(const Vec3& a, const Vec3& b)
{
  return Vec3(a(1)*b(2)-a(2)*b(1),
	      a(2)*b(0)-a(0)*b(2),
	      a(0)*b(1)-a(1)*b(0));
}

inline float dot(const Vec3& a, const Vec3& b)
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

  inline float& operator()(unsigned y, unsigned x) { return m[y][x]; }
  inline float operator()(unsigned y, unsigned x) const { return m[y][x]; }
  inline void set(unsigned i, unsigned j, float f) { m[i][j] = f; }
  inline float get(unsigned i, unsigned j) const { return m[i][j]; }

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
  float m[4][4];
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
Mat4 rotateM4(float angle, Vec3 vec);

// create a translation matrix
Mat4 translationM4(Vec3 vec);

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

  inline float& operator()(unsigned y, unsigned x) { return m[y][x]; }
  inline float operator()(unsigned y, unsigned x) const { return m[y][x]; }
  inline void set(unsigned i, unsigned j, float f) { m[i][j] = f; }
  inline float get(unsigned i, unsigned j) const { return m[i][j]; }

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
  float m[3][3];
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

//////////////////////////////////////////////////////////////////////////////
// Helper functions

// do projection, getting x,y coordinate and depth
inline Vec3 calcProjVec(const Mat4& projM, const Vec4& v)
{
  Vec4 nv(projM*v);
  float inv = 1/nv(3);
  return Vec3(nv(0)*inv, nv(1)*inv, -nv(2)*inv);
}

inline Vec3 vec3to4(const Vec4& v)
{
  float inv = 1/v(3);
  return Vec3(v(0)*inv, v(1)*inv, v(2)*inv);
}

//////////////////////////////////////////////////////////////////////////////
// Helper types

typedef std::vector<Vec3> Vec3Vector;
typedef std::vector<Vec4> Vec4Vector;

//////////////////////////////////////////////////////////////////////////////

#endif
