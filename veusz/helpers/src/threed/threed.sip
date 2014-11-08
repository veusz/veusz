//    Copyright (C) 2014 Jeremy S. Sanders
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

%Module(name=threed, version=0)

%Import(name=QtCore/QtCoremod.sip)
%Import(name=QtGui/QtGuimod.sip)

struct SurfaceProp
{
%TypeHeaderCode
#include <objects.h>
%End

  SurfaceProp(float r=0.5, float g=0.5, float b=0.5,
	      float specular=0.5, float diffuse=0.5, float trans=0,
	      bool hide=0);
  float r;
  float g;
  float b;
  float specular;
  float diffuse;
  float trans;
  bool hide;
};

struct LineProp
{
%TypeHeaderCode
#include <objects.h>
%End

  LineProp(float r=0, float g=0, float b=0,
	   float specular=0.5, float diffuse=0.5, float trans=0,
	   float width=1, bool hide=0);
  float r;
  float g;
  float b;
  float specular;
  float diffuse;
  float trans;
  float width;
  bool hide;
};

struct Vec4
{
%TypeHeaderCode
#include <mmaths.h>
%End

  Vec4();
  Vec4(float, float, float, float);

  void set(unsigned, float);
%MethodCode
    if(a0 < 0 || a0 > 3)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=3");
      }
    else
      sipCpp->operator()(a0) = a1;
%End
  float get(unsigned) const;

  void operator*=(float);
  Vec4 operator+(const Vec4&) const;
  Vec4 operator-(const Vec4&) const;
  Vec4 operator*(float) const;
  bool operator==(const Vec4&) const;
  bool operator!=(const Vec4&) const;
  float rad2() const;
  float rad() const;
  void normalise();
};

struct Vec3
{
%TypeHeaderCode
#include <mmaths.h>
%End

  Vec3();
  Vec3(float, float, float);

  void set(unsigned, float);
  float get(unsigned) const;

  void operator*=(float);
  Vec3 operator+(const Vec3&) const;
  Vec3 operator-(const Vec3&) const;
  Vec3 operator*(float) const;
  bool operator==(const Vec3&) const;
  bool operator!=(const Vec3&) const;
  float rad2() const;
  float rad() const;
  void normalise();
};

struct Mat4
{
%TypeHeaderCode
#include <mmaths.h>
%End

  Mat4();
  void set(unsigned, unsigned, float);
  float get(unsigned, unsigned) const;
  Mat4 operator*(const Mat4& o) const;
  Vec4 operator*(const Vec4& v) const;
  Mat4 transpose() const;
};
Vec4 operator*(const Vec4& v, const Mat4& o);
Mat4 identityM4();
Mat4 rotateM4(float angle, Vec3 vec);
Mat4 translationM4(Vec3 vec);

struct Mat3
{
%TypeHeaderCode
#include <mmaths.h>
%End

  Mat3();
  void set(unsigned, unsigned, float);
  float get(unsigned, unsigned) const;
  Mat3 operator*(const Mat3& o) const;
  Vec3 operator*(const Vec3& v) const;
  Mat3 transpose() const;
};
Vec3 operator*(const Vec3& v, const Mat3& o);
Mat3 identityM3();

class Object
{
%TypeHeaderCode
#include <objects.h>
%End
};

class Triangle : public Object
{
%TypeHeaderCode
#include <objects.h>
%End
 public:
};
