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
%MethodCode
    if(a0 < 0 || a0 > 3)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=3");
      }
    else
      sipRes = sipCpp->operator()(a0);
%End

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
%MethodCode
    if(a0 < 0 || a0 > 2)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=2");
      }
    else
      sipCpp->operator()(a0) = a1;
%End
  float get(unsigned) const;
%MethodCode
    if(a0 < 0 || a0 > 2)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=2");
      }
    else
      sipRes = sipCpp->operator()(a0);
%End

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
%MethodCode
    if(a0 < 0 || a0 > 3 || a1 < 0 || a1 > 3)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=3");
      }
    else
      sipCpp->operator()(a0, a1) = a2;
%End
  float get(unsigned, unsigned) const;
%MethodCode
    if(a0 < 0 || a0 > 3 || a1 < 0 || a1 > 3)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=3");
      }
    else
      sipRes = sipCpp->operator()(a0, a1);
%End

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
%MethodCode
    if(a0 < 0 || a0 > 2 || a1 < 0 || a1 > 2)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=2");
      }
    else
      sipCpp->operator()(a0, a1) = a2;
%End
  float get(unsigned, unsigned) const;
%MethodCode
    if(a0 < 0 || a0 > 2 || a1 < 0 || a1 > 2)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=2");
      }
    else
      sipRes = sipCpp->operator()(a0, a1);
%End
  Mat3 operator*(const Mat3& o) const;
  Vec3 operator*(const Vec3& v) const;
  Mat3 transpose() const;
};
Vec3 operator*(const Vec3& v, const Mat3& o);
Mat3 identityM3();

/////////////////////////////////////////////////
// Objects

struct SurfaceProp
{
%TypeHeaderCode
#include <objects.h>
%End

  SurfaceProp(float r=0.5, float g=0.5, float b=0.5,
	      float specular=0.5, float diffuse=0.5, float trans=0,
	      bool hide=0) /KeywordArgs="All"/;
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
	   float width=1, bool hide=0) /KeywordArgs="All"/;
  float r;
  float g;
  float b;
  float specular;
  float diffuse;
  float trans;
  float width;
  bool hide;
};

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
  Triangle();
  Triangle(const Vec4&, const Vec4&, const Vec4&,
	   const SurfaceProp* surfaceprop=0);
  const SurfaceProp* surfaceprop;
};

class PolyLine : public Object
{
%TypeHeaderCode
#include <objects.h>
%End
 public:
  PolyLine(const LineProp* prop=0);
  void addPoint(const Vec4& v);
  const LineProp* lineprop;
};

class ObjectContainer : public Object
{
%TypeHeaderCode
#include <objects.h>
%End
 public:
  ObjectContainer();
  void addObject(Object* obj /Transfer/);

  Mat4 objM;
};

////////////////////////////////////////////////////////////////
// Camera

class Camera
{
%TypeHeaderCode
#include <camera.h>
%End

 public:
  Camera();
  void setPointing(const Vec3 &eye, const Vec3 &target, const Vec3 &up);
  void setPerspective(float fovy_degrees=45, float aspect=1,
		      float znear=0.1, float zfar=100.);

 public:
  Mat4 viewM;   // view matrix
  Mat4 perspM;  // perspective matrix
  Mat4 combM;   // combined matrix
  Vec3 eye;     // location of eye
};

////////////////////////////////////////////////////////////////
// Scene

class Scene
{
%TypeHeaderCode
#include <scene.h>
%End

 public:
  Scene();
  void render(QPainter* painter, const Camera& cam,
	      float x1, float y1, float x2, float y2);

 public:
  ObjectContainer root;
};
