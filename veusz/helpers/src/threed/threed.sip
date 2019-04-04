// -*- mode: C++; -*-

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

%Module(name=threed)

%Import(name=QtCore/QtCoremod.sip)
%Import(name=QtGui/QtGuimod.sip)

%ModuleHeaderCode
#include <numpy_helpers.h>
%End

%PostInitialisationCode
doNumpyInitPackage();
%End

struct Vec4
{
%TypeHeaderCode
#include <mmaths.h>
%End

  Vec4();
  Vec4(double, double, double, double);

  void set(unsigned, double);
%MethodCode
    if(a0 > 3)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=3");
      }
    else
      sipCpp->operator()(a0) = a1;
%End
  double get(unsigned) const;
%MethodCode
    if(a0 > 3)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=3");
      }
    else
      sipRes = sipCpp->operator()(a0);
%End

  void operator*=(double);
  Vec4 operator+(const Vec4&) const;
  Vec4 operator-(const Vec4&) const;
  Vec4 operator*(double) const;
  bool operator==(const Vec4&) const;
  bool operator!=(const Vec4&) const;
  double rad2() const;
  double rad() const;
  void normalise();
};

struct Vec3
{
%TypeHeaderCode
#include <mmaths.h>
%End

  Vec3();
  Vec3(double, double, double);

  void set(unsigned, double);
%MethodCode
    if(a0 > 2)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=2");
      }
    else
      sipCpp->operator()(a0) = a1;
%End
  double get(unsigned) const;
%MethodCode
    if(a0 > 2)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=2");
      }
    else
      sipRes = sipCpp->operator()(a0);
%End

  void operator*=(double);
  Vec3 operator+(const Vec3&) const;
  Vec3 operator-(const Vec3&) const;
  Vec3 operator*(double) const;
  bool operator==(const Vec3&) const;
  bool operator!=(const Vec3&) const;
  double rad2() const;
  double rad() const;
  void normalise();
};

struct Vec2
{
%TypeHeaderCode
#include <mmaths.h>
%End

  Vec2();
  Vec2(double, double);

  void set(unsigned, double);
%MethodCode
    if(a0 > 1)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=1");
      }
    else
      sipCpp->operator()(a0) = a1;
%End
  double get(unsigned) const;
%MethodCode
    if(a0 > 1)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=1");
      }
    else
      sipRes = sipCpp->operator()(a0);
%End

  void operator*=(double);
  Vec2 operator+(const Vec2&) const;
  Vec2 operator-(const Vec2&) const;
  Vec2 operator*(double) const;
  bool operator==(const Vec2&) const;
  bool operator!=(const Vec2&) const;
  double rad2() const;
  double rad() const;
  void normalise();
};

struct Mat4
{
%TypeHeaderCode
#include <mmaths.h>
%End

  Mat4();
  void set(unsigned, unsigned, double);
%MethodCode
    if(a0 > 3 || a1 > 3)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=3");
      }
    else
      sipCpp->operator()(a0, a1) = a2;
%End
  double get(unsigned, unsigned) const;
%MethodCode
    if(a0 > 3 || a1 > 3)
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
Mat4 rotateM4(double angle, Vec3 vec);
Mat4 rotate3M4(double ax, double ay, double az);
Mat4 translationM4(Vec3 vec);
Mat4 scaleM4(Vec3 scalevec);

struct Mat3
{
%TypeHeaderCode
#include <mmaths.h>
%End

  Mat3();
  void set(unsigned, unsigned, double);
%MethodCode
    if(a0 > 2 || a1 > 2)
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index should be 0<=i<=2");
      }
    else
      sipCpp->operator()(a0, a1) = a2;
%End
  double get(unsigned, unsigned) const;
%MethodCode
    if(a0 > 2 || a1 > 2)
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


// this is actually std::vector<double>
class ValVector
{
%TypeHeaderCode
#include <mmaths.h>
%End
public:
  ValVector();
  ValVector(SIP_PYOBJECT obj);
%MethodCode
  try
    {
      sipCpp = new ValVector(numpyToValVector(a0));
    }
   catch(const char *msg)
     {
       sipIsErr = 1; PyErr_SetString(PyExc_TypeError, msg);
     }
%End

  void push_back(double d);
  double operator[](unsigned i) const;
%MethodCode
    if(a0 >= sipCpp->size())
      {
	sipIsErr = 1;
	PyErr_SetString(PyExc_ValueError, "Index out of range");
      }
    else
      sipRes = sipCpp->operator[](a0);
%End
  unsigned size() const;
  bool empty() const;
};

Vec3 calcProjVec(const Mat4& projM, const Vec3& v);
Vec3 calcProjVec(const Mat4& projM, const Vec4& v);
Vec2 projVecToScreen(const Mat3& screenM, const Vec3& vec);

/////////////////////////////////////////////////
// Properties

struct SurfaceProp /NoDefaultCtors/
{
%TypeHeaderCode
#include <properties.h>
%End

  SurfaceProp(double r=0.5, double g=0.5, double b=0.5,
	      double refl=0.5, double trans=0,
	      bool hide=0) /KeywordArgs="All"/;
  void setRGBs(const QImage& img);

  double r;
  double g;
  double b;
  double refl;
  double trans;
  bool hide;
};

struct LineProp /NoDefaultCtors/
{
%TypeHeaderCode
#include <properties.h>
%End

  LineProp(double r=0, double g=0, double b=0,
	   double trans=0,
           double refl=0,
	   double width=1,
           bool hide=0,
           Qt::PenStyle style=Qt::SolidLine) /KeywordArgs="All"/;
  void setRGBs(const QImage& img);
  void setDashPattern(const ValVector& vec);

  double r;
  double g;
  double b;
  double trans;
  double refl;
  double width;
  bool hide;
};


///////////////////////////////////////////////////////////////
// Objects

class Object /NoDefaultCtors/
{
%TypeHeaderCode
#include <objects.h>
%End
  public:
   virtual ~Object();
   virtual void assignWidgetId(unsigned long id);
   unsigned long widgetid;
};

class Triangle : public Object /NoDefaultCtors/
{
%TypeHeaderCode
#include <objects.h>
%End
 public:
  Triangle(const Vec3&, const Vec3&, const Vec3&,
	   const SurfaceProp* surfaceprop /Transfer/);
};

class PolyLine : public Object /NoDefaultCtors/
{
%TypeHeaderCode
#include <objects.h>
%End
 public:
  PolyLine(const LineProp* prop /Transfer/);
  void addPoint(const Vec3& v);
  void addPoints(const ValVector& x, const ValVector& y, const ValVector& z);
};

class LineSegments : public Object
{
%TypeHeaderCode
#include <objects.h>
%End
 public:
  LineSegments(const ValVector& x1, const ValVector& y1, const ValVector& z1,
               const ValVector& x2, const ValVector& y2, const ValVector& z2,
               const LineProp* prop /Transfer/);
  LineSegments(const ValVector& pts1, const ValVector& pts2,
               const LineProp* prop /Transfer/);
};

class Mesh : public Object /NoDefaultCtors/
{
%TypeHeaderCode
#include <objects.h>
%End
 public:
  enum Direction {X_DIRN, Y_DIRN, Z_DIRN};

  Mesh(const ValVector& pos1, const ValVector& pos2,
       const ValVector& heights,
       Direction dir,
       const LineProp* lprop /Transfer/,
       const SurfaceProp* sprop /Transfer/,
       bool hidehorzline=0, bool hidevertline=0);
};

class DataMesh : public Object /NoDefaultCtors/
{
%TypeHeaderCode
#include <objects.h>
%End
 public:
  DataMesh(const ValVector& edges1, const ValVector& edges2,
           const ValVector& vals,
           unsigned idxval, unsigned idxedge1, unsigned idxedge2,
           bool highres,
           const LineProp* lprop /Transfer/,
           const SurfaceProp* sprop /Transfer/,
           bool hidehorzline=0, bool hidevertline=0);
};

class MultiCuboid : public Object
{
%TypeHeaderCode
#include <objects.h>
%End
public:
  MultiCuboid(const ValVector& _xmin, const ValVector& _xmax,
              const ValVector& _ymin, const ValVector& _ymax,
              const ValVector& _zmin, const ValVector& _zmax,
              const LineProp* lprop /Transfer/,
              const SurfaceProp* sprop /Transfer/);
};

class Points : public Object /NoDefaultCtors/
{
%TypeHeaderCode
#include <objects.h>
%End
 public:
  Points(const ValVector& px, const ValVector& py, const ValVector& pz,
         QPainterPath pp,
         const LineProp* pointedge /Transfer/,
         const SurfaceProp* pointfill /Transfer/);

  void setSizes(const ValVector& _sizes);

  bool scaleline;
  bool scalepersp;
};

// a "text" class which calls back draw() when drawing is requested
class Text : public Object
{
%TypeHeaderCode
#include <objects.h>
%End
public:
  // pos1 and pos2 contain a list of x,y,z values
  Text(const ValVector& _pos1, const ValVector& _pos2);

  virtual void draw(QPainter* painter,
                    QPointF pt1, QPointF pt2, QPointF pt3,
                    unsigned index, double scale, double linescale);
};

class ObjectContainer : public Object
{
%TypeHeaderCode
#include <objects.h>
%End
 public:
  ObjectContainer();
  void addObject(Object* obj /Transfer/);
  void assignWidgetId(unsigned long id);

  Mat4 objM;
};

class TriangleFacing : public Triangle
{
%TypeHeaderCode
#include <objects.h>
%End
public:
  TriangleFacing(const Vec3& a, const Vec3& b, const Vec3& c,
                 const SurfaceProp* prop /Transfer/);
};

class FacingContainer : public ObjectContainer
{
%TypeHeaderCode
#include <objects.h>
%End
public:
  FacingContainer(Vec3 _norm);
  Vec3 norm;
};

class AxisLabels : public Object
{
%TypeHeaderCode
#include <objects.h>
%End
public:
  AxisLabels(const Vec3& _box1, const Vec3& _box2,
             const ValVector& _tickfracs,
             double _labelfrac);

  void addAxisChoice(const Vec3& start, const Vec3& end);

  virtual void drawLabel(QPainter* painter, int index,
                         QPointF pt,
                         QPointF ax1, QPointF ax2,
                         double axangle);
};


class ClipContainer : public ObjectContainer
{
%TypeHeaderCode
#include <clipcontainer.h>
%End
 public:
  ClipContainer(Vec3 minpt, Vec3 maxpt);
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
  void setPerspective(double fov_degrees=45,
		      double znear=0.1, double zfar=100.);

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
  enum RenderMode {RENDER_PAINTERS, RENDER_BSP};

 public:
  Scene(RenderMode mode);
  void addLight(Vec3 posn, QColor col, double intensity);
  void render(Object* root,
              QPainter* painter, const Camera& cam,
	      double x1, double y1, double x2, double y2, double scale);
  unsigned long idPixel(Object* root,
                        QPainter* painter, const Camera& cam,
                        double x1, double y1, double x2, double y2,
                        double scale,
                        double scaling, int x, int y);

 public:
  Mat3 screenM;
};
