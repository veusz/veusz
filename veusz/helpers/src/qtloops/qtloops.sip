//    Copyright (C) 2009 Jeremy S. Sanders
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

%Module qtloops 0

%Import QtCore/QtCoremod.sip
%Import QtGui/QtGuimod.sip

%ModuleHeaderCode
#include <qtloops_helpers.h>
%End

%PostInitialisationCode
do_numpy_init_package();
%End

class QtLoops
 {

%TypeHeaderCode
#include <qtloops.h>
#include <polygonclip.h>
#include <polylineclip.h>
#include <beziers_qtwrap.h>
#include <numpyfuncs.h>
%End

public:
   QtLoops();
};

void addNumpyToPolygonF(QPolygonF&, ...);
%MethodCode
   {
   try
     {
       Tuple2Ptrs t(a1);
       addNumpyToPolygonF(*a0, t);
     }
   catch( const char *msg )
     {
       sipIsErr = 1; PyErr_SetString(PyExc_TypeError, msg);
     }
   }
%End

void addNumpyPolygonToPath(QPainterPath&, const QRectF*, ...);
%MethodCode
   {
   try
     {
       Tuple2Ptrs t(a2);
       addNumpyPolygonToPath(*a0, t, a1);
     }
   catch( const char *msg )
     {
       sipIsErr = 1; PyErr_SetString(PyExc_TypeError, msg);
     }
   }
%End


void plotPathsToPainter(QPainter&, QPainterPath&, SIP_PYOBJECT, SIP_PYOBJECT,
			SIP_PYOBJECT,
			const QRectF* clip=0,
			const QImage* colorimg=0);
%MethodCode
{
  Numpy1DObj* scaling = 0;
  
  try
    {
      // x and y coordinates
      Numpy1DObj x(a2);
      Numpy1DObj y(a3);
      
      // a4 is scaling or None
      if (a4 != Py_None) {
	scaling = new Numpy1DObj(a4);
      }
      
      plotPathsToPainter(*a0, *a1, x, y, scaling, a5, a6);
    }
  catch( const char *msg )
    {
      sipIsErr = 1; PyErr_SetString(PyExc_TypeError, msg);
    }
  
  delete scaling;
}
%End

void plotLinesToPainter(QPainter& painter,
			SIP_PYOBJECT, SIP_PYOBJECT,
			SIP_PYOBJECT, SIP_PYOBJECT,
			const QRectF* clip = 0,
			bool autoexpand = true);
%MethodCode
   {
   try
     {
       Numpy1DObj x1(a1);
       Numpy1DObj y1(a2);
       Numpy1DObj x2(a3);
       Numpy1DObj y2(a4);
       plotLinesToPainter(*a0, x1, y1, x2, y2, a5, a6);
     }
   catch( const char *msg )
     {
       sipIsErr = 1; PyErr_SetString(PyExc_TypeError, msg);
     }
   }
%End

void plotBoxesToPainter(QPainter& painter,
			SIP_PYOBJECT, SIP_PYOBJECT,
			SIP_PYOBJECT, SIP_PYOBJECT,
			const QRectF* clip = 0,
			bool autoexpand = true);
%MethodCode
   {
   try
     {
       Numpy1DObj x1(a1);
       Numpy1DObj y1(a2);
       Numpy1DObj x2(a3);
       Numpy1DObj y2(a4);
       plotBoxesToPainter(*a0, x1, y1, x2, y2, a5, a6);
     }
   catch( const char *msg )
     {
       sipIsErr = 1; PyErr_SetString(PyExc_TypeError, msg);
     }
   }
%End

QImage numpyToQImage(SIP_PYOBJECT, SIP_PYOBJECT, bool forcetrans = false);
%MethodCode
  {
   try
     {
       Numpy2DObj data(a0);
       Numpy2DIntObj colors(a1);
       QImage *img = new QImage( numpyToQImage(data, colors, a2) );
       sipRes = img;
     }
   catch( const char *msg )
     {
       sipIsErr = 1; PyErr_SetString(PyExc_TypeError, msg);
     }
  }
%End

void applyImageTransparancy(QImage& img, SIP_PYOBJECT);
%MethodCode
  {
   try
     {
       Numpy2DObj data(a1);
       applyImageTransparancy(*a0, data);
     }
   catch( const char *msg )
     {
       sipIsErr = 1; PyErr_SetString(PyExc_TypeError, msg);
     }
  }
%End

void polygonClip(const QPolygonF& inpoly,
		 const QRectF& cliprect,
		 QPolygonF& outpoly);

void plotClippedPolygon(QPainter& painter,
			QRectF rect,
			const QPolygonF& inpoly,
			bool autoexpand = true);

void plotClippedPolyline(QPainter& painter,
			 QRectF clip,
			 const QPolygonF& poly,
			 bool autoexpand = true);

QPolygonF bezier_fit_cubic_single(const QPolygonF& data, double error);

QPolygonF bezier_fit_cubic_multi(const QPolygonF& data, double error,
				 unsigned max_beziers);

SIP_PYOBJECT binData(SIP_PYOBJECT data, int binning, bool average);
%MethodCode
   try
     {
       Numpy1DObj d(a0);
       double* data;
       int numelem;
       binData(d, a1, a2, &numelem, &data);
       sipRes = doubleArrayToNumpy(data, numelem);
       delete[] data;
     }
   catch( const char *msg )
     {
       sipIsErr = 1; PyErr_SetString(PyExc_TypeError, msg);
     }
%End

SIP_PYOBJECT rollingAverage(SIP_PYOBJECT data, SIP_PYOBJECT weights,
			    int width);
%MethodCode
{
  Numpy1DObj* weightarray = 0;
  try
    {
      Numpy1DObj d(a0);
      if( a1 != Py_None ) {
	weightarray = new Numpy1DObj(a1);
      }
      double* data;
      int numelem;
      rollingAverage(d, weightarray, a2, &numelem, &data);
      sipRes = doubleArrayToNumpy(data, numelem);
      delete[] data;
    }
  catch( const char *msg )
    {
      sipIsErr = 1; PyErr_SetString(PyExc_TypeError, msg);
    }
  delete weightarray;
}
%End