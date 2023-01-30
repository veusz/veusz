//    Copyright (C) 2012 Jeremy S. Sanders
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

%Module(name=qtmml)

%Import(name=QtCore/QtCoremod.sip)
%Import(name=QtGui/QtGuimod.sip)
%Import(name=QtWidgets/QtWidgetsmod.sip)

%ModuleHeaderCode
 // we have to do this twice, so define as a template
namespace
 {
   template<class T>
   void callSetContent(T* obj, const QString &txt, int *sipIsErr)
   {
     QString errormsg;
     int errorline, errorcol;
     bool retn;
     
     Py_BEGIN_ALLOW_THREADS;
     retn = obj->setContent(txt, &errormsg, &errorline, &errorcol);
     Py_END_ALLOW_THREADS;

     if( ! retn )
       {
	 // if fails, create an exception 
	 QByteArray extxt( QString("Error on line %1, column %2: \"%3\"").
			   arg(errorline).arg(errorcol).arg(errormsg).
			   toUtf8() );

	 PyObject *pystr = PyUnicode_DecodeUTF8(extxt.data(), extxt.size(),
						"ignore");
	 if( pystr != 0 )
	   {
	     PyErr_SetObject(PyExc_ValueError, pystr);
	     Py_DECREF(pystr);
	     *sipIsErr = 1;
	   }
       }
   }
 }
%End

class QtMmlWidget : QFrame
{
%TypeHeaderCode
#include <QtMmlWidget>
%End

public:
  enum MmlFont { NormalFont, FrakturFont, SansSerifFont, ScriptFont,
		 MonospaceFont, DoublestruckFont };

  QtMmlWidget(QWidget *parent = 0);
  ~QtMmlWidget();

  QString fontName(MmlFont type) const;
  void setFontName(MmlFont type, const QString &name);
  int baseFontPointSize() const;
  void setBaseFontPointSize(int size);

 //  bool setContent(const QString &text, QString *errorMsg = 0,
 //		  int *errorLine = 0, int *errorColumn = 0);
  void setContent(const QString &text);
%MethodCode
  callSetContent(sipCpp, *a0, &sipIsErr);
%End

  void dump() const;
  virtual QSize sizeHint() const;

  void setDrawFrames(bool b);
  bool drawFrames() const;

  void clear();

protected:
  virtual void paintEvent(QPaintEvent *e);
};

class QtMmlDocument
{
%TypeHeaderCode
#include <QtMmlDocument>
%End
public:
  QtMmlDocument();
  ~QtMmlDocument();
  void clear();
  
  //    bool setContent(QString text, QString *errorMsg = 0,
  //                    int *errorLine = 0, int *errorColumn = 0);
  void setContent(const QString &text);
%MethodCode
  callSetContent(sipCpp, *a0, &sipIsErr);
%End

  void paint(QPainter *p, const QPoint &pos) const;
  QSize size() const;
  
  QString fontName(QtMmlWidget::MmlFont type) const;
  void setFontName(QtMmlWidget::MmlFont type, const QString &name);

  int baseFontPointSize() const;
  void setBaseFontPointSize(int size);
};
