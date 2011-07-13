//    Copyright (C) 2011 Jeremy S. Sanders
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

#include <math.h>

#include <QPainter>
#include <QImage>
#include <QRectF>
#include <QLineF>
#include <QVector>

namespace {

  class Element {
  public:
    virtual ~Element() {};
    virtual void paint(QPainter& painter) = 0;
  };

  class EllipseElement : public Element {
  public:
    EllipseElement(const QRectF &rect) : _ellipse(rect) {}

  private:
    QRectF _ellipse;
  };

  class ImageElement : public Element {
  public:
    ImageElement(const QRectF& rect, const QImage& image,
		 const QRectF& sr, Qt::ImageConversionFlags flags)
      : _image(image), _rect(rect), _sr(sr), _flags(flags)
    {}

  private:
    QImage _image;
    QRectF _rect, _sr;
    Qt::ImageConversionFlags _flags;
  };

  class LineElement : public Element {
  public:
    LineElement(const QLineF *lines, int linecount)
    {
      for(int i = 0; i < linecount; i++)
	_lines.push_back(lines[i]);
    }

  private:
    QVector<QLineF> lines;
  }
}
