// -*- mode: C++; -*-

// Copyright (C) 2010 Jeremy Sanders

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
// 02110-1301, USA.
////////////////////////////////////////////////////////////////////

#ifndef POLYLINECLIP_HH
#define POLYLINECLIP_HH

#include <QRectF>
#include <QPainter>
#include <QPolygonF>
#include <QVector>
#include <QSizeF>

// clip a line made up of the points given, returning true
// if is in region or false if not
bool clipLine(const QRectF& clip, QPointF& pt1, QPointF& pt2);

// plot a polyline poly on the painter, clipping by the rectangle given
// if autoexpand is true, then the rectangle is expanded by the line width
void plotClippedPolyline(QPainter& painter,
                         QRectF clip,
                         const QPolygonF& poly,
                         bool autoexpand = true);


// clip polyline to rectangle
// return list of lines to plot
QVector<QPolygonF> clipPolyline(QRectF clip, const QPolygonF& poly);

// Do the polygons intersect?
bool doPolygonsIntersect(const QPolygonF& a, const QPolygonF& b);

// class for describing a rectangle with a rotation angle
struct RotatedRectangle
{
  // a lot of boilerplate so it can go in QVector
  RotatedRectangle()
    : cx(0), cy(0), xw(0), yw(0), angle(0)
  {}
  RotatedRectangle(double _cx, double _cy,
                   double _xw, double _yw, double _angle)
    : cx(_cx), cy(_cy), xw(_xw), yw(_yw), angle(_angle)
  {}
  RotatedRectangle(const RotatedRectangle& o)
    : cx(o.cx), cy(o.cy), xw(o.xw), yw(o.yw), angle(o.angle)
  {}
  RotatedRectangle &operator=(const RotatedRectangle& o)
  {
    cx = o.cx; cy = o.cy; xw=o.xw; yw=o.yw; angle=o.angle;
    return *this;
  }
  bool isValid() const { return xw > 0 && yw > 0; }
  void rotate(double dtheta) { angle += dtheta; }
  void rotateAboutOrigin(double dtheta);
  void translate(double dx, double dy) { cx+=dx; cy+=dy; }

  QPolygonF makePolygon() const;

  double cx, cy, xw, yw, angle;
};

// for labelling of sets of contour lines
class LineLabeller
{
public:
  LineLabeller(QRectF cliprect, bool rotatelabels);
  virtual ~LineLabeller();

  // override this to receive the label to draw
  virtual void drawAt(int idx, RotatedRectangle r);

  void addLine(const QPolygonF& poly, QSizeF textsize);

  void process();

  int getNumPolySets() const { return _polys.size(); };
  QVector<QPolygonF> getPolySet(int i) const;

private:
  RotatedRectangle findLinePosition(const QPolygonF& poly, double frac,
                                    QSizeF size);

private:
  QRectF _cliprect;
  bool _rotatelabels;

  QVector< QVector<QPolygonF> > _polys;
  QVector<QSizeF> _textsizes;
};

class RectangleOverlapTester
{
public:
  RectangleOverlapTester();
  bool willOverlap(const RotatedRectangle& rect) const;
  void addRect(const RotatedRectangle& rect) { _rects.append(rect); };
  void reset() { _rects.clear(); };

  // debug by drawing all the rectangles
  void debug(QPainter& painter) const;

private:
  QVector<RotatedRectangle> _rects;
};

#endif
