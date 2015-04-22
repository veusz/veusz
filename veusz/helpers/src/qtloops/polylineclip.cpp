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

#include <cmath>
#include <limits>
#include <algorithm>

#include <QPointF>
#include <QPen>

#include <polylineclip.h>

using std::fabs;

// Cohen-Sutherland line clipping algorithm

// codes used classify endpoints are combinations of these
#define LEFT 1
#define RIGHT 2
#define TOP 4
#define BOTTOM 8

namespace
{
  // compute intersection with horizontal line
  inline QPointF horzIntersection(qreal yval, const QPointF& pt1,
                                  const QPointF& pt2)
  {
    if( pt1.y() == pt2.y() )
      // line is vertical
      return QPointF( pt1.x(), yval );
    else
      return QPointF( pt1.x() + (yval-pt1.y()) *
                      (pt2.x()-pt1.x()) / (pt2.y()-pt1.y()),
                      yval );
  }

  // compute intersection with vertical line
  inline QPointF vertIntersection(qreal xval, const QPointF& pt1,
                                  const QPointF& pt2)
  {
    if( pt1.x() == pt2.x() )
      // line is horizontal
      return QPointF( xval, pt1.y() );
    else
      return QPointF( xval,
                      pt1.y() + (xval-pt1.x()) *
                      (pt2.y()-pt1.y()) / (pt2.x()-pt1.x()) );
  }

  // is difference between points very small?
  inline bool smallDelta(const QPointF& pt1, const QPointF& pt2)
  {
    return fabs(pt1.x() - pt2.x()) < 0.01 &&
      fabs(pt1.y()- pt2.y()) < 0.01;
  }

  template<class T> T sqr(T v)
  {
    return v*v;
  }

  // private class for helping clip
  class _Clipper
  {
  public:
    _Clipper(const QRectF& cliprect);
    unsigned computeCode(const QPointF& pt) const;
    void fixPt(QPointF& pt) const;
    bool clipLine(QPointF& pt1, QPointF& pt2) const;

  private:
    QRectF clip;
  };


  // This class is use to separate out the clipping of polylines
  // overrite emitPolyline to handle the clipped part of the original
  // polyline
  class _PolyClipper
  {
  public:
    _PolyClipper(QRectF clip)
      : _clipper(clip)
    {}
    virtual ~_PolyClipper() {};

    // override this to draw the output polylines
    virtual void emitPolyline(const QPolygonF& poly) = 0;

    // do clipping on the polyline
    void clipPolyline(const QPolygonF& poly);

  private:
    _Clipper _clipper;
  };

}

////////////////////////////////////////////////////////////////////////

_Clipper::_Clipper(const QRectF& cliprect)
  : clip(cliprect)
{
}

// compute the Cohen-Sutherland code
unsigned _Clipper::computeCode(const QPointF& pt) const
{
  unsigned code = 0;
  if (pt.x() < clip.left())
    code |= LEFT;
  else if (pt.x() > clip.right())
    code |= RIGHT;
  if (pt.y() < clip.top())
    code |= TOP;
  else if (pt.y() > clip.bottom())
    code |= BOTTOM;
  return code;
}

// get consistent clipping on different platforms by making line
// edges meet clipping box if close
void _Clipper::fixPt(QPointF& pt) const
{
  if( fabs(pt.x() - clip.left()) < 1e-4 )
    pt.setX(clip.left());
  if( fabs(pt.x() - clip.right()) < 1e-4 )
    pt.setX(clip.right());
  if( fabs(pt.y() - clip.top()) < 1e-4 )
    pt.setY(clip.top());
  if( fabs(pt.y() - clip.bottom()) < 1e-4 )
    pt.setY(clip.bottom());
}

// modifies points, returning true if okay to accept
bool _Clipper::clipLine(QPointF& pt1, QPointF& pt2) const
{
  // fixup ends to meet clip box if close
  fixPt(pt1);
  fixPt(pt2);

  unsigned code1 = computeCode(pt1);
  unsigned code2 = computeCode(pt2);

  // repeat until points are at edge of box
  // bail out if we repeat too many times (avoid numerical issues)
  for(unsigned count = 0 ; count < 16 ; count++ )
    {
      if( (code1 | code2) == 0 )
        {
          // no more clipping required - inside
          return true;
        }
      else if( (code1 & code2) != 0 )
        {
          // line should not be drawn - outside
          return false;
        }
      else
        {
          // compute intersection

          // which point to compute for?
          const unsigned code = (code1 != 0) ? code1 : code2;

          // get intersection new point and new code for line
          QPointF pt;
          if( code & LEFT )
            pt = vertIntersection(clip.left(), pt1, pt2);
          else if( code & RIGHT )
            pt = vertIntersection(clip.right(), pt1, pt2);
          else if( code & TOP )
            pt = horzIntersection(clip.top(), pt1, pt2);
          else if ( code & BOTTOM )
            pt = horzIntersection(clip.bottom(), pt1, pt2);

          // update point as intersection
          if( code == code1 )
            {
              // changed first point
              pt1 = pt;
              code1 = computeCode(pt1);
            }
          else
            {
              // changed second point
              pt2 = pt;
              code2 = computeCode(pt2);
            }
        }
    } // loop
  return false;
}


bool clipLine(const QRectF& clip, QPointF& pt1, QPointF& pt2)
{
  _Clipper clipper(clip);
  return clipper.clipLine(pt1, pt2);
}

//////////////////////////////////////////////////////////////////////

void _PolyClipper::clipPolyline(const QPolygonF& poly)
{
  // exit if fewer than 2 points in polygon
  if ( poly.size() < 2 )
    return;

  // output goes here
  QPolygonF pout;

  QPolygonF::const_iterator polyiter = poly.begin();
  QPointF lastpt = *polyiter;
  polyiter++;

  for( ; polyiter != poly.end(); ++polyiter )
    {
      QPointF p1 = lastpt;
      QPointF p2 = *polyiter;

      bool plotline = _clipper.clipLine(p1, p2);
      if( plotline )
        {
          if( pout.isEmpty() )
            {
              // add first line
              pout << p1;
              if( ! smallDelta(p1, p2) )
                pout << p2;
            }
          else
            {
              if( p1 == pout.last() )
                {
                  if( ! smallDelta(p1, p2) )
                    // extend polyline
                    pout << p2;
                }
              else
                {
                  // paint existing line
                  if( pout.size() >= 2 )
                    emitPolyline(pout);

                  // start new line
                  pout.clear();
                  pout << p1;
                  if( ! smallDelta(p1, p2) )
                    pout << p2;
                }
            }
        }
      else
        {
          // line isn't in region, so ignore results from clip function

          // paint existing line
          if( pout.size() >= 2 )
            emitPolyline(pout);

          // cleanup
          pout.clear();
        }


      lastpt = *polyiter;
    }

  if( pout.size() >= 2 )
    emitPolyline(pout);
}

// class used for drawing clipped polylines

class PlotDrawCallback : public _PolyClipper
{
 public:
  PlotDrawCallback(QRectF clip, QPainter& painter)
    : _PolyClipper(clip), 
      _painter(painter)
  {}

  void emitPolyline(const QPolygonF& poly)
  {
    _painter.drawPolyline(poly);
  }

 private:
  QPainter& _painter;
};

// take polyline and paint to painter, clipping
void plotClippedPolyline(QPainter& painter,
                         QRectF clip,
                         const QPolygonF& poly,
                         bool autoexpand)
{
  // if autoexpand, expand rectangle by line width
  if ( autoexpand )
    {
      const qreal lw = painter.pen().widthF();
      clip.adjust(-lw, -lw, lw, lw);
    }

  PlotDrawCallback pcb(clip, painter);
  pcb.clipPolyline(poly);
}

//////////////////////////////////////////////////////
// clip polyline and add polyines clipped to a vector

class PolyAddCallback : public _PolyClipper
{
 public:
  PolyAddCallback(QRectF clip)
    : _PolyClipper(clip) {}

  void emitPolyline(const QPolygonF& poly)
  {
    polys.push_back(poly);
  }

public:
  QVector<QPolygonF> polys;
};

QVector<QPolygonF> clipPolyline(QRectF clip, const QPolygonF& poly)
{
  PolyAddCallback pcb(clip);
  pcb.clipPolyline(poly);
  return pcb.polys;
}

//////////////////////////////////////////////////////

typedef QVector<QPolygonF> PolyVector;

// clip polygon, adding clipped parts to output vector of polygons
class _LineLabClipper : public _PolyClipper
{
public:
  _LineLabClipper(QRectF cliprect, PolyVector& polyvec)
    : _PolyClipper(cliprect),
      _polyvec(polyvec)
  {
  }

  void emitPolyline(const QPolygonF& poly)
  {
    _polyvec.append(poly);
  }

private:
  PolyVector& _polyvec;
};

///////////////////////////////////////////////////////

// Check whether polygons intersect
// http://stackoverflow.com/questions/10962379/how-to-check-intersection-between-2-rotated-rectangles

bool doPolygonsIntersect(const QPolygonF& a, const QPolygonF& b)
{
  for(int polyi = 0; polyi < 2; ++polyi)
    {
      const QPolygonF& polygon = polyi == 0 ? a : b;

      for(int i1 = 0; i1 < polygon.size(); ++i1)
        {
          const int i2 = (i1 + 1) % polygon.size();

          const double normalx = polygon[i2].y() - polygon[i1].y();
          const double normaly = polygon[i2].x() - polygon[i1].x();

          double minA = std::numeric_limits<double>::max();
          double maxA = std::numeric_limits<double>::min();
          for(int ai = 0; ai < a.size(); ++ai)
            {
              const double projected = normalx * a[ai].x() +
                normaly * a[ai].y();
              if( projected < minA ) minA = projected;
              if( projected > maxA ) maxA = projected;
            }

          double minB = std::numeric_limits<double>::max();
          double maxB = std::numeric_limits<double>::min();
          for(int bi = 0; bi < b.size(); ++bi)
            {
              const double projected = normalx * b[bi].x() +
                normaly * b[bi].y();
              if( projected < minB ) minB = projected;
              if( projected > maxB ) maxB = projected;
            }

          if( maxA < minB || maxB < minA )
            return false;
        }
    }

  return true;
}


///////////////////////////////////////////////////////

QPolygonF RotatedRectangle::makePolygon() const
{
  QPolygonF poly;
  const double c = std::cos(angle);
  const double s = std::sin(angle);

  poly.append( QPointF( (-xw/2)*c - (-yw/2)*s + cx,
                        (-xw/2)*s + (-yw/2)*c + cy ) );
  poly.append( QPointF( (-xw/2)*c - ( yw/2)*s + cx,
                        (-xw/2)*s + ( yw/2)*c + cy ) );
  poly.append( QPointF( ( xw/2)*c - ( yw/2)*s + cx,
                        ( xw/2)*s + ( yw/2)*c + cy ) );
  poly.append( QPointF( ( xw/2)*c - (-yw/2)*s + cx,
                        ( xw/2)*s + (-yw/2)*c + cy ) );
  return poly;
}

RectangleOverlapTester::RectangleOverlapTester()
{
}

bool RectangleOverlapTester::willOverlap(const RotatedRectangle& rect)
{
  const QPolygonF thispoly(rect.makePolygon());

  for(int i = 0; i < _rects.size(); ++i)
    {
      if( doPolygonsIntersect(thispoly, _rects.at(i).makePolygon()) )
        return true;
    }

  return false;
}

///////////////////////////////////////////////////////

LineLabeller::LineLabeller(QRectF cliprect, bool rotatelabels)
  : _cliprect(cliprect),
    _rotatelabels(rotatelabels)
{
}

LineLabeller::~LineLabeller()
{
}

void LineLabeller::drawAt(int idx, RotatedRectangle r)
{
}

void LineLabeller::addLine(const QPolygonF& poly, QSizeF textsize)
{
  _polys.append( PolyVector() );
  _textsizes.append(textsize);
  _LineLabClipper clipper(_cliprect, _polys.last());
  clipper.clipPolyline(poly);
}

// returns RotatedRectangle with zero size if error
RotatedRectangle LineLabeller::findLinePosition(const QPolygonF& poly,
                                                double frac, QSizeF size)
{
  double totlength = 0;
  for(int i = 1; i < poly.size(); ++i)
    {
      totlength += std::sqrt( sqr(poly[i-1].x()-poly[i].x()) +
                              sqr(poly[i-1].y()-poly[i].y()) );
    }

  // don't label lines which are too short
  if( totlength/2 < std::max(size.width(), size.height()) )
    return RotatedRectangle();

  // now repeat and stop when we've reached the right length
  double length = 0;
  for(int i = 1; i < poly.size(); ++i)
    {
      const double seglength = std::sqrt( sqr(poly[i-1].x()-poly[i].x()) +
                                          sqr(poly[i-1].y()-poly[i].y()) );
      if(length + seglength >= totlength*frac)
        {
          // interpolate along edge
          const double fseg = (totlength*frac - length) / seglength;
          const double xp = poly[i-1].x()*(1-fseg) + poly[i].x()*fseg;
          const double yp = poly[i-1].y()*(1-fseg) + poly[i].y()*fseg;

          const double angle = _rotatelabels ?
            std::atan2( poly[i].y() - poly[i-1].y(),
                        poly[i].x() - poly[i-1].x() )
            : 0.;
          return RotatedRectangle(xp, yp, size.width(),
                                  size.height(), angle);
        }

      length += seglength;
    }

  // shouldn't get here
  return RotatedRectangle();
}

// these are the positions where labels might be placed
namespace
{
#define NUM_LABEL_POSITIONS 7
  const double label_positions[NUM_LABEL_POSITIONS] = {
    0.5, 1/3., 2/3., 0.4, 0.6, 0.25, 0.75};
}

void LineLabeller::process()
{
  RectangleOverlapTester rtest;

  // iterate over each set of polylines
  for(int polyseti = 0; polyseti < _polys.size(); ++polyseti)
    {
      const PolyVector& pv = _polys[polyseti];
      QSizeF size = _textsizes[polyseti];

      for(int polyi = 0; polyi < pv.size(); ++polyi)
        {
          for(int posi = 0; posi < NUM_LABEL_POSITIONS; ++posi)
            {
              const RotatedRectangle r = 
                findLinePosition(pv[polyi], label_positions[posi], size);
              if( ! r.isValid() )
                break;

              if( ! rtest.willOverlap(r) )
                {
                  drawAt(polyseti, r);
                  rtest.addRect(r);
                  break; // only add label once
                }
            } // positions

        } // polylines in set of polylines

    } // sets of polylines
}

QVector<QPolygonF> LineLabeller::getPolySet(int i) const
{
  if( i >= 0 && i < _polys.size() )
    return _polys[i];
  return QVector<QPolygonF>();
}
