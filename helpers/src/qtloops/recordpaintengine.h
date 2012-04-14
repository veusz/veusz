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

#ifndef RECORD_PAINT_ENGINE__H
#define RECORD_PAINT_ENGINE__H

#include <QPaintEngine>
#include <QRectF>
#include <QRect>
#include <QImage>
#include <QLineF>
#include <QLine>
#include <QPainterPath>
#include <QRectF>
#include <QRect>
#include <QPixmap>

class RecordPaintDevice;

class RecordPaintEngine : public QPaintEngine
{
public:
  RecordPaintEngine();

  // standard methods to be overridden in engine
  bool begin(QPaintDevice* pdev);

  void drawEllipse(const QRectF& rect);
  void drawEllipse(const QRect& rect);
  void drawImage(const QRectF& rectangle, const QImage& image,
		 const QRectF& sr,
		 Qt::ImageConversionFlags flags = Qt::AutoColor);
  void drawLines(const QLineF* lines, int lineCount);
  void drawLines(const QLine* lines, int lineCount);
  void drawPath(const QPainterPath& path);
  void drawPixmap(const QRectF& r, const QPixmap& pm, const QRectF& sr);
  void drawPoints(const QPointF* points, int pointCount);
  void drawPoints(const QPoint* points, int pointCount);
  void drawPolygon(const QPointF* points, int pointCount,
		   QPaintEngine::PolygonDrawMode mode);
  void drawPolygon(const QPoint* points, int pointCount,
		   QPaintEngine::PolygonDrawMode mode);
  void drawRects(const QRectF* rects, int rectCount);
  void drawRects(const QRect* rects, int rectCount);
  void drawTextItem(const QPointF& p, const QTextItem& textItem);
  void drawTiledPixmap(const QRectF& rect, const QPixmap& pixmap,
		       const QPointF& p);
  bool end ();
  QPaintEngine::Type type () const;
  void updateState(const QPaintEngineState& state);
  
  // return an estimate of number of items drawn
  int drawItemCount() const { return _drawitemcount; }

private:
  int _drawitemcount;
  RecordPaintDevice* _pdev;
};

#endif
