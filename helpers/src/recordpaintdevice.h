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

#ifndef RECORD_PAINT_DEVICE__H
#define RECORD_PAINT_DEVICE__H

#include <QPaintDevice>
#include <QVector>
#include "paintelement.h"
#include "recordpaintengine.h"

class RecordPaintDevice : public QPaintDevice
{
public:
  RecordPaintDevice(int width, int height, int dpix, int dpiy);
  ~RecordPaintDevice();
  QPaintEngine* paintEngine() const;

  // play back all 
  void play(QPainter& painter);

  int metric(QPaintDevice::PaintDeviceMetric metric) const;

  int drawItemCount() const { return _engine->drawItemCount(); }

public:
  friend class RecordPaintEngine;

private:
  // add an element to the list of maintained elements
  void addElement(PaintElement* el)
  {
    _elements.push_back(el);
  }

private:
  int _width, _height, _dpix, _dpiy;
  RecordPaintEngine* _engine;
  QVector<PaintElement*> _elements;
};

#endif
