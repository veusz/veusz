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

#ifndef INTERMEDIATE_PAINT_DEVICE__H
#define INTERMEDIATE_PAINT_DEVICE__H

#include <QPaintDevice>
#include <QVector>
#include "paintelement.h"
#include "intermediatepaintengine.h"

class IntermediatePaintDevice : public QPaintDevice
{
public:
  IntermediatePaintDevice(int width, int height, int dpi);
  ~IntermediatePaintDevice();
  QPaintEngine* paintEngine() const;

  // play back all 
  void playback(QPainter& painter);

  int metric(QPaintDevice::PaintDeviceMetric metric) const;

public:
  friend class IntermediatePaintEngine;

private:
  // add an element to the list of maintained elements
  void addElement(PaintElement* el)
  {
    _elements.push_back(el);
  }

private:
  int _width, _height, _dpi;
  IntermediatePaintEngine* _engine;
  QVector<PaintElement*> _elements;
};

#endif
