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

#include <QtAlgorithms>
#include <limits>
#include "recordpaintdevice.h"
#include "recordpaintengine.h"

#define INCH_MM 25.4

RecordPaintDevice::RecordPaintDevice(int width, int height,
				     int dpix, int dpiy)
  :_width(width), _height(height), _dpix(dpix), _dpiy(dpiy),
   _engine(new RecordPaintEngine)
{
}

RecordPaintDevice::~RecordPaintDevice()
{
  delete _engine;
  qDeleteAll(_elements);
}

QPaintEngine* RecordPaintDevice::paintEngine() const
{
  return _engine;
}

int RecordPaintDevice::metric(QPaintDevice::PaintDeviceMetric metric) const
{
  switch(metric) {
  case QPaintDevice::PdmWidth:
    return _width;
  case QPaintDevice::PdmHeight:
    return _height;
  case QPaintDevice::PdmWidthMM:
    return int(_width * INCH_MM / _dpix);
  case QPaintDevice::PdmHeightMM:
    return int(_height * INCH_MM / _dpiy);
  case QPaintDevice::PdmNumColors:
    return std::numeric_limits<int>::max();
  case QPaintDevice::PdmDepth:
    return 24;
  case QPaintDevice::PdmDpiX:
  case QPaintDevice::PdmPhysicalDpiX:
    return _dpix;
  case QPaintDevice::PdmDpiY:
  case QPaintDevice::PdmPhysicalDpiY:
    return _dpiy;
  default:
    return -1;
  }
}

void RecordPaintDevice::play(QPainter& painter)
{
  QTransform origtransform(painter.worldTransform());
  foreach(PaintElement* el, _elements)
    {
      el->paint(painter, origtransform);
    }
}
