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

%Module(name=recordpaint)

%Import(name=QtCore/QtCoremod.sip)
%Import(name=QtGui/QtGuimod.sip)

class RecordPaintDevice : QPaintDevice
 {
%TypeHeaderCode
#include <recordpaintdevice.h>
%End

public:
  RecordPaintDevice(int width, int height, int dpix, int dpiy);
  ~RecordPaintDevice();
  void play(QPainter& painter);

  QPaintEngine* paintEngine() const;

  int metric(QPaintDevice::PaintDeviceMetric metric) const;
  int drawItemCount() const;
 };
