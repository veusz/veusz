// -*-c++-*-

//    Copyright (C) 2015 Jeremy S. Sanders
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

#ifndef CLIPCONTAINER_H
#define CLIPCONTAINER_H

#include "objects.h"
#include "fragment.h"

// container which clips children in a 3D box
class ClipContainer : public ObjectContainer
{
public:
  ClipContainer(Vec3 _minpt, Vec3 _maxpt)
    : ObjectContainer(), minpt(_minpt), maxpt(_maxpt)
  {
  }

  void getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v);

  bool pointInBounds(Vec3 pt) const
  {
    return (pt(0) >= minpt(0) && pt(1) >= minpt(1) && pt(2) >= minpt(2) &&
            pt(0) <= maxpt(0) && pt(1) <= maxpt(1) && pt(2) <= maxpt(2));
  }

 public:
  Vec3 minpt, maxpt;
};

#endif
