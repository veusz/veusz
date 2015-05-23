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

#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <QtGui/QPainter>
#include "mmaths.h"
#include "objects.h"
#include "camera.h"

class Scene
{
public:
  Scene()
  {
  }

  void render(QPainter* painter, const Camera& cam,
	      double x1, double y1, double x2, double y2);

private:
  void projectFragments(const Camera& cam);
  void splitIntersectIn3D(unsigned idx1, const Camera& cam);
  void doDrawing(QPainter* painter, const Mat3& screenM, double linescale);
  void fineZCompare();
  void splitProjected();
  void simpleDump();
  void objDump();

  void drawPath(QPainter* painter, const Fragment& frag, QPointF pt,
                double linescale);

  // insert newnum1 fragments at idx1 and newnum2 fragments at idx2
  // into the depths array from the end of fragments
  // also sort the idx1->idx2+newnum1+newnum2 in depth order
  void insertFragmentsIntoDepths(unsigned idx1, unsigned newnum1,
                                 unsigned idx2, unsigned newnum2);

public:
  ObjectContainer root;

private:
  FragmentVector fragments;
  std::vector<unsigned> depths;
};

#endif
