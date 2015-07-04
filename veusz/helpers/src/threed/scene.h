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
  enum RenderMode {RENDER_PAINTERS, RENDER_BSP};

private:
  // internal light color and position
  struct Light
  {
    Vec3 posn;
    double r, g, b;
  };

public:
  Scene(RenderMode _mode)
    : mode(_mode)
  {
  }

  // add a light to a list
  void addLight(Vec3 posn, QColor col, double intensity);

  // render scene to painter in coordinate range given
  void render(Object* root,
              QPainter* painter, const Camera& cam,
	      double x1, double y1, double x2, double y2);

private:
  // calculate lighting norms for triangles
  void calcLighting();

  // compute projected coordinates
  void projectFragments(const Camera& cam);

  void doDrawing(QPainter* painter, const Mat3& screenM, double linescale);

  void drawPath(QPainter* painter, const Fragment& frag,
                QPointF pt1, QPointF pt2, double linescale);

  // different rendering modes
  void renderPainters(const Camera& cam);
  void renderBSP(const Camera& cam);

  // create pens/brushes
  QPen lineProp2QPen(const Fragment& frag, double linescale) const;
  QColor surfaceProp2QColor(const Fragment& frag) const;
  QBrush surfaceProp2QBrush(const Fragment& frag) const;
  QPen surfaceProp2QPen(const Fragment& frag) const;

private:
  RenderMode mode;
  FragmentVector fragments;
  std::vector<unsigned> draworder;
  std::vector<Light> lights;
};

#endif
