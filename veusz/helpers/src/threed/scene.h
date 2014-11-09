#ifndef SCENE_H
#define SCENE_H

#include <QtGui/QPainter>
#include "mmaths.h"
#include "objects.h"

class Scene
{
 public:
  Scene()
  {
  }
  void render(QPainter* painter, const Camera& cam,
	      float x1, float y1, float x2, float y2);

 public:
  ObjectContainer root;

private:
  FragmentVector fragments;
};

#endif
