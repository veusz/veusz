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
	      const Mat3& screenM);

 public:
  ObjectContainer root;

private:
  FragmentVector fragments;
};

#endif
