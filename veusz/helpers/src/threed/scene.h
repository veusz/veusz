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
	      float x1, float y1, float x2, float y2);

private:
  void doSplitting(unsigned idx1, const Camera& cam);
  void doDrawing(QPainter* painter, const Mat3& screenM);
  void fineZCompare();

public:
  ObjectContainer root;

private:
  FragmentVector fragments;
  std::vector<unsigned> depths;
};

#endif
