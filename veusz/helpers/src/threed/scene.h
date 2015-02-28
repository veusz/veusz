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
  void doSplitting(unsigned idx1, const Camera& cam);
  void doDrawing(QPainter* painter, const Mat3& screenM);
  void fineZCompare();
  void simpleDump();
  void objDump();

public:
  ObjectContainer root;

private:
  FragmentVector fragments;
  std::vector<unsigned> depths;
};

#endif
