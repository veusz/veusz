// -*-c++-*-

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
  void splitIntersectIn3D(unsigned idx1, const Camera& cam);
  void doDrawing(QPainter* painter, const Mat3& screenM);
  void fineZCompare();
  void splitProjected();
  void simpleDump();
  void objDump();

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
