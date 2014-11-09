#include <stdio.h>

#include <cmath>
#include <limits>
#include <QtCore/QPointF>
#include <QtGui/QPolygonF>
#include <QtGui/QPen>
#include <QtGui/QBrush>
#include <QtGui/QColor>

#include "scene.h"

namespace
{
  float minDepth(const Fragment& f)
  {
    switch(f.type)
      {
      case Fragment::FR_TRIANGLE:
	return std::min(f.proj[0](2), std::min(f.proj[1](2), f.proj[2](2)));
      case Fragment::FR_LINESEG:
	return std::min(f.proj[0](2), f.proj[1](2));
      case Fragment::FR_PATH:
	return f.proj[0](2);
      default:
	return std::numeric_limits<float>::infinity();
      }
  }

  float maxDepth(const Fragment& f)
  {
    switch(f.type)
      {
      case Fragment::FR_TRIANGLE:
	return std::max(f.proj[0](2), std::max(f.proj[1](2), f.proj[2](2)));
      case Fragment::FR_LINESEG:
	return std::max(f.proj[0](2), f.proj[1](2));
      case Fragment::FR_PATH:
	return f.proj[0](2);
      default:
	return -std::numeric_limits<float>::infinity();
      }
  }

  float meanDepth(const Fragment& f)
  {
    switch(f.type)
      {
      case Fragment::FR_TRIANGLE:
	return (f.proj[0](2) + f.proj[1](2) + f.proj[2](2))*(1/3.f);
      case Fragment::FR_LINESEG:
	return (f.proj[0](2) + f.proj[1](2))*0.5f;
      case Fragment::FR_PATH:
	return f.proj[0](2);
      default:
	return std::numeric_limits<float>::infinity();
      }
  }

  struct FragDepthCompare
  {
    FragDepthCompare(FragmentVector& v)
      : vec(v)
    {}

    bool operator()(unsigned i, unsigned j) const
    {
      return meanDepth(vec[i]) > meanDepth(vec[j]);
    }

    FragmentVector& vec;
  };

  // Make scaling matrix to move points to correct output range
  Mat3 makeScreenM(const FragmentVector& frags,
		   float x1, float y1, float x2, float y2)
  {
    // get range of projected points in x and y
    float minx, miny, maxx, maxy;
    minx = miny = std::numeric_limits<float>::infinity();
    maxx = maxy = -std::numeric_limits<float>::infinity();

    for(FragmentVector::const_iterator f=frags.begin(); f!=frags.end(); ++f)
      {
	for(unsigned p=0, np=f->nPoints(); p<np; ++p)
	  {
	    float x = f->proj[p](0);
	    float y = f->proj[p](1);
	    if(std::isfinite(x) && std::isfinite(y))
	      {
		minx = std::min(minx, x);
		maxx = std::max(maxx, x);
		miny = std::min(miny, y);
		maxy = std::max(maxy, y);
	      }
	  }
      }

    // catch bad values or empty arrays
    if(maxx == minx || !std::isfinite(minx) || !std::isfinite(maxx))
      {
	maxx=1; minx=0;
      }
    if(maxy == miny || !std::isfinite(miny) || !std::isfinite(maxy))
      {
	maxy=1; miny=0;
      }

    // now make matrix to scale to range x1->x2,y1->y2
    float minscale = std::min((x2-x1)/(maxx-minx), (y2-y1)/(maxy-miny));
    return
      translateM3(0.5*(x1+x2), 0.5*(y1+y2)) *
      scaleM3(minscale) *
      translateM3(-0.5*(minx+maxx), -0.5*(miny+maxy));
  }

  QPen LineProp2QPen(const LineProp& p)
  {
    if(p.hide)
      return QPen(Qt::NoPen);
    else
      return QPen(QBrush(QColor(int(p.r*255), int(p.g*255),
				int(p.b*255), int((1-p.trans)*255))),
		  p.width);
  }

  QBrush SurfaceProp2QBrush(const SurfaceProp& p)
  {
    if(p.hide)
      return QBrush();
    else
      return QBrush(QColor(int(p.r*255), int(p.g*255),
			   int(p.b*255), int((1-p.trans)*255)));
  }

  // convert (x,y,depth) -> screen coordinates
  QPointF vecToScreen(const Mat3& screenM, const Vec3& vec)
  {
    Vec3 mult(screenM*Vec3(vec(0), vec(1), 1));
    float inv = 1/mult(2);
    return QPointF(mult(0)*inv, mult(1)*inv);
  }

};

void Scene::render(QPainter* painter, const Camera& cam,
		   float x1, float y1, float x2, float y2)
{
  fragments.clear();

  // get fragments for whole scene
  root.getFragments(cam.viewM, cam, fragments);

  // store sorted indices to fragments here
  std::vector<unsigned> depths(fragments.size());;
  for(unsigned i=0, s=fragments.size(); i<s; ++i)
    depths[i]=i;

  // stable sort used to preserve original order
  std::stable_sort(depths.begin(), depths.end(), FragDepthCompare(fragments));

  // how to transform points to screen
  const Mat3 screenM(makeScreenM(fragments, x1, y1, x2, y2));

  // draw fragments
  LineProp const* lline = 0;
  SurfaceProp const* lsurf = 0;

  QPen no_pen(Qt::NoPen);
  QBrush no_brush(Qt::NoBrush);
  painter->setPen(no_pen);
  painter->setBrush(no_brush);

  QPolygonF temppoly(3);

  for(unsigned i=0, s=depths.size(); i<s; ++i)
    {
      const Fragment& f(fragments[i]);
      switch(f.type)
	{
	case Fragment::FR_TRIANGLE:
	  if(lline != 0)
	    {
	      painter->setPen(no_pen);
	      lline = 0;
	    }
	  if(f.surfaceprop != 0 && lsurf != f.surfaceprop)
	    {
	      lsurf = f.surfaceprop;
	      painter->setBrush(SurfaceProp2QBrush(*lsurf));
	    }

	  temppoly[0] = vecToScreen(screenM, f.proj[0]);
	  temppoly[1] = vecToScreen(screenM, f.proj[1]);
	  temppoly[2] = vecToScreen(screenM, f.proj[2]);
	  painter->drawPolygon(temppoly);
	  break;

	case Fragment::FR_LINESEG:
	  if(lsurf != 0)
	    {
	      painter->setBrush(no_brush);
	      lsurf = 0;
	    }
	  if(f.lineprop != 0 && lline != f.lineprop)
	    {
	      lline = f.lineprop;
	      painter->setPen(LineProp2QPen(*lline));
	    }

	  painter->drawLine(vecToScreen(screenM, f.proj[0]),
			    vecToScreen(screenM, f.proj[1]));
	  break;

	case Fragment::FR_PATH:
	  break;
	}
    }
}
