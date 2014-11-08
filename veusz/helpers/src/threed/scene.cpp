#include <limits>
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
		   const Mat3& screenM)
{
  fragments.clear();

  // get fragments for whole scene
  root.getFragments(identityM4(), cam, fragments);

  // store sorted indices to fragments here
  std::vector<unsigned> depths(fragments.size());;
  for(unsigned i=0, s=fragments.size(); i<s; ++i)
    depths[i]=i;

  // stable sort used to preserve original order
  std::stable_sort(depths.begin(), depths.end(), FragDepthCompare(fragments));

  // draw fragments
  LineProp const* lline = 0;
  SurfaceProp const* lsurf = 0;

  QPen no_pen(Qt::NoPen);
  QBrush no_brush(Qt::NoBrush);

  QPolygonF temppoly(3);

  for(FragmentVector::const_iterator f=fragments.begin();
      f != fragments.end(); ++f)
    {
      switch(f->type)
	{
	case Fragment::FR_TRIANGLE:
	  if(lline != 0)
	    {
	      painter->setPen(no_pen);
	      lline = 0;
	    }
	  if(lsurf != f->surfaceprop)
	    {
	      lsurf = f->surfaceprop;
	      painter->setBrush(SurfaceProp2QBrush(*lsurf));
	    }
	  temppoly[0] = vecToScreen(screenM, f->proj[0]);
	  temppoly[1] = vecToScreen(screenM, f->proj[1]);
	  temppoly[2] = vecToScreen(screenM, f->proj[2]);
	  painter->drawPolygon(temppoly);
	  break;

	case Fragment::FR_LINESEG:
	  if(lsurf != 0)
	    {
	      painter->setBrush(no_brush);
	      lsurf = 0;
	    }
	  if(lline != f->lineprop)
	    {
	      lline = f->lineprop;
	      painter->setPen(LineProp2QPen(*lline));
	    }

	  painter->drawLine(vecToScreen(screenM, f->proj[0]),
			    vecToScreen(screenM, f->proj[1]));
	  break;

	case Fragment::FR_PATH:
	  break;
	}
    }
}
