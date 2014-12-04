#include <cstdio>
#include <cmath>
#include <limits>
#include <QtCore/QPointF>
#include <QtGui/QPolygonF>
#include <QtGui/QPen>
#include <QtGui/QBrush>
#include <QtGui/QColor>

#include "scene.h"
#include "fragment.h"

namespace
{

  struct FragDepthCompare
  {
    FragDepthCompare(FragmentVector& v)
      : vec(v)
    {}

    bool operator()(unsigned i, unsigned j) const
    {
      float d1 = vec[i].maxDepth();
      float d2 = vec[j].maxDepth();
      if(std::abs(d1-d2)<1e-5)
	{
	  // if the maxima are the same, then look at the minima
	  return vec[i].minDepth() > vec[j].minDepth();
	}
      return d1 > d2;
    }

    FragmentVector& vec;
  };


  struct FragDepthCompareMin
  {
    FragDepthCompareMin(FragmentVector& v)
      : vec(v)
    {}

    bool operator()(unsigned i, unsigned j) const
    {
      float d1 = vec[i].minDepth();
      float d2 = vec[j].minDepth();

      if(std::abs(d1-d2)<1e-5)
	{
	  // if the maxima are the same, then look at the minima
	  return vec[i].maxDepth() > vec[j].maxDepth();
	}
      return d1 > d2;
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

}; // namespace

void Scene::doSplitting(unsigned idx1, const Camera& cam)
{
  float thismindepth = fragments[depths[idx1]].minDepth();

 start:
  for(unsigned idx2=idx1+1; idx2<depths.size(); ++idx2)
    {
      // don't compare object with self
      if(fragments[depths[idx2]].object == fragments[depths[idx1]].object)
	continue;

      if(fragments[depths[idx2]].maxDepth() < thismindepth)
	// no others are overlapping
	break;

      unsigned newnum1=0, newnum2=0;
      splitFragments(fragments[depths[idx1]],
		     fragments[depths[idx2]],
		     fragments, &newnum1, &newnum2);
      if(newnum1>0)
	{
	  if(newnum1>1)
	    {
	      depths.insert(depths.begin()+idx1, newnum1-1, 0);
	      idx2 += newnum1-1;
	    }
	  unsigned base=fragments.size()-newnum1-newnum2;
	  for(unsigned i=0; i<newnum1; ++i)
	    depths[idx1+i] = base+i;
	}
      if(newnum2>0)
	{
	  if(newnum2>1)
	    depths.insert(depths.begin()+idx2, newnum2-1, 0);
	  unsigned base=fragments.size()-newnum2;
	  for(unsigned i=0; i<newnum2; ++i)
	    depths[idx2+i] = base+i;
	}
      if(newnum1+newnum2 > 0)
	{
	  unsigned nlen = fragments.size();
	  // calculate projected coordinates (with depths)
	  for(unsigned i=nlen-(newnum1+newnum2); i != nlen; ++i)
	    fragments[i].updateProjCoords(cam.perspM);

	  // sort depth of new items
	  std::sort(depths.begin()+idx1,
		    depths.begin()+(idx2+newnum2-1+1),
		    FragDepthCompare(fragments));
	  // go back to start
	  goto start;
	}
    }
}

void Scene::doDrawing(QPainter* painter, const Mat3& screenM)
{
  // draw fragments
  LineProp const* lline = 0;
  SurfaceProp const* lsurf = 0;

  QPen solid;
  QPen no_pen(Qt::NoPen);
  QBrush no_brush(Qt::NoBrush);
  painter->setPen(no_pen);
  painter->setBrush(no_brush);

  QPolygonF temppoly(3);
  QPointF projpts[3];

  for(unsigned i=0, s=depths.size(); i<s; ++i)
    {
      const Fragment& f(fragments[depths[i]]);

      // convert projected points to screen
      for(unsigned pi=0, s=f.nPoints(); pi<s; ++pi)
	projpts[pi] = vecToScreen(screenM, f.proj[pi]);

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

	  // debug
	  //painter->setPen(solid);
	  //painter->setBrush(no_brush);

	  temppoly[0] = projpts[0];
	  temppoly[1] = projpts[1];
	  temppoly[2] = projpts[2];
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
	  painter->drawLine(projpts[0], projpts[1]);
	  break;

	case Fragment::FR_PATH:
	  break;

	default:
	  break;
	}
    }
}

void Scene::fineZCompare()
{
  // look at intersections between triangles (if any) and compare
  // depths at these intersections to determine order

  for(unsigned idx1=0; idx1+1 < depths.size(); ++idx1)
    {
      float thismindepth = fragments[depths[idx1]].minDepth();
      unsigned loopcount = 0;

    start:
      for(unsigned idx2=idx1+1; idx2<depths.size(); ++idx2)
	{
	  if(fragments[depths[idx2]].maxDepth() < thismindepth)
	    // no others are overlapping
	    break;

	  float d1 = -1;
	  float d2 = -1;
	  overlapDepth(fragments[depths[idx1]], fragments[depths[idx2]],
		       &d1, &d2);
	  if( d1-d2 < -1e-5 )
	    {
	      std::swap(depths[idx1], depths[idx2]);
	      // avoid infinite loops
	      if(loopcount++ < 16)
		goto start;
	    }
	}
    }
}


void Scene::render(QPainter* painter, const Camera& cam,
		   float x1, float y1, float x2, float y2)
{
  fragments.clear();

  // get fragments for whole scene
  root.getFragments(cam.viewM, cam, fragments);

  // store sorted indices to fragments here
  depths.resize(fragments.size());
  for(unsigned i=0, s=fragments.size(); i<s; ++i)
    depths[i]=i;

  // sort depth of items
  std::sort(depths.begin(), depths.end(), FragDepthCompare(fragments));

  for(unsigned idx=0; idx+1 < depths.size(); ++idx)
    doSplitting(idx, cam);

  // final sorting
  std::sort(depths.begin(),
  	    depths.end(),
  	    FragDepthCompareMin(fragments));

  fineZCompare();

  // how to transform projected points to screen
  const Mat3 screenM(makeScreenM(fragments, x1, y1, x2, y2));

  // finally draw items
  doDrawing(painter, screenM);
}
