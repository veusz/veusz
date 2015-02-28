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
      double d1 = vec[i].maxDepth();
      double d2 = vec[j].maxDepth();
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
      double d1 = vec[i].minDepth();
      double d2 = vec[j].minDepth();

      if(std::abs(d1-d2)<1e-5)
	{
	  // if the minima are the same, then look at the maxima
	  return vec[i].maxDepth() > vec[j].maxDepth();
	}
      return d1 > d2;
    }

    FragmentVector& vec;
  };

  struct FragDepthCompareMean
  {
    FragDepthCompareMean(FragmentVector& v)
      : vec(v)
    {}

    bool operator()(unsigned i, unsigned j) const
    {
      double d1 = vec[i].meanDepth();
      double d2 = vec[j].meanDepth();
      return d1 > d2;
    }

    FragmentVector& vec;
  };


  struct FragDepthCompareOverlap
  {
    FragDepthCompareOverlap(FragmentVector& v)
      : vec(v)
    {}

    bool operator()(unsigned i, unsigned j) const
    {
      const Fragment& f1(vec[i]);
      const Fragment& f2(vec[j]);

      if( f1.maxDepth() <= f2.minDepth() )
        // definitely f1 is in front of f2
        return 0;
      if( f1.minDepth() >= f2.maxDepth() )
        // definitely f1 is behind f2
        return 1;

      double d1, d2;
      d1 = d2 = std::numeric_limits<double>::min();
      overlapDepth(f1, f2, &d1, &d2);
      if( d1 == std::numeric_limits<double>::min() )
        // did not get a sensible answer
        return f1.meanDepth() >= f2.meanDepth();
      else
        return d1 >= d2;
    }

    FragmentVector& vec;
  };

  // Make scaling matrix to move points to correct output range
  Mat3 makeScreenM(const FragmentVector& frags,
		   double x1, double y1, double x2, double y2)
  {
    // get range of projected points in x and y
    double minx, miny, maxx, maxy;
    minx = miny = std::numeric_limits<double>::infinity();
    maxx = maxy = -std::numeric_limits<double>::infinity();

    for(FragmentVector::const_iterator f=frags.begin(); f!=frags.end(); ++f)
      {
	for(unsigned p=0, np=f->nPoints(); p<np; ++p)
	  {
	    double x = f->proj[p](0);
	    double y = f->proj[p](1);
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
    double minscale = std::min((x2-x1)/(maxx-minx), (y2-y1)/(maxy-miny));
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
    double inv = 1/mult(2);
    return QPointF(mult(0)*inv, mult(1)*inv);
  }

}; // namespace

// split up fragments which overlap in 3D into mutiple non-overlapping
// fragments
void Scene::doSplitting(unsigned idx1, const Camera& cam)
{
  // printf("split: %i, %g\n", idx1, fragments[depths[idx1]].minDepth());
  double thismindepth = fragments[depths[idx1]].minDepth();

 start:
  for(unsigned idx2=idx1+1; idx2<depths.size(); ++idx2)
    {
      // printf(" %i, %g, %g\n", idx2, fragments[depths[idx2]].minDepth(),
      //        fragments[depths[idx2]].maxDepth());

      // don't compare object with self
      if(fragments[depths[idx2]].object == fragments[depths[idx1]].object)
	continue;

      if(fragments[depths[idx2]].maxDepth() < thismindepth)
	// no others fragments are overlapping, as any others would be
	// less deep
	break;

      // try to split, returning number of new fragments (if any)
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

      // calculate new depths for fragments and resort region
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

  const unsigned depthssize = depths.size();
  for(unsigned idx1=0; idx1+1<depthssize; ++idx1)
    {
      unsigned loopcount = 0;
      double thismindepth;

    start:
      thismindepth = fragments[depths[idx1]].minDepth();

      for(unsigned idx2=idx1+1; idx2<depthssize; ++idx2)
	{
	  if(fragments[depths[idx2]].maxDepth() < thismindepth)
	    // no others are overlapping
	    break;

	  double d1 = -1;
	  double d2 = -1;
	  overlapDepth(fragments[depths[idx1]], fragments[depths[idx2]],
		       &d1, &d2);

	  if( d2-d1 > 1e-5 )
	    {
	      // bubble fragment above one at idx1
	      unsigned tmp = depths[idx2];
	      for(int i=int(idx2)-1; i >= int(idx1); --i)
		depths[i+1] = depths[i];
	      depths[idx1] = tmp;

	      // avoid infinite loops
	      if(loopcount++ < 16)
		goto start;
	    }
	}
    }
}

void Scene::render(QPainter* painter, const Camera& cam,
		   double x1, double y1, double x2, double y2)
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

  printf("before: %li %li\n", fragments.size(), depths.size());
  for(unsigned idx=0; idx+1 < depths.size(); ++idx)
    doSplitting(idx, cam);
  printf("after:  %li %li\n", fragments.size(), depths.size());

  // final sorting
  std::sort(depths.begin(),
  	    depths.end(),
  	    FragDepthCompareMean(fragments));

  //fineZCompare();
  printf("after2: %li %li\n", fragments.size(), depths.size());

  // how to transform projected points to screen
  const Mat3 screenM(makeScreenM(fragments, x1, y1, x2, y2));

  // finally draw items
  doDrawing(painter, screenM);

  simpleDump();
  objDump();
}

void Scene::objDump()
{
  FILE* fobj = fopen("dump.obj", "w");
  FILE *fmtl = fopen("dump.mtl", "w");

  fprintf(fobj, "mtllib dump.mtl\n");

  unsigned ct = 1;
  for(unsigned i=0, s=depths.size(); i<s; ++i)
    {
      const Fragment& f = fragments[depths[i]];
      for(unsigned j=0; j!=f.nPoints(); ++j)
        fprintf(fobj, "v %g %g %g\n", f.proj[j](0), f.proj[j](1), f.proj[j](2));

      if(f.nPoints()==3)
        {
          double r = rand() / double(RAND_MAX);
          double g = rand() / double(RAND_MAX);
          double b = rand() / double(RAND_MAX);

          fprintf(fmtl, "newmtl col%i\n", i);
          fprintf(fmtl, "Ka %g %g %g\n", r, g, b);
          fprintf(fmtl, "Kd %g %g %g\n", r, g, b);
          fprintf(fmtl, "Ks %g %g %g\n", r, g, b);

          fprintf(fobj, "usemtl col%i\n", i);
          fprintf(fobj, "f %i %i %i\n", ct, ct+1, ct+2);
        }

      if(f.nPoints()==2)
        fprintf(fobj, "l %i %i\n", ct, ct+1);

      ct += f.nPoints();
    }


  fclose(fobj);
  fclose(fmtl);
}


void Scene::simpleDump()
{
  FILE* file = fopen("dump.svg", "w");
  fprintf(file, "<svg>\n");

  double mind = 1e10;
  double maxd = -1e10;
  for(unsigned i=0, s=depths.size(); i<s; ++i)
    {
      const Fragment& f(fragments[depths[i]]);

      if(f.type==Fragment::FR_TRIANGLE)
        {
          double av = (f.proj[0](2)+f.proj[1](2)+f.proj[2](2))/3.;
          if(av<mind) mind=av;
          if(av>maxd) maxd=av;
        }
    }
  printf("%g %g\n", mind, maxd);

  for(unsigned i=0, s=depths.size(); i<s; ++i)
    {
      const Fragment& f(fragments[depths[i]]);

      double av = (f.proj[0](2)+f.proj[1](2)+f.proj[2](2))/3.;

      switch(f.type)
	{
	  // debug
	  //painter->setPen(solid);
	  //painter->setBrush(no_brush);

	case Fragment::FR_TRIANGLE:
	  fprintf(file,
		  "<polygon fill=\"#%02x%02x%02x\" "
		  "id=\"p%i\" "
		  "points=\"%g,%g %g,%g %g,%g\" "
		  "/>\n",
                  //		  int(f.surfaceprop->r*255),
                  //		  int(f.surfaceprop->g*255),
                  //		  int(f.surfaceprop->b*255),
                  //int(f.surfaceprop->r*255),
                  //int(f.surfaceprop->g*255),
                  int((av-mind)/(maxd-mind)*255),
                  int(f.surfaceprop->g*255),
                  int((av-mind)/(maxd-mind)*255),
		  f.index,
		  f.proj[0](0)*300+300, f.proj[0](1)*300+300,
		  f.proj[1](0)*300+300, f.proj[1](1)*300+300,
		  f.proj[2](0)*300+300, f.proj[2](1)*300+300);

	  break;

	case Fragment::FR_LINESEG:
	  break;

	case Fragment::FR_PATH:
	  break;

	default:
	  break;
	}
    }

  fprintf(file, "</svg>\n");
  fclose(file);
}
