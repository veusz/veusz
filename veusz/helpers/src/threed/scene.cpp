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

#include <cmath>
#include <limits>
#include <QtCore/QPointF>
#include <QtGui/QPolygonF>
#include <QtGui/QPen>
#include <QtGui/QBrush>
#include <QtGui/QColor>

#include "scene.h"
#include "fragment.h"
#include "bsp.h"

#include <cstdio>
#include <iostream>

namespace
{

  struct FragDepthCompareMax
  {
    FragDepthCompareMax(FragmentVector& v)
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

  QPen LineProp2QPen(const LineProp* p, double linescale)
  {
    if(p==0 || p->hide)
      return QPen(Qt::NoPen);
    else
      return QPen(QBrush(QColor(int(p->r*255), int(p->g*255),
				int(p->b*255), int((1-p->trans)*255))),
		  p->width*linescale);
  }

  QBrush SurfaceProp2QBrush(const SurfaceProp* p)
  {
    if(p==0 || p->hide)
      return QBrush();
    else
      return QBrush(QColor(int(p->r*255), int(p->g*255),
			   int(p->b*255), int((1-p->trans)*255)));
  }

  // convert (x,y,depth) -> screen coordinates
  QPointF vecToScreen(const Mat3& screenM, const Vec3& vec)
  {
    Vec3 mult(screenM*Vec3(vec(0), vec(1), 1));
    double inv = 1/mult(2);
    return QPointF(mult(0)*inv, mult(1)*inv);
  }

}; // namespace

// insert newnum1 fragments at idx1 and newnum2 fragments at idx2
// into the depths array from the end of fragments
// also sort the idx1->idx2+newnum1+newnum2 in depth order
void Scene::insertFragmentsIntoDepths(unsigned idx1, unsigned newnum1,
                                      unsigned idx2, unsigned newnum2)
{
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
      // sort depth of new items
      std::sort(depths.begin()+idx1,
                depths.begin()+(idx2+newnum2-1+1),
                FragDepthCompareMax(fragments));
    }
}



// split up fragments which overlap in 3D into mutiple non-overlapping
// fragments
void Scene::splitIntersectIn3D(unsigned idx1, const Camera& cam)
{
 RESTART:

  double thismindepth = fragments[depths[idx1]].minDepth();
  for(unsigned idx2=idx1+1; idx2<depths.size(); ++idx2)
    {
      // printf(" %i, %g, %g\n", idx2, fragments[depths[idx2]].minDepth(),
      //        fragments[depths[idx2]].maxDepth());

      // don't compare object with self
      if(fragments[depths[idx2]].object == fragments[depths[idx1]].object)
	continue;

      // ignore FR_NONE fragments
      if(fragments[depths[idx2]].type == Fragment::FR_NONE)
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

      // calculate new depths for fragments and resort region
      if(newnum1+newnum2 > 0)
	{
          // put and sort new fragments into depths
          insertFragmentsIntoDepths(idx1, newnum1, idx2, newnum2);

	  unsigned nlen = fragments.size();
	  // calculate projected coordinates (with depths)
	  for(unsigned i=nlen-(newnum1+newnum2); i != nlen; ++i)
	    fragments[i].updateProjCoords(cam.perspM);

	  // go back to start
	  goto RESTART;
	}
    }
}

void Scene::splitProjected()
{

  // assume depths already sorted in terms of maximum depth of
  // fragments

  // iterate over fragments
  for(unsigned idx1=0; idx1+1<depths.size(); ++idx1)
    {
    RESTART:

      double thismindepth = fragments[depths[idx1]].minDepth();
      for(unsigned idx2=idx1+1; idx2<depths.size(); ++idx2)
        {
          // fragments beyond this do not overlap
          if(fragments[depths[idx2]].maxDepth() < thismindepth)
            break;

          //printf("%i %i /%li\n", idx1, idx2, depths.size());

          unsigned num1=0, num2=0;
          splitOn2DOverlap(fragments, depths[idx1], depths[idx2], &num1, &num2);

          if(num1>0 || num2>0)
            {
              //printf("num1=%i num2=%i\n", num1, num2);

              // put and sort new fragments into depths
              insertFragmentsIntoDepths(idx1, num1, idx2, num2);
              goto RESTART;
            }
        }
    }
}

void Scene::drawPath(QPainter* painter, const Fragment& frag, QPointF pt, double linescale)
{
  FragmentPathParameters* pars = static_cast<FragmentPathParameters*>(frag.params);
  double scale = frag.pathsize*linescale;
  if(pars->scaleedges)
    {
      painter->save();
      painter->translate(pt.x(), pt.y());
      painter->scale(scale, scale);
      painter->drawPath(*(pars->path));
      painter->restore();
    }
  else
    {
      // scale point and relocate
      QPainterPath path(*(pars->path));
      int elementct = path.elementCount();
      for(int i=0; i<elementct; ++i)
        {
          QPainterPath::Element el = path.elementAt(i);
          path.setElementPositionAt(i,
                                    el.x*scale+pt.x(),
                                    el.y*scale+pt.y());
        }
      painter->drawPath(path);
    }
}

void Scene::doDrawing(QPainter* painter, const Mat3& screenM, double linescale)
{
  // draw fragments
  LineProp const* lline = 0;
  SurfaceProp const* lsurf = 0;

  QPen solid;
  QPen no_pen(Qt::NoPen);
  QBrush no_brush(Qt::NoBrush);
  painter->setPen(no_pen);
  painter->setBrush(no_brush);

  QPointF projpts[3];

  for(unsigned i=0, s=depths.size(); i<s; ++i)
    {
      const Fragment& frag(fragments[depths[i]]);

      // convert projected points to screen
      for(unsigned pi=0, s=frag.nPoints(); pi<s; ++pi)
	projpts[pi] = vecToScreen(screenM, frag.proj[pi]);

      switch(frag.type)
	{
	case Fragment::FR_TRIANGLE:
          if(frag.surfaceprop != 0 && !frag.surfaceprop->hide)
            {
              if(lline != 0)
                {
                  painter->setPen(no_pen);
                  lline = 0;
                }
              if(lsurf != frag.surfaceprop)
                {
                  lsurf = frag.surfaceprop;
                  painter->setBrush(SurfaceProp2QBrush(lsurf));
                }

              // debug
              //painter->setPen(solid);
              //painter->setBrush(no_brush);

              painter->drawPolygon(projpts, 3);
            }
          break;

	case Fragment::FR_LINESEG:
          if(frag.lineprop != 0 && !frag.lineprop->hide)
            {
              if(lsurf != 0)
                {
                  painter->setBrush(no_brush);
                  lsurf = 0;
                }
              if(lline != frag.lineprop)
                {
                  lline = frag.lineprop;
                  painter->setPen(LineProp2QPen(lline, linescale));
                }
              painter->drawLine(projpts[0], projpts[1]);
            }
          break;

	case Fragment::FR_PATH:
          if( (frag.lineprop != 0 && !frag.lineprop->hide) ||
              (frag.surfaceprop != 0 && !frag.surfaceprop->hide) )
            {
              if(lline != frag.lineprop)
                {
                  lline = frag.lineprop;
                  painter->setPen(LineProp2QPen(lline, linescale));
                }
              if(lsurf != frag.surfaceprop)
                {
                  lsurf = frag.surfaceprop;
                  painter->setBrush(SurfaceProp2QBrush(lsurf));
                }
              drawPath(painter, frag, projpts[0], linescale);
            }
	  break;

	default:
	  break;
	}
    }
}

void Scene::projectFragments(const Camera& cam)
{
  for(unsigned i=0, s=fragments.size(); i<s; ++i)
    {
      Fragment& f = fragments[i];
      for(unsigned pi=0, np=f.nPoints(); pi<np; ++pi)
        f.proj[pi] = calcProjVec(cam.perspM, f.points[pi]);
    }
}

void Scene::render(QPainter* painter, const Camera& cam,
		   double x1, double y1, double x2, double y2)
{
  //printf("\nstarting\n");
  fragments.resize(0);

  // get fragments for whole scene
  root.getFragments(cam.viewM, fragments);

  std::cout << "\nFragment size 1 " << fragments.size() << '\n';

  // does this assume a particular layout for the view matrix?
  Vec3 viewdirn(-cam.viewM(2,0), -cam.viewM(2,1), -cam.viewM(2,2));
  BSPBuilder bsp(fragments, viewdirn);
  depths = bsp.getFragmentIdxs(fragments);

  std::cout << "BSP recs size " << bsp.bsp_recs.size() << '\n';
  std::cout << "Fragment size 2 " << fragments.size() << '\n';

  unsigned ct=0;
  for(unsigned i=0; i<fragments.size(); ++i)
    if(fragments[i].type != Fragment::FR_NONE)
      ++ct;
  std::cout << "Used fragments " << ct << '\n';
  std::cout << "Num indexs " << depths.size() << '\n';


  // work out projected coordinates
  projectFragments(cam);

  // store sorted indices to fragments here
  //depths.resize(fragments.size());
  //for(unsigned i=0, s=fragments.size(); i<s; ++i)
   // depths[i]=i;

  // sort depth of items
  //std::sort(depths.begin(), depths.end(), FragDepthCompareMax(fragments));

  //printf("\nsplit in 3d\n");
  //for(unsigned idx=0; idx+1 < depths.size(); ++idx)
  //  splitIntersectIn3D(idx, cam);

  //std::sort(depths.begin(), depths.end(), FragDepthCompareMax(fragments));

  // split on sky
  //printf("\nsplit projected\n");
  //splitProjected();

  // final sorting
  //std::sort(depths.begin(), depths.end(), FragDepthCompareMean(fragments));

  // how to transform projected points to screen
  const Mat3 screenM(makeScreenM(fragments, x1, y1, x2, y2));

  double linescale = std::max(std::abs(x2-x1), std::abs(y2-y1)) * (1./1000);

  // finally draw items
  //printf("\ndoing drawing\n");
  doDrawing(painter, screenM, linescale);

  //printf("ended\n");

  //simpleDump();
  //objDump();
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
