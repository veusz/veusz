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

// gamma value used in calculating brightness for lighting
#define GAMMA 2.2

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
	for(unsigned p=0, np=f->nPointsVisible(); p<np; ++p)
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

  // convert (x,y,depth) -> screen coordinates
  QPointF vecToScreen(const Mat3& screenM, const Vec3& vec)
  {
    Vec3 mult(screenM*Vec3(vec(0), vec(1), 1));
    double inv = 1/mult(2);
    return QPointF(mult(0)*inv, mult(1)*inv);
  }

  unsigned init_fragments_size = 512;

}; // namespace

QPen Scene::LineProp2QPen(const LineProp* p, double linescale,
                          unsigned colindex) const
{
  if(p==0 || p->hide)
    return QPen(Qt::NoPen);
  else
    {
      QColor col;
      if(p->hasRGBs())
        col = QColor::fromRgba
          ( p->rgbs[std::min(unsigned(p->rgbs.size())-1,colindex)] );
      else
        col = QColor(int(p->r*255), int(p->g*255),
                     int(p->b*255), int((1-p->trans)*255));
      return QPen(QBrush(col), p->width*linescale);
    }
}

QBrush Scene::SurfaceProp2QBrush(const SurfaceProp* p,
                                 const Fragment& frag) const
{
  if(p==0 || p->hide)
    return QBrush();
  else
    {
      double r, g, b, a;

      if(p->hasRGBs())
        {
          QColor col = QColor::fromRgba
            ( p->rgbs[std::min(unsigned(p->rgbs.size())-1,frag.index)] );
          r=col.redF(); g=col.greenF(); b=col.blueF(); a=col.alphaF();
        }
      else
        {
          r=p->r; g=p->g; b=p->b; a=1-p->trans;
        }

      // use directional lighting
      if(lighton)
        {
          double scale = 1 - p->specular * (1-frag.lighting);
          scale = std::pow(scale, 1./GAMMA);
          r*=scale; g*=scale; b*=scale;
        }

      return QBrush(QColor(int(r*255), int(g*255), int(b*255), int(a*255)));
    }
}

void Scene::drawPath(QPainter* painter, const Fragment& frag,
                     QPointF pt1, QPointF pt2, double linescale)
{
  FragmentPathParameters* pars =
    static_cast<FragmentPathParameters*>(frag.params);
  double scale = frag.pathsize*linescale;

  // hook into drawing routine
  if(pars->runcallback)
    {
      pars->callback(painter, pt1, pt2, frag.index, scale, linescale);
      return;
    }

  if(pars->scaleedges)
    {
      painter->save();
      painter->translate(pt1.x(), pt1.y());
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
                                    el.x*scale+pt1.x(),
                                    el.y*scale+pt1.y());
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

  for(unsigned i=0, s=draworder.size(); i<s; ++i)
    {
      const Fragment& frag(fragments[draworder[i]]);

      // convert projected points to screen
      for(unsigned pi=0, s=frag.nPointsTotal(); pi<s; ++pi)
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
              if(lsurf != frag.surfaceprop ||
                 (frag.surfaceprop!=0 && (frag.surfaceprop->hasRGBs() ||
                                          (lighton && frag.surfaceprop->specular!=0))))
                {
                  lsurf = frag.surfaceprop;
                  painter->setBrush(SurfaceProp2QBrush(lsurf, frag));
                }

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
              if(lline != frag.lineprop ||
                 ((frag.lineprop!=0 && frag.lineprop->hasRGBs())))
                {
                  lline = frag.lineprop;
                  painter->setPen(LineProp2QPen(lline, linescale, frag.index));
                }
              painter->drawLine(projpts[0], projpts[1]);
            }
          break;

	case Fragment::FR_PATH:
            {
              if(lline != frag.lineprop ||
                 ((frag.lineprop!=0 && frag.lineprop->hasRGBs())))
                {
                  lline = frag.lineprop;
                  painter->setPen(LineProp2QPen(lline, linescale, frag.index));
                }
              if(lsurf != frag.surfaceprop ||
                 (frag.surfaceprop!=0 && (frag.surfaceprop->hasRGBs() ||
                                          (lighton && frag.surfaceprop->specular!=0))))
                {
                  lsurf = frag.surfaceprop;
                  painter->setBrush(SurfaceProp2QBrush(lsurf, frag));
                }
              drawPath(painter, frag, projpts[0], projpts[1], linescale);
            }
	  break;

	default:
	  break;
	}
    }
}

void Scene::calcLighting()
{
  // lighting is full on
  if(!lighton)
    return;

  for(unsigned i=0, s=fragments.size(); i<s; ++i)
    {
      Fragment& f = fragments[i];
      if(f.type == Fragment::FR_TRIANGLE)
        {
          // Calculate triangle norm. Make sure norm points towards
          // the viewer @ (0,0,0)
          Vec3 tripos = (f.points[0]+f.points[1]+f.points[2])*(1./3.);
          Vec3 norm = cross(f.points[1]-f.points[0], f.points[2]-f.points[0]);
          if(dot(tripos, norm)<0)
            norm = -norm;
          norm.normalise();

          // Now dot vector from light source to triangle with norm
          Vec3 light2tri = tripos-lightposn;
          light2tri.normalise();

          f.lighting = std::max(0., dot(light2tri, norm));
        }
    }
}

void Scene::projectFragments(const Camera& cam)
{
  // convert 3d to 2d coordinates using the Camera
  for(unsigned i=0, s=fragments.size(); i<s; ++i)
    {
      Fragment& f = fragments[i];
      for(unsigned pi=0, np=f.nPointsTotal(); pi<np; ++pi)
        f.proj[pi] = calcProjVec(cam.perspM, f.points[pi]);
    }
}

void Scene::renderPainters(const Camera& cam)
{
  calcLighting();
  projectFragments(cam);

  // simple painter's algorithm
  draworder.reserve(fragments.size());
  for(unsigned i=0; i<fragments.size(); ++i)
    draworder.push_back(i);

  std::sort(draworder.begin(), draworder.end(),
            FragDepthCompareMax(fragments));
}

void Scene::renderBSP(const Camera& cam)
{
  calcLighting();

  //std::cout << "\nFragment size 1 " << fragments.size() << '\n';

  BSPBuilder bsp(fragments, Vec3(0,0,1));
  draworder = bsp.getFragmentIdxs(fragments);

  //std::cout << "BSP recs size " << bsp.bsp_recs.size() << '\n';
  //std::cout << "Fragment size 2 " << fragments.size() << '\n';

  projectFragments(cam);
}

void Scene::render(Object* root,
                   QPainter* painter, const Camera& cam,
		   double x1, double y1, double x2, double y2)
{
  fragments.reserve(init_fragments_size);
  fragments.resize(0);
  draworder.resize(0);

  // get fragments for whole scene
  root->getFragments(cam.viewM, fragments);

  switch(mode)
    {
    case RENDER_BSP:
      renderBSP(cam);
      break;
    case RENDER_PAINTERS:
      renderPainters(cam);
      break;
    default:
      break;
    }

  // how to transform projected points to screen
  const Mat3 screenM(makeScreenM(fragments, x1, y1, x2, y2));

  double linescale = std::max(std::abs(x2-x1), std::abs(y2-y1)) * (1./1000);

  // finally draw items
  doDrawing(painter, screenM, linescale);

  // don't decrease size of fragments unnecessarily, unless it is large
  init_fragments_size = fragments.size();
  if(init_fragments_size > 65536)
    init_fragments_size /= 2;
}
