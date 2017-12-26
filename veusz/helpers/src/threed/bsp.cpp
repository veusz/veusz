#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include "fragment.h"
#include "bsp.h"

#define EPS 1e-6

namespace
{
  // 2d triangle area squared (only considering X/Y)
  inline double triAreaSqd2D(const Vec3* pts)
  {
    double a = (pts[0](0)*pts[1](1)-pts[0](1)*pts[1](0)+
                pts[1](0)*pts[2](1)-pts[1](1)*pts[2](0)+
                pts[2](0)*pts[0](1)-pts[2](1)*pts[0](0));
    return a*a;
  }

  // Find set of three points to define a plane (pts).
  // Needs to find points which are not the same return true if ok
  bool findPlane(const IdxVector& idxs, unsigned startidx,
                 const FragmentVector& frags, Vec3* pts)
  {
    double maxtriarea2 = -1;
    unsigned besttri = EMPTY_BSP_IDX;

    Vec3 temppts[3];
    unsigned ptct = 0;
    double maxptsarea2 = -1;

    // Algorithm is to find the biggest triangle on the plane of the
    // image (X/Y), then if none, the set of 3 points with the largest
    // area. We prefer triangles as splitting them is expensive in
    // terms of numbers of fragments.

    // Plane of image triangles are preferred as splitting in the
    // opposite directions gives a weird viewing order for points

    // Using larger triangles is a heuristic to prevent lots of split
    // triangles. Empirically it seems to reduce the number of final
    // fragments by quite a lot.
    const unsigned endidx = idxs.size();
    for(unsigned i=startidx; i<endidx; ++i)
      {
        const Fragment& f = frags[idxs[i]];
        if(f.type == Fragment::FR_TRIANGLE)
          {
            double areasqd = triAreaSqd2D(f.points);
            if(areasqd > maxtriarea2)
              {
                maxtriarea2 = areasqd;
                besttri = i;
              }
          }
        else if(besttri == EMPTY_BSP_IDX)
          // only bother looking at other points if we haven't found a
          // triangle yet
          {
            // this is a crude rotating buffer looking for larger triangles
            // in the plane of the image
            if(f.type == Fragment::FR_LINESEG || f.type == Fragment::FR_PATH)
              temppts[(ptct++)%3] = f.points[0];
            if(f.type == Fragment::FR_LINESEG)
              temppts[(ptct++)%3] = f.points[1];
            if(ptct >= 3)
              {
                double areasqd = triAreaSqd2D(temppts);
                if(areasqd > maxptsarea2)
                  {
                    for(unsigned j=0; j<3; ++j)
                      pts[j] = temppts[j];
                    maxptsarea2 = areasqd;
                  }
              }
          }
      }

    // return triangle
    if(besttri != EMPTY_BSP_IDX)
      {
        for(unsigned i=0; i<3; ++i)
          pts[i] = frags[idxs[besttri]].points[i];
        return 1;
      }
    else
      {
        // is the returned triangle valid?
        return maxptsarea2 > EPS;
      }
  }

  // sign of calculated dot
  inline int dotsign(double dot)
  {
    return dot > EPS ? 1 : dot < -EPS ? -1 : 0;
  }

  // is path in front, on or behind plane?
  void handlePath(const Vec3& norm, const Vec3& plane0,
                  FragmentVector& v, unsigned fidx,
                  IdxVector& idxsame, IdxVector& idxfront, IdxVector& idxback)
  {
    int sign = dotsign(dot(norm, v[fidx].points[0]-plane0));
    switch(sign)
      {
      case 1: idxfront.push_back(fidx); break;
      case -1: idxback.push_back(fidx); break;
      default: idxsame.push_back(fidx); break;
      }
  }

  // is line in front, on or behind plane
  void handleLine(const Vec3& norm, const Vec3& plane0,
                  FragmentVector& fragvec, unsigned fidx,
                  IdxVector& idxsame, IdxVector& idxfront, IdxVector& idxback)
  {
    Fragment& f = fragvec[fidx];

    double dot0 = dot(norm, f.points[0]-plane0);
    int sign0 = dotsign(dot0);
    int sign1 = dotsign(dot(norm, f.points[1]-plane0));
    int signsum = sign0+sign1;

    // first cases are that the line is simply on one side
    if(sign0==0 && sign1==0)
      idxsame.push_back(fidx);
    else if(signsum > 0)
      idxfront.push_back(fidx);
    else if(signsum < 0)
      idxback.push_back(fidx);
    else
      {
        // split line. Note: we change original, then push a copy, as
        // a push invalidates the original reference
        Vec3 linevec = f.points[1]-f.points[0];
        double d = -dot0 / dot(linevec, norm);
        Vec3 newpt = f.points[0] + linevec*d;
        Fragment fcpy(f);

        // overwrite original with +ve part
        f.points[sign0 < 0 ? 0 : 1] = newpt;
        idxfront.push_back(fidx);

        // write copy with -ve part
        fcpy.points[sign0 < 0 ? 1 : 0] = newpt;
        idxback.push_back(fragvec.size());
        fragvec.push_back(fcpy);
      }
  }

  // is triangle in front, behind or on plane?
  void handleTriangle(const Vec3& norm, const Vec3& plane0,
                      FragmentVector& fragvec, unsigned fidx,
                      IdxVector& idxsame, IdxVector& idxfront, IdxVector& idxback)
  {
    Fragment& f = fragvec[fidx];

    double dots[3];
    int signs[3];
    for(unsigned i=0; i<3; ++i)
      {
        dots[i] = dot(norm, f.points[i]-plane0);
        signs[i] = dotsign(dots[i]);
      }
    int signsum = signs[0]+signs[1]+signs[2];
    int nzero = (signs[0]==0)+(signs[1]==0)+(signs[2]==0);

    if(nzero == 3)
      // all on plane
      idxsame.push_back(fidx);
    else if(signsum+nzero == 3)
      // all +ve or on plane
      idxfront.push_back(fidx);
    else if(signsum-nzero == -3)
      // all -ve or on plane
      idxback.push_back(fidx);
    else if(nzero == 1)
      {
        // split triangle into two as one point is on the plane and
        // the other two are either side
        // index of point on plane
        unsigned idx0 = signs[0]==0 ? 0 : signs[1]==0 ? 1 : 2;

        Vec3 linevec = f.points[(idx0+2)%3]-f.points[(idx0+1)%3];
        double d = -dots[(idx0+1)%3] / dot(linevec, norm);
        Vec3 newpt = f.points[(idx0+1)%3] + linevec*d;

        Fragment fcpy(f);

        // modify original
        f.points[(idx0+2)%3] = newpt;
        (dots[(idx0+1)%3]>0 ? idxfront : idxback).push_back(fidx);

        // then make a copy for the other side
        fcpy.points[(idx0+1)%3] = newpt;
        (dots[(idx0+2)%3]>0 ? idxfront : idxback).push_back(fragvec.size());
        fragvec.push_back(fcpy);
      }
    else
      {
        // nzero==0
        // split triangle into three, as no points are on the plane

        // point index by itself on one side of plane
        unsigned diffidx = signs[1]==signs[2] ? 0 : signs[0]==signs[2] ? 1 : 2;

        // new points on plane
        Vec3 linevec_p1 = f.points[(diffidx+1)%3]-f.points[diffidx];
        double d_p1 = -dots[diffidx] / dot(linevec_p1, norm);
        Vec3 newpt_p1 = f.points[diffidx] + linevec_p1*d_p1;
        Vec3 linevec_p2 = f.points[(diffidx+2)%3]-f.points[diffidx];
        double d_p2 = -dots[diffidx] / dot(linevec_p2, norm);
        Vec3 newpt_p2 = f.points[diffidx] + linevec_p2*d_p2;

        // now make one triangle on one side and two on the other
        Fragment fcpy1(f);
        Fragment fcpy2(f);

        // modify original: triangle by itself on one side
        f.points[(diffidx+1)%3] = newpt_p1;
        f.points[(diffidx+2)%3] = newpt_p2;
        (dots[diffidx] > 0 ? idxfront : idxback).push_back(fidx);

        // then add the other two on the other side
        fcpy1.points[diffidx] = newpt_p1;
        fcpy1.points[(diffidx+2)%3] = newpt_p2;
        (dots[diffidx] < 0 ? idxfront : idxback).push_back(fragvec.size());
        fragvec.push_back(fcpy1);
        fcpy2.points[diffidx] = newpt_p2;
        (dots[diffidx] < 0 ? idxfront : idxback).push_back(fragvec.size());
        fragvec.push_back(fcpy2);
      }
  }

  struct BSPStackItem
  {
    BSPStackItem(unsigned _bspidx, unsigned _nidxs)
      : bspidx(_bspidx), nidxs(_nidxs)
    {}
    unsigned bspidx; // BSPRecord we are working on here
    unsigned nidxs;  // Number of fragment indices in to_process
  };

  // get Z component of fragment, nudging points and lines forward
  // Z decreases away from viewer
  double fragZ(const Fragment& f)
  {
    switch(f.type)
      {
      case Fragment::FR_TRIANGLE:
	return std::min(f.points[0](2), std::min(f.points[1](2),
                                                 f.points[2](2)));
      case Fragment::FR_LINESEG:
	return std::min(f.points[0](2), f.points[1](2)) + 1e-5;
      case Fragment::FR_PATH:
	return f.points[0](2) + 2e-5;
      default:
	return std::numeric_limits<double>::infinity();
      }
  }

  struct FragZCompare
  {
    FragZCompare(const FragmentVector& _v)
      : v(_v)
    {}
    bool operator()(unsigned a, unsigned b)
    {
      return fragZ(v[a])<fragZ(v[b]);
    }
    const FragmentVector& v;
  };

}

// This is a non-recursive BSP building routines. Fragment indices to
// examine are built up on the to_process vector. A stack of
// BSPStackItem items is used to keep track which BSP record the
// fragment indices belong to.

BSPBuilder::BSPBuilder(FragmentVector& fragvec, Vec3 viewdirn)
{
  // initial record
  bsp_recs.reserve(fragvec.size());
  bsp_recs.push_back(BSPRecord());

  // add every non-empty fragment onto a list of fragments to process
  IdxVector to_process;
  to_process.reserve(fragvec.size()*2);
  for(unsigned i=0, s=fragvec.size(); i<s; ++i)
    {
      if(fragvec[i].type != Fragment::FR_NONE)
        to_process.push_back(i);
    }

  // these are where indices for the front and back side of the plane
  IdxVector idxback;
  IdxVector idxfront;
  idxback.reserve(fragvec.size());
  idxfront.reserve(fragvec.size());
  Vec3 planepts[3];

  // stack of items to process
  std::vector<BSPStackItem> stack;
  stack.reserve(128);
  stack.push_back( BSPStackItem(0, to_process.size()) );

  while( !stack.empty() )
    {
      BSPStackItem stackitem(stack.back());
      stack.pop_back();

      // this is the bsp record with which the items are associated
      BSPRecord& rec = bsp_recs[stackitem.bspidx];
      rec.minfragidxidx = frag_idxs.size(); // where the items get added

      // if more than item to process then choose a plane, then split
      if( stackitem.nidxs > 1 &&
          findPlane(to_process, to_process.size()-stackitem.nidxs,
                    fragvec, planepts) )
        {
          // norm of plane (making sure it points to observer)
          Vec3 norm = cross(planepts[1]-planepts[0], planepts[2]-planepts[0]);
          if(dot(norm, viewdirn) < 0)
            norm = -norm;
          // approximately normalise
          norm *= 1./(std::abs(norm(0))+std::abs(norm(1))+std::abs(norm(2)));

          unsigned to_process_size = to_process.size();
          for(unsigned i=to_process_size-stackitem.nidxs; i<to_process_size; ++i)
            {
              unsigned fidx = to_process[i];
              switch(fragvec[fidx].type)
                {
                  case Fragment::FR_PATH:
                  handlePath(norm, planepts[0], fragvec, fidx,
                             frag_idxs, idxfront, idxback);
                  break;
                case Fragment::FR_LINESEG:
                  handleLine(norm, planepts[0], fragvec, fidx,
                             frag_idxs, idxfront, idxback);
                  break;
                case Fragment::FR_TRIANGLE:
                  handleTriangle(norm, planepts[0], fragvec, fidx,
                                 frag_idxs, idxfront, idxback);
                  break;
                default:
                  break;
                }
            }

          // number added to this node
          rec.nfrags = frag_idxs.size()-rec.minfragidxidx;
          // remove items to process
          to_process.resize(to_process_size-stackitem.nidxs);

          if(rec.nfrags == 0)
            {
              if(idxfront.empty() && !idxback.empty())
                {
                  frag_idxs.insert(frag_idxs.end(), idxback.begin(), idxback.end());
                  rec.nfrags = idxback.size();
                  idxback.resize(0);
                }
              else if(idxback.empty() && !idxfront.empty())
                {
                  frag_idxs.insert(frag_idxs.end(), idxfront.begin(), idxfront.end());
                  rec.nfrags = idxfront.size();
                  idxfront.resize(0);
                }
            }

          // push_back invalidates rec, so we don't use it below
          if(!idxfront.empty())
            {
              unsigned newbspidx = bsp_recs.size();
              bsp_recs[stackitem.bspidx].frontidx = newbspidx;
              bsp_recs.push_back(BSPRecord());
              stack.push_back( BSPStackItem(newbspidx, idxfront.size()) );
              to_process.insert(to_process.end(), idxfront.begin(), idxfront.end());
              idxfront.resize(0);
            }

          if(!idxback.empty())
            {
              unsigned newbspidx = bsp_recs.size();
              bsp_recs[stackitem.bspidx].backidx = newbspidx;
              // add the record to be processed
              bsp_recs.push_back(BSPRecord());
              // new set of items to process
              stack.push_back( BSPStackItem(newbspidx, idxback.size()) );
              // and add onto to process list
              to_process.insert(to_process.end(), idxback.begin(), idxback.end());
              idxback.resize(0);
            }
        }
      else
        {
          // single item to process or plane couldn't be found
          frag_idxs.insert(frag_idxs.end(),
                           to_process.end()-stackitem.nidxs,
                           to_process.end());
          to_process.resize(to_process.size()-stackitem.nidxs);
          rec.nfrags = stackitem.nidxs;
        }

    } // while !stack.empty()
}

namespace
{
  // Similar stack item to keep track of the tree navigation while
  // "drawing" the tree
  struct WalkStackItem
  {
    WalkStackItem(unsigned _idx, unsigned _stage)
      : bsp_idx(_idx), stage(_stage)
    {}

    unsigned bsp_idx;
    unsigned stage;
  };
};

// This is a non-recursive function to walk the tree. We keep a
// "stack" to do the walking. Because we have to walk the back before
// the current items and then the front, we have two types of stack
// items: WALK_START and WALK_RECS

IdxVector BSPBuilder::getFragmentIdxs(const FragmentVector& fragvec) const
{
  IdxVector retn;

  std::vector<WalkStackItem> stack;
  stack.reserve(128);
  stack.push_back(WalkStackItem(0, 0));

  IdxVector temp;

  while( !stack.empty() )
    {
      WalkStackItem stackitem(stack.back());
      stack.pop_back();

      const BSPRecord &rec = bsp_recs[stackitem.bsp_idx];

      if(stackitem.stage == 0)
        {
          if(rec.frontidx != EMPTY_BSP_IDX)
            stack.push_back( WalkStackItem(rec.frontidx, 0) );
          stack.push_back( WalkStackItem(stackitem.bsp_idx, 1) );
          if(rec.backidx != EMPTY_BSP_IDX)
            stack.push_back( WalkStackItem(rec.backidx, 0) );
        }
      else
        {
          // Sort images in plane by Z. This is helpful for points
          // which may overlap.
          temp.resize(0);
          temp.insert(temp.end(),
                      frag_idxs.begin()+rec.minfragidxidx,
                      frag_idxs.begin()+(rec.minfragidxidx+rec.nfrags));

          std::sort(temp.begin(), temp.end(), FragZCompare(fragvec));

          for(int type=Fragment::FR_TRIANGLE; type<=Fragment::FR_PATH; ++type)
            {
              for(unsigned i : temp)
                if(fragvec[i].type == type)
                  retn.push_back(i);
            }
        }
    }

  return retn;
}

#if 0
#include <iostream>
int main()
{
  FragmentVector v;

  for(unsigned i=0; i<10; ++i)
    {
      Fragment f;
      f.type =Fragment::FR_TRIANGLE;

      for(unsigned j=0;j<3;j++)
        f.points[j] = Vec3(rand()*1./RAND_MAX, rand()*1./RAND_MAX, rand()*1./RAND_MAX);

      v.push_back(f);
    }

  BSPBuilder builder(v);

  std::cout << "BSP recs size " << builder.bsp_recs.size() << '\n';
  std::cout << "Fragment size " << v.size() << '\n';

  IdxVector out = builder.getFragmentIdxs();
  for(unsigned i=0; i<out.size(); ++i)
    std::cout << ' ' << out[i];
  std::cout << '\n';

  return 0;
};
#endif
