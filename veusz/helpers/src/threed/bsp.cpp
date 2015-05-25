#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include "fragment.h"
#include "bsp.h"

#include <iostream>

#define EPS 1e-5

namespace
{

  double fragZ(const Fragment& f)
  {
    switch(f.type)
      {
      case Fragment::FR_PATH:
        return f.points[0](2);
      case Fragment::FR_LINESEG:
        return (f.points[0](2)+f.points[1](2))*(1./2);
      case Fragment::FR_TRIANGLE:
        return (f.points[0](2)+f.points[1](2)+f.points[2](2))*(1./3);
      default:
        return std::numeric_limits<double>::max();
      }
  }

  struct FragZCompareMin
  {
    FragZCompareMin(const FragmentVector& v)
      : vec(v)
    {}
    bool operator()(unsigned i, unsigned j) const
    {
      return fragZ(vec[i]) > fragZ(vec[j]);
    }
    const FragmentVector& vec;    
  };


  // struct FragDepthCompareMin
  // {
  //   FragDepthCompareMin(FragmentVector& v)
  //     : vec(v)
  //   {}
  //   bool operator()(unsigned i, unsigned j) const
  //   {
  //     return vec[i].minDepth() > vec[j].minDepth();
  //   }
  //   FragmentVector& vec;
  // };

  inline bool close(const Vec3& p1, const Vec3& p2)
  {
    return (std::abs(p1(0)-p2(0))<EPS && std::abs(p1(1)-p2(1))<EPS &&
            std::abs(p1(2)-p2(2))<EPS);
  }

  inline bool badNormal(const Vec3* pts)
  {
    double d1x = pts[1](0)-pts[0](0);
    double d1y = pts[1](1)-pts[0](1);
    double d2x = pts[2](0)-pts[0](0);
    double d2y = pts[2](1)-pts[0](1);
    return std::abs(d1x*d2y-d2x*d1y) < EPS;
  }

  // find set of three points to define a plane
  // needs to find points which are not the same
  // return 1 if ok
  bool findPlane(const IdxVector& idxs, FragmentVector& v,
                 Vec3* pts)
  {
    const unsigned size = idxs.size();
    const unsigned centre = size/2;

    unsigned totpts = 0;
    unsigned ptct = 0;
    for(unsigned delta=0; delta<=centre; ++delta)
      {
        if(centre+delta < size)
          {
            const Fragment& f = v[idxs[centre+delta]];
            for(unsigned i=0; i<f.nPoints(); ++i)
              {
                bool diff = 1;
                for(unsigned j=0; j<ptct && diff; ++j)
                  if(close(f.points[i], pts[j]))
                    diff = 0;
                if(diff)
                  {
                    pts[ptct++] = f.points[i];
                    if(ptct==3)
                      {
                        if(badNormal(pts))
                          ptct--;
                        else
                          return 1;
                      }
                  }
              }
            totpts += f.nPoints();
          }
        if(delta > 0)
          {
            const Fragment& f = v[idxs[centre-delta]];
            for(unsigned i=0; i<f.nPoints(); ++i)
              {
                bool diff = 1;
                for(unsigned j=0; j<ptct && diff; ++j)
                  if(close(f.points[i], pts[j]))
                    diff = 0;
                if(diff)
                  {
                    pts[ptct++] = f.points[i];
                    if(ptct==3)
                      {
                        if(badNormal(pts))
                          ptct--;
                        else
                          return 1;
                      }
                  }
              }
          }
      }
    return 0;
  }

  inline int dotsign(double dot)
  {
    return dot > EPS ? 1 : dot < -EPS ? -1 : 0;
  }

  void handlePath(const Vec3& norm, const Vec3& plane0,
                  FragmentVector& v, unsigned fidx,
                  IdxVector& idxsame, IdxVector& idxmore, IdxVector& idxless)
  {
    Fragment& f = v[fidx];
    int sign = dotsign(dot(norm, f.points[0]-plane0));
    switch(sign)
      {
      case 1: idxmore.push_back(fidx); break;
      case -1: idxless.push_back(fidx); break;
      default: idxsame.push_back(fidx); break;
      }
  }

  void handleLine(const Vec3& norm, const Vec3& plane0,
                  FragmentVector& fragvec, unsigned fidx,
                  IdxVector& idxsame, IdxVector& idxmore, IdxVector& idxless)
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
      idxmore.push_back(fidx);
    else if(signsum < 0)
      idxless.push_back(fidx);
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
        idxmore.push_back(fidx);

        // write copy with -ve part
        fcpy.points[sign0 < 0 ? 1 : 0] = newpt;
        idxless.push_back(fragvec.size());
        fragvec.push_back(fcpy);
      }
  }

  void handleTriangle(const Vec3& norm, const Vec3& plane0,
                      FragmentVector& fragvec, unsigned fidx,
                      IdxVector& idxsame, IdxVector& idxmore, IdxVector& idxless)
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

    //std::cout << "signsum " << signsum << " nzero " << nzero << '\n';

    if(nzero == 3)
      // all on plane
      idxsame.push_back(fidx);
    else if(signsum+nzero == 3)
      // all +ve or on plane
      idxmore.push_back(fidx);                
    else if(signsum-nzero == -3)
      // all -ve or on plane
      idxless.push_back(fidx);
    else if(nzero == 1)
      {
        //std::cout << "Split in two\n";

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
        (dots[(idx0+1)%3]>0 ? idxmore : idxless).push_back(fidx);

        // then make a copy for the other side
        fcpy.points[(idx0+1)%3] = newpt;
        (dots[(idx0+2)%3]>0 ? idxmore : idxless).push_back(fragvec.size());
        fragvec.push_back(fcpy);
      }
    else
      {
        //std::cout << "Split in 3\n";

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
        (dots[diffidx] > 0 ? idxmore : idxless).push_back(fidx);

        // then add the other two on the other side
        fcpy1.points[diffidx] = newpt_p1;
        fcpy1.points[(diffidx+2)%3] = newpt_p2;
        (dots[diffidx] < 0 ? idxmore : idxless).push_back(fragvec.size());
        fragvec.push_back(fcpy1);
        fcpy2.points[diffidx] = newpt_p2;
        (dots[diffidx] < 0 ? idxmore : idxless).push_back(fragvec.size());
        fragvec.push_back(fcpy2);
      }
  }

  void split(const IdxVector& idxs, BSPNode* node, FragmentVector& fragvec)
  {
    Vec3 planepts[3];
    if(!findPlane(idxs, fragvec, planepts))
      {
        // can't find a place to split the points, so we stick
        // everything into the final node
        node->more = node->less = 0;
        node->fragidxs = idxs;
        return;
      }

    // for(unsigned i=0; i<3; ++i)
    //   {
    //     std::cout << "plane " << i << ' '
    //               << planepts[i](0) << ' ' << planepts[i](1) << ' ' << planepts[i](2) << '\n';
    //   }

    Vec3 norm = cross(planepts[1]-planepts[0], planepts[2]-planepts[0]);

    // make sure normal is pointing towards viewer
    // FIXME: if normal is perp, what happens?
    if(norm(2) < 0)
      {
        norm(0) = -norm(0);
        norm(1) = -norm(1);
        norm(2) = -norm(2);
      }

    //std::cout << "Norm " << norm(0) << ' ' << norm(1) << ' ' << norm(2) << '\n';

    IdxVector idxless;
    IdxVector idxmore;
    for(unsigned i=0, s=idxs.size(); i<s; ++i)
      {
        unsigned fidx = idxs[i];
        switch(fragvec[fidx].type)
          {
          case Fragment::FR_PATH:
            handlePath(norm, planepts[0], fragvec, fidx,
                       node->fragidxs, idxmore, idxless);
            break;

          case Fragment::FR_LINESEG:
            handleLine(norm, planepts[0], fragvec, fidx,
                       node->fragidxs, idxmore, idxless);
            break;

          case Fragment::FR_TRIANGLE:
            handleTriangle(norm, planepts[0], fragvec, fidx,
                           node->fragidxs, idxmore, idxless);
            break;

          default:
            break;
          }
      }

    if(!idxless.empty())
      {
        BSPNode* newless = new BSPNode;
        std::sort(idxless.begin(), idxless.end(), FragZCompareMin(fragvec));
        split(idxless, newless, fragvec);
        node->less = newless;
      }
    else
      node->less = 0;

    if(!idxmore.empty())
      {
        BSPNode* newmore = new BSPNode;
        std::sort(idxmore.begin(), idxmore.end(), FragZCompareMin(fragvec));
        split(idxmore, newmore, fragvec);
        node->more = newmore;
      }
    else
      node->more = 0;
  }

}

BSPNode* buildBSPTree(FragmentVector& fragvec)
{
  std::vector<unsigned> idxs;
  idxs.reserve(fragvec.size());
  for(unsigned i=0, s=fragvec.size(); i<s; ++i)
    {
      if(fragvec[i].type != Fragment::FR_NONE)
        idxs.push_back(i);
    }

  std::sort(idxs.begin(), idxs.end(), FragZCompareMin(fragvec));

  BSPNode* root = new BSPNode;
  split(idxs, root, fragvec);

  return root;
};

void deleteBSPTree(BSPNode *node)
{
  if(node->more)
    deleteBSPTree(node->more);
  if(node->less)
    deleteBSPTree(node->less);
  delete node;
}

void printTree(BSPNode *node, unsigned level)
{
  std::cout << level << " leaf:\n";
  for(unsigned i=0; i<node->fragidxs.size(); ++i)
    std::cout << ' ' << node->fragidxs[i];
  std::cout << '\n';

  std::cout << level << " more:\n";
  if(node->more)
    printTree(node->more, level+1);
  std::cout << level << " less:\n";
  if(node->less)
    printTree(node->less, level+1);
}

void printfragments(FragmentVector& v)
{
  for(unsigned i=0; i<v.size(); ++i)
    {
      for(unsigned j=0;j<4;++j)
        {
          Vec3 p = v[i].points[j%3];
          std::cout << p(0) << ' ' << p(1) << ' ' << p(2) << '\n';
        }
      std::cout << "\n\n";
    }
}

#if 0
int main()
{
  FragmentVector v;

  Fragment f;
  f.type = Fragment::FR_TRIANGLE;

  for(unsigned i=0; i<20; ++i)
    {
      f.points[0] = Vec3(0,i,i);
      f.points[1] = Vec3(0,i+1,i);
      f.points[2] = Vec3(0,i,1+i);
      v.push_back(f);
    }

  BSPNode* root = buildBSPTree(v);

  //std::cout << v.size() << '\n';

  //printTree(root, 0);
  printfragments(v);

  deleteTree(root);
}
#endif
