#include "objects.h"

Object::~Object()
{
}

void Triangle::getFragments(const Mat4& outerM, const Camera& cam,
			    FragmentVector& v) const
{
  Mat4 totM(outerM*objM);

  Fragment f;
  f.type = Fragment::FR_TRIANGLE;
  f.surfaceprop = surfaceprop;
  f.lineprop = 0;
  f.points3d[0] = totM*points[0];
  f.projpoints[0] = calcProjVec(cam.perspM, f.points3d[0]);
  f.points3d[1] = totM*points[1];
  f.projpoints[1] = calcProjVec(cam.perspM, f.points3d[1]);
  f.points3d[2] = totM*points[2];
  f.projpoints[2] = calcProjVec(cam.perspM, f.points3d[2]);
  f.object = const_cast<Triangle*>(this);

  v.push_back(f);
}

void PolyLine::getFragments(const Mat4& outerM, const Camera& cam,
			    FragmentVector& v) const
{
  if(points.size()<2)
    return;

  Mat4 totM(outerM*objM);

  Fragment f;
  f.type = Fragment::FR_LINESEG;
  f.surfaceprop = 0;
  f.lineprop = lineprop;
  f.object = const_cast<PolyLine*>(this);

  // iterators use many more instructions here...
  Vec4 lp3d = totM*points[0];
  Vec3 lpproj = calcProjVec(cam.perspM, lp3d);
  for(unsigned i=1, s=points.size(); i<s; ++i)
    {
      f.points3d[0] = lp3d;
      f.projpoints[0] = lpproj;
      f.points3d[1] = lp3d = totM*points[i];
      f.projpoints[1] = lpproj = calcProjVec(cam.perspM, lp3d);
      v.push_back(f);
    }
}
