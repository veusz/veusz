#include "shapes.h"

Object::~Object()
{
}

void Triangle::getFragments(const Mat4& outerM, const Camera& cam,
			    FragmentVec& v) const
{
  Mat4 totM(cam.perspM*outerM*objM);

  Fragment f;
  f.type = Fragment::FR_TRIANGLE;
  f.points[0] = calcProjVec(totM, points[0]);
  f.points[1] = calcProjVec(totM, points[1]);
  f.points[2] = calcProjVec(totM, points[2]);
  f.object = const_cast<Triangle*>(this);

  v.push_back(f);
}

void PolyLine::getFragments(const Mat4& outerM, const Camera& cam,
			    FragmentVec& v) const
{
  if(points.size() < 2)
    return;

  Mat4 totM(cam.perspM*outerM*objM);

  Fragment f;
  f.type = Fragment::FR_LINESEG;
  f.object = const_cast<PolyLine*>(this);

  Vec3 lv = calcProjVec(totM, points[0]);
  for(unsigned i=1; i != points.size(); ++i)
    {
      f.points[0] = lv;
      f.points[1] = lv = calcProjVec(totM, points[i]);
      v.push_back(f);
    }
}
