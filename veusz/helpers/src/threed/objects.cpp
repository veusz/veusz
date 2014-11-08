#include "objects.h"

Object::~Object()
{
}

void Triangle::getFragments(const Mat4& outerM, const Camera& cam,
			    FragmentVector& v) const
{
  Fragment f;
  f.type = Fragment::FR_TRIANGLE;
  f.surfaceprop = surfaceprop;
  f.lineprop = 0;
  f.points[0] = outerM*points[0];
  f.proj[0] = calcProjVec(cam.perspM, f.points[0]);
  f.points[1] = outerM*points[1];
  f.proj[1] = calcProjVec(cam.perspM, f.points[1]);
  f.points[2] = outerM*points[2];
  f.proj[2] = calcProjVec(cam.perspM, f.points[2]);
  f.object = const_cast<Triangle*>(this);

  v.push_back(f);
}

void PolyLine::getFragments(const Mat4& outerM, const Camera& cam,
			    FragmentVector& v) const
{
  if(points.size()<2)
    return;

  Fragment f;
  f.type = Fragment::FR_LINESEG;
  f.surfaceprop = 0;
  f.lineprop = lineprop;
  f.object = const_cast<PolyLine*>(this);

  // iterators use many more instructions here...
  Vec4 lp3d = outerM*points[0];
  Vec3 lpproj = calcProjVec(cam.perspM, lp3d);
  for(unsigned i=1, s=points.size(); i<s; ++i)
    {
      f.points[0] = lp3d;
      f.proj[0] = lpproj;
      f.points[1] = lp3d = outerM*points[i];
      f.proj[1] = lpproj = calcProjVec(cam.perspM, lp3d);
      v.push_back(f);
    }
}

ObjectContainer::~ObjectContainer()
{
  for(unsigned i=0, s=objects.size(); i<s; ++i)
    delete objects[i];
}

void ObjectContainer::getFragments(const Mat4& outerM, const Camera& cam,
				   FragmentVector& v) const
{
  Mat4 totM(outerM*objM);
  for(unsigned i=0, s=objects.size(); i<s; ++i)
    objects[i]->getFragments(totM, cam, v);
}
