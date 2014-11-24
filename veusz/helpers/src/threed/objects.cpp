#include "objects.h"

Object::~Object()
{
}

void Object::getFragments(const Mat4& outerM, const Camera& cam,
			  FragmentVector& v) const
{
}

void Triangle::getFragments(const Mat4& outerM, const Camera& cam,
			    FragmentVector& v) const
{
  Fragment f;
  f.type = Fragment::FR_TRIANGLE;
  f.surfaceprop = surfaceprop.ptr();
  f.lineprop = 0;
  Vec4 p0 = outerM*points[0];
  f.points[0] = vec4to3(p0);
  f.proj[0] = calcProjVec(cam.perspM, p0);
  Vec4 p1 = outerM*points[1];
  f.points[1] = vec4to3(p1);
  f.proj[1] = calcProjVec(cam.perspM, p1);
  Vec4 p2 = outerM*points[2];
  f.points[2] = vec4to3(p2);
  f.proj[2] = calcProjVec(cam.perspM, p2);
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
  f.lineprop = lineprop.ptr();
  f.object = const_cast<PolyLine*>(this);

  // iterators use many more instructions here...
  Vec4 lp3d = outerM*points[0];
  Vec3 lpproj = calcProjVec(cam.perspM, lp3d);
  lpproj(2) -= 1e-4;
  for(unsigned i=1, s=points.size(); i<s; ++i)
    {
      f.points[0] = vec4to3(lp3d);
      f.proj[0] = lpproj;
      lp3d = outerM*points[i];
      f.points[1] = vec4to3(lp3d);
      lpproj = calcProjVec(cam.perspM, lp3d);
      f.proj[1] = lpproj;
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
