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

Mesh::Mesh(const ValVector& _pos1, const ValVector& _pos2,
           const ValVector& _heights,
           Direction dirn,
           const LineProp* lprop, const SurfaceProp* sprop)
  : pos1(_pos1), pos2(_pos2), heights(_heights),
    lineprop(lprop), surfaceprop(sprop)
{
  switch(dirn)
    {
    default:
    case X_DIRN:
      hidx=0; didx1=1; didx2=2; break;
    case Y_DIRN:
      hidx=1; didx1=2; didx2=0; break;
    case Z_DIRN:
      hidx=2; didx1=0; didx2=1; break;
    }
}

void Mesh::getFragments(const Mat4& outerM, const Camera& cam,
                        FragmentVector& v) const
{
  getLineFragments(outerM, cam, v);
  getSurfaceFragments(outerM, cam, v);
}

void Mesh::getLineFragments(const Mat4& outerM, const Camera& cam,
                            FragmentVector& v) const
{
  if(lineprop.ptr() == 0)
    return;

  Fragment fl;
  fl.type = Fragment::FR_LINESEG;
  fl.surfaceprop = 0;
  fl.lineprop = lineprop.ptr();
  fl.object = const_cast<Mesh*>(this);

  const unsigned n2 = pos2.size();
  Vec4 pt(0,0,0,1);

  for(unsigned stepindex=0; stepindex<=1; ++stepindex)
    {
      const ValVector& stepvec = stepindex==0 ? pos1 : pos2;
      const ValVector& constvec = stepindex==0 ? pos2 : pos1;
      const unsigned stepdidx = stepindex==0 ? didx1 : didx2;
      const unsigned constdidx = stepindex==0 ? didx2 : didx1;

      for(unsigned consti=0; consti<constvec.size(); ++consti)
        {
          pt(constdidx) = constvec[consti];
          for(unsigned stepi=0; stepi<stepvec.size(); ++stepi)
            {
              double heightsval = heights[stepindex==0 ? stepi*n2+consti : consti*n2+stepi];
              pt(stepdidx) = stepvec[stepi];
              pt(hidx) = heightsval;
              Vec4 rpt = outerM*pt;

              // shuffle new to old positions and calculate new new
              fl.points[1] = fl.points[0];
              fl.points[0] = vec4to3(rpt);
              fl.proj[1] = fl.proj[0];
              fl.proj[0] = calcProjVec(cam.perspM, rpt);

              if(stepi > 0 && (fl.points[0]+fl.points[1]).isfinite())
                {
                  v.push_back(fl);
                  fl.bumpIndex();
                }
            }
        }
    }
}

void Mesh::getSurfaceFragments(const Mat4& outerM, const Camera& cam,
                               FragmentVector& v) const
{
  if(surfaceprop.ptr() == 0)
    return;

  Fragment fs;
  fs.type = Fragment::FR_TRIANGLE;
  fs.surfaceprop = surfaceprop.ptr();
  fs.lineprop = 0;
  fs.object = const_cast<Mesh*>(this);

  const unsigned n1 = pos1.size();
  const unsigned n2 = pos2.size();

  Vec4 p0, p1, p2, p3;
  p0(3) = p1(3) = p2(3) = p3(3) = 1;
  for(unsigned i1=0; (i1+1)<n1; ++i1)
    for(unsigned i2=0; (i2+1)<n2; ++i2)
      {
        // grid point coordinates
        p0(hidx) = heights[i1*n2+i2];
        p0(didx1) = pos1[i1];
        p0(didx2) = pos2[i2];
        p1(hidx) = heights[(i1+1)*n2+i2];
        p1(didx1) = pos1[i1+1];
        p1(didx2) = pos2[i2];
        p2(hidx) = heights[i1*n2+(i2+1)];
        p2(didx1) = pos1[i1];
        p2(didx2) = pos2[i2+1];
        p3(hidx) = heights[(i1+1)*n2+(i2+1)];
        p3(didx1) = pos1[i1+1];
        p3(didx2) = pos2[i2+1];

        if(! ((p0+p1+p2+p3).isfinite()) )
          continue;

        // these are converted to the outer coordinate system
        Vec4 rp0 = outerM*p0;
        Vec4 rp1 = outerM*p1;
        Vec4 rp2 = outerM*p2;
        Vec4 rp3 = outerM*p3;

        fs.points[1] = vec4to3(rp1);
        fs.points[2] = vec4to3(rp2);
        fs.proj[1] = calcProjVec(cam.perspM, rp1);
        fs.proj[2] = calcProjVec(cam.perspM, rp2);

        fs.points[0] = vec4to3(rp0);
        fs.proj[0] = calcProjVec(cam.perspM, rp0);
        v.push_back(fs);
        fs.bumpIndex();

        fs.points[0] = vec4to3(rp3);
        fs.proj[0] = calcProjVec(cam.perspM, rp3);
        v.push_back(fs);
        fs.bumpIndex();
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
