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

#include "clipcontainer.h"

#define EPS 1e-8

namespace
{

  // clip line by plane
  void clipLine(Fragment& f, const Vec3& onplane, const Vec3& normal)
  {
    double dot0 = dot(f.points[0]-onplane, normal);
    bool bad0 = dot0 < -EPS;
    double dot1 = dot(f.points[1]-onplane, normal);
    bool bad1 = dot1 < -EPS;

    if(!bad0 && !bad1)
      // both points on good side of plane
      ;
    else if(bad0 && bad1)
      // both line points on bad side of plane
      f.type = Fragment::FR_NONE;
    else
      {
        // clip line to plane (bad1 or bad2, not both)
        Vec3 linevec = f.points[1]-f.points[0];
        double d = -dot0 / dot(linevec, normal);
        f.points[bad0 ? 0 : 1] = f.points[0] + linevec*d;
      }
  }

  // clip triangle by plane
  void clipTriangle(FragmentVector& v, unsigned idx,
                    const Vec3& onplane, const Vec3& normal)
  {
    Fragment& f = v[idx];
    double dotv[3];
    unsigned bad[3];
    for(unsigned i=0; i<3; ++i)
      {
        dotv[i] = dot(f.points[i]-onplane, normal);
        bad[i] = dotv[i] < -EPS;
      }
    unsigned badsum = bad[0]+bad[1]+bad[2];

    switch(badsum)
      {
      case 0:
        // all points ok
        break;
      case 1:
        // two points are good, one is bad
        {
          unsigned badidx = bad[0] ? 0 : bad[1] ? 1 : 2;

          // calculate where vectors from good to bad points
          // intercept plane
          Vec3 good1 = f.points[(badidx+1)%3];
          Vec3 linevec1 = good1 - f.points[badidx];
          double d1 = -dotv[badidx] / dot(linevec1, normal);
          Vec3 icept1 = f.points[badidx] + linevec1*d1;

          Vec3 good2 = f.points[(badidx+2)%3];
          Vec3 linevec2 = good2 - f.points[badidx];
          double d2 = -dotv[badidx] / dot(linevec2, normal);
          Vec3 icept2 = f.points[badidx] + linevec2*d2;

          // break into two triangles from good points to intercepts
          // note: the push back invalidates the original, so we have
          // to make a copy
          f.points[0] = good2;
          f.points[1] = icept2;
          f.points[2] = good1;
          Fragment fcpy(f);
          fcpy.points[0] = good1;
          fcpy.points[1] = icept1;
          fcpy.points[2] = icept2;
          v.push_back(fcpy);
        }
        break;
      case 2:
        // one point is ok, the other two are bad
        {
          unsigned goodidx = !bad[0] ? 0 : !bad[1] ? 1 : 2;

          // work out where vectors from ok point intercept with plane
          Vec3 linevec1 = f.points[(goodidx+1)%3] - f.points[goodidx];
          double d1 = -dotv[goodidx] / dot(linevec1, normal);
          f.points[(goodidx+1)%3] = f.points[goodidx] + linevec1*d1;

          Vec3 linevec2 = f.points[(goodidx+2)%3] - f.points[goodidx];
          double d2 = -dotv[goodidx] / dot(linevec2, normal);
          f.points[(goodidx+2)%3] = f.points[goodidx] + linevec2*d2;
        }
        break;
      case 3:
        // all points are bad
        f.type = Fragment::FR_NONE;
        break;
      }
  }

  // clip all fragments to the plane given
  void clipFragments(FragmentVector& v, unsigned start,
                     const Vec3& onplane, const Vec3& normal)
  {
    unsigned nfrags = v.size();
    for(unsigned i=start; i<nfrags; ++i)
      {
        Fragment& f = v[i];
        switch(f.type)
          {
          case Fragment::FR_PATH:
            // point on wrong side of plane
            if(dot(f.points[0]-onplane, normal) < -EPS)
              f.type = Fragment::FR_NONE;
            break;

          case Fragment::FR_LINESEG:
            clipLine(f, onplane, normal);
            break;

          case Fragment::FR_TRIANGLE:
            clipTriangle(v, i, onplane, normal);
            break;

          default:
            break;
          }
      }
  }

} // namespace


void ClipContainer::getFragments(const Mat4& perspM, const Mat4& outerM, FragmentVector& v)
{
  // get fragments for children (and range in vector)
  const unsigned fragstart = v.size();
  for(unsigned i=0, s=objects.size(); i<s; ++i)
    objects[i]->getFragments(perspM, outerM, v);

  // these are the points defining the clipping cube
  Vec3 pts[8];
  pts[0] = minpt;
  pts[1] = Vec3(minpt(0), minpt(1), maxpt(2));
  pts[2] = Vec3(minpt(0), maxpt(1), minpt(2));
  pts[3] = Vec3(minpt(0), maxpt(1), maxpt(2));
  pts[4] = Vec3(maxpt(0), minpt(1), minpt(2));
  pts[5] = Vec3(maxpt(0), minpt(1), maxpt(2));
  pts[6] = Vec3(maxpt(0), maxpt(1), minpt(2));
  pts[7] = maxpt;

  // convert cube coordinates to outer coordinates
  for(unsigned i=0; i<8; ++i)
    pts[i] = vec4to3(outerM*vec3to4(pts[i]));

  // clip with plane point and normal
  // dotting points with plane with these will give all >= 0 if in cube
  clipFragments(v, fragstart, pts[0], cross(pts[2]-pts[0], pts[1]-pts[0]));
  clipFragments(v, fragstart, pts[0], cross(pts[1]-pts[0], pts[4]-pts[0]));
  clipFragments(v, fragstart, pts[0], cross(pts[4]-pts[0], pts[2]-pts[0]));
  clipFragments(v, fragstart, pts[7], cross(pts[5]-pts[7], pts[3]-pts[7]));
  clipFragments(v, fragstart, pts[7], cross(pts[3]-pts[7], pts[6]-pts[7]));
  clipFragments(v, fragstart, pts[7], cross(pts[6]-pts[7], pts[5]-pts[7]));
}
