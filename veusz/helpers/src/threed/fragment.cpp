#include <stdio.h>

#include <algorithm>
#include <cmath>
#include "mmaths.h"
#include "fragment.h"

#define EPS 1e-5f

// Based on Thomas Moller's Public Domain code
// See article "A Fast Triangle-Triangle Intersection Test",
// Journal of Graphics Tools, 2(2), 1997

namespace
{
  inline float trimEps(float v)
  {
    return std::abs(v) < EPS ? 0 : v;
  }

  // calculate intersection of point with axis
  void iSectPts(const Vec3& VTX0, const Vec3& VTX1, const Vec3& VTX2,
		float VV0, float VV1, float VV2,
		float D0, float D1, float D2,
		float *isect0, float *isect1,
		Vec3* isectpoint0, Vec3* isectpoint1)
  {
    float tmp1=D0/(D0-D1);
    *isect0 = VV0+(VV1-VV0)*tmp1;
    Vec3 diff1 = (VTX1-VTX0)*tmp1;
    *isectpoint0 = VTX0 + diff1;

    float tmp2=D0/(D0-D2);
    *isect1 = VV0+(VV2-VV0)*tmp2;
    Vec3 diff2 = (VTX2-VTX0)*tmp2;
    *isectpoint1 = VTX0 + diff2;
  }

  // return 0 if f1<f2, else 1 and swap f1, f2
  inline unsigned order(float& f1, float& f2)
  {
    if(f1<f2)
      return 0;
    std::swap(f1, f2);
    return 1;
  }

  // return index of maximum absolute component
  inline unsigned indexMaxAbs(const Vec3& v)
  {
    float max=std::abs(v(0));
    unsigned idx=0;
    float b=std::abs(v(1));
    if(b>max) { max=b; idx=1; }
    float c=std::abs(v(2));
    if(c>max) { max=c; idx=2; }
    return idx;
  }

  bool computeISectInterval(const Vec3& VERT0, const Vec3& VERT1,
			    const Vec3& VERT2,
			    const Vec3& VV, const Vec3& D,
			    float D0D1, float D0D2,
			    float *isect0, float *isect1,
			    Vec3* isectpoint0, Vec3* isectpoint1)
  {
    if(D0D1>0)
      {
	// here we know that D0D2<=0.0 that is D(0), D(1) are on the
	// same side, D(2) on the other or on the plane
	iSectPts(VERT2, VERT0, VERT1, VV(2), VV(0), VV(1), D(2), D(0), D(1),
		 isect0, isect1, isectpoint0, isectpoint1);
      }
    else if(D0D2>0)
      {
	// here we know that d0d1<=0.0
	iSectPts(VERT1, VERT0, VERT2, VV(1), VV(0), VV(2), D(1), D(0), D(2),
		 isect0, isect1, isectpoint0, isectpoint1);
      }
    else if(D(1)*D(2)>0 || D(0)!=0)
      {
	// here we know that d0d1<=0.0 or that D(0)!=0.0
	iSectPts(VERT0, VERT1, VERT2, VV(0), VV(1), VV(2), D(0), D(1), D(2),
		 isect0, isect1, isectpoint0, isectpoint1);
      }
    else if(D(1)!=0)
      {
	iSectPts(VERT1, VERT0, VERT2, VV(1), VV(0), VV(2), D(1), D(0), D(2),
		 isect0, isect1, isectpoint0, isectpoint1);
      }
    else if(D(2)!=0)
      {
	iSectPts(VERT2, VERT0, VERT1, VV(2), VV(0), VV(1), D(2), D(0), D(1),
		 isect0, isect1, isectpoint0 ,isectpoint1);
      }
    else
      {
	// triangles are coplanar
	return 1;
      }
    return 0;
  }

}

bool triangleIntersection(const Vec3* U, const Vec3* V,
			  bool* coplanar,
			  Vec3* isectpt1, Vec3 *isectpt2)
{
  // compute plane equation of triangle(V0,V1,V2)
  Vec3 N1(cross(V[1]-V[0],V[2]-V[0]));
  float d1 = -dot(N1,V[0]);

  // plane equation 1: N1.X+d1=0

  // put U0,U1,U2 into plane equation 1 to compute signed distances to
  // the plane
  Vec3 du(trimEps(dot(N1,U[0])+d1),
	  trimEps(dot(N1,U[1])+d1),
	  trimEps(dot(N1,U[2])+d1));

  float du0du1 = du(0)*du(1);
  float du0du2 = du(0)*du(2);

  // same sign on all of them and != 0
  if(du0du1>0 && du0du2>0)
    return 0;

  // compute plane of triangle (U0,U1,U2)
  Vec3 N2(cross(U[1]-U[0], U[2]-U[0]));
  float d2 = -dot(N2, U[0]);
  // plane equation 2: N2.X+d2=0 //

  // put V0,V1,V2 into plane equation 2
  Vec3 dv(trimEps(dot(N2,V[0])+d2),
	  trimEps(dot(N2,V[1])+d2),
	  trimEps(dot(N2,V[2])+d2));

  float dv0dv1 = dv(0)*dv(1);
  float dv0dv2 = dv(0)*dv(2);

  // same sign on all of them and != 0
  if(dv0dv1>0 && dv0dv2>0)
    return 0;

  // compute direction of intersection line
  Vec3 D = cross(N1, N2);

  // compute and index to the largest component of D
  unsigned maxindex=indexMaxAbs(D);

  // this is the simplified projection onto L
  Vec3 vp(V[0](maxindex), V[1](maxindex), V[2](maxindex));
  Vec3 up(U[0](maxindex), U[1](maxindex), U[2](maxindex));

  // compute interval for triangle 1
  Vec3 isectpointA1, isectpointA2;
  float isect1[2];
  *coplanar = computeISectInterval(V[0], V[1], V[2], vp, dv,
				   dv0dv1, dv0dv2,
				   &isect1[0], &isect1[1],
				   &isectpointA1, &isectpointA2);
  if(*coplanar)
    // we treat coplanar triangles as not intersecting
    return 0;

  // compute interval for triangle 2
  Vec3 isectpointB1, isectpointB2;
  float isect2[2];
  computeISectInterval(U[0], U[1], U[2], up, du,
		       du0du1,du0du2,
		       &isect2[0], &isect2[1],
		       &isectpointB1, &isectpointB2);

  // intervals need to be in increasing order
  unsigned smallest1 = order(isect1[0], isect1[1]);
  unsigned smallest2 = order(isect2[0], isect2[1]);

  // triangles don't intersect
  if(isect1[1]<isect2[0] || isect2[1]<isect1[0])
    return 0;

  // triangles intersect
  if(isectpt1!=0 && isectpt2!=0)
    {
      // store points
      if(isect2[0]<isect1[0])
	{
	  *isectpt1 = smallest1==0 ? isectpointA1 : isectpointA2;

	  if(isect2[1]<isect1[1])
	    *isectpt2 = smallest2==0 ? isectpointB2 : isectpointB1;
	  else
	    *isectpt2 = smallest1==0 ? isectpointA2 : isectpointA1;
	}
      else
	{
	  *isectpt1 = smallest2==0 ? isectpointB1 : isectpointB2;

	  if(isect2[1]>isect1[1])
	    *isectpt2 = smallest1==0 ? isectpointA2 : isectpointA1;
	  else
	    *isectpt2 = smallest2==0 ? isectpointB2 : isectpointB1;
	}
    }
  return 1;
}

bool fragmentOverlap(const Fragment& f1, const Fragment& f2)
{
  //if(f1.type == Fragment::FR_TRIANGLE && f2.type == Fragment::FR_TRIANGLE)
  //  return triOverlap(f1.points, f2.points);

  return 0;
}

Vec3 randPoint()
{
  Vec3 p(rand()*1./RAND_MAX, rand()*1./RAND_MAX, rand()*1./RAND_MAX);
  return p;
}

int main()
{
  for(unsigned i=0; i<2000; ++i)
    {
      Vec3 t1[3];
      t1[0] = randPoint();
      t1[1] = randPoint();
      t1[2] = randPoint();
      Vec3 t2[3];
      t2[0] = randPoint();
      t2[1] = randPoint();
      t2[2] = randPoint();
      Vec3 ipts[2];

      Vec3 p1, p2;
      bool intersect, coplanar;
      intersect = triangleIntersection(t1, t2, &coplanar, &ipts[0], &ipts[1]);

      for(unsigned i=0;i!=4;++i)
	{
	  printf("%.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
		 t1[i%3](0), t1[i%3](1), t1[i%3](2),
		 t2[i%3](0), t2[i%3](1), t2[i%3](2),
		 ipts[i%2](0), ipts[i%2](1), ipts[i%2](2));
	}

      printf("Overlap=%i\n", intersect);
    }

  return 0;
}
