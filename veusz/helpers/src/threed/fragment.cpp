#include <algorithm>
#include <cmath>
#include "mmaths.h"
#include "fragment.h"

#include <stdio.h>

#define EPS 1e-5f

namespace
{
  inline float trimEps(float v)
  {
    return std::abs(v) < EPS ? 0 : v;
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

  // void printv(const Vec3& a)
  // {
  //   printf("%g %g %g\n", a(0), a(1), a(2));
  // }

  // return scale factor a/b between vectors (0 if not scaled)
  float scaleFactor(const Vec3& a, const Vec3& b)
  {
    // find non-zero index
    unsigned idx;
    for(idx=0; idx<3 && std::abs(b(idx))<EPS; ++idx)
      ;
    if(idx==3)
      return 0;

    float ratio = a(idx)/b(idx);
    if ( std::abs(b((idx+1)%3)*ratio - a((idx+1)%3)) < EPS &&
	 std::abs(b((idx+2)%3)*ratio - a((idx+2)%3)) < EPS )
      return ratio;
    else
      return 0;
  }

  // is this a zero vector?
  inline bool nearZero(const Vec3& v)
  {
    return std::abs(v(0))<EPS && std::abs(v(1))<EPS && std::abs(v(2))<EPS;
  }

  // are these points nearly the same
  inline bool nearPoints(const Vec3& a, const Vec3& b)
  {
    return nearZero(a-b);
  }

  // are these two (infinite) lines overlapping?
  bool overlappingLine(const Vec3& a1, const Vec3& a2,
		       const Vec3& b1, const Vec3& b2)
  {
    // are they parallel?
    Vec3 v1 = a2-a1;
    Vec3 v2 = b2-b1;
    if(scaleFactor(v1, v2) == 0)
      // no, they're not
      return 0;

    // solve x*v1+a1 = b1
    // find non-zero index of v1
    unsigned idx;
    for(idx=0; idx<3 && std::abs(v1(idx))<EPS; ++idx)
      ;
    if(idx==3)
      // a1==a2
      return 0;

    // solve for x in one coordinate
    float x = (b1(idx)-a1(idx))/v1(idx);
    // the check the rest match that x
    return (std::abs(x*v1((idx+1)%3) + a1((idx+1)%3) - b1((idx+1)%3)) < EPS &&
	    std::abs(x*v1((idx+2)%3) + a1((idx+2)%3) - b1((idx+2)%3)) < EPS);
  }

  // for a triangular set of points, compute (unnormalized) normal
  inline Vec3 triNormal(const Vec3* points)
  {
    return cross(points[1]-points[0], points[2]-points[0]);
  }

  bool computeISectInterval(const Vec3* verts,
			    const Vec3& VV, const Vec3& D,
			    float* isect0, float* isect1,
			    Vec3* isectpoint0, Vec3* isectpoint1)
  {
    // these are the appropriate vertices to do the computation on
    unsigned i0, i1, i2;

    if(D(0)*D(1)>0)
      {
	// here we know that D0D2<=0.0 that is D(0), D(1) are on the
	// same side, D(2) on the other or on the plane
        i0 = 2; i1 = 0; i2 = 1;
      }
    else if(D(0)*D(2)>0)
      {
	// here we know that d0d1<=0.0
        i0 = 1; i1 = 0; i2 = 2;
      }
    else if(D(1)*D(2)>0 || D(0)!=0)
      {
	// here we know that d0d1<=0.0 or that D(0)!=0.0
        i0 = 0; i1 = 1; i2 = 2;
      }
    else if(D(1)!=0)
      {
        i0 = 1; i1 = 0; i2 = 2;
      }
    else if(D(2)!=0)
      {
        i0 = 2; i1 = 0; i2 = 1;
      }
    else
      {
	// triangles are coplanar
	return 1;
      }

    float tmp1 = D(i0)/(D(i0)-D(i1));
    *isect0 = VV(i0)+(VV(i1)-VV(i0))*tmp1;
    *isectpoint0 = verts[i0] + (verts[i1]-verts[i0])*tmp1;

    float tmp2 = D(i0)/(D(i0)-D(i2));
    *isect1 = VV(i0)+(VV(i2)-VV(i0))*tmp2;
    *isectpoint1 = verts[i0] + (verts[i2]-verts[i0])*tmp2;

    return 0;
  }

  // code to compute intersection of triangles is
  // based on Thomas Moller's Public Domain code
  // See article "A Fast Triangle-Triangle Intersection Test",
  // Journal of Graphics Tools, 2(2), 1997

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
    Vec3 dotU(trimEps(dot(N1,U[0])+d1),
	      trimEps(dot(N1,U[1])+d1),
	      trimEps(dot(N1,U[2])+d1));

    // same sign on all of them and != 0
    if(dotU(0)*dotU(1)>0 && dotU(0)*dotU(2)>0)
      return 0;

    // compute plane of triangle (U0,U1,U2)
    Vec3 N2(cross(U[1]-U[0], U[2]-U[0]));
    float d2 = -dot(N2, U[0]);
    // plane equation 2: N2.X+d2=0 //

    // put V0,V1,V2 into plane equation 2
    Vec3 dotV(trimEps(dot(N2,V[0])+d2),
	      trimEps(dot(N2,V[1])+d2),
	      trimEps(dot(N2,V[2])+d2));

    // same sign on all of them and != 0
    if(dotV(0)*dotV(1)>0 && dotV(0)*dotV(2)>0)
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
    *coplanar = computeISectInterval(V, vp, dotV,
				     &isect1[0], &isect1[1],
				     &isectpointA1, &isectpointA2);
    if(*coplanar)
      // we treat coplanar triangles as not intersecting
      return 0;

    // compute interval for triangle 2
    Vec3 isectpointB1, isectpointB2;
    float isect2[2];
    computeISectInterval(U, up, dotU,
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

  // find corner where vec intersects triangle from corner and split
  // into two triangles
  unsigned splitOnCorner(const Fragment& f, unsigned corner,
			 FragmentVector& frags, const Vec3& vec)
  {
    printf("split on corner\n");

    // line 1 is f.points[corner] + a*vec
    // line 2 is f.points[corner+1] + b*(f.points[corner+2]-f.points[corner])
    Vec3 p1 = f.points[corner];
    Vec3 p2 = f.points[(corner+1)%3];
    Vec3 V2 = f.points[(corner+2)%3] - p2;

    float a = scaleFactor(cross(p2-p1, V2), cross(vec, V2));
    if(a == 0)
      // triangles don't overlap, except at corner
      return 0;
    Vec3 newcorner = p1 + vec*a;

    // this is the normal to the original triangle
    Vec3 orignorm = triNormal(f.points);

    Fragment newf = f;
    newf.points[0] = f.points[corner];
    newf.points[1] = newcorner;
    newf.points[2] = f.points[(corner+1)%3];
    // swap points if normal is in wrong direction
    if(scaleFactor(triNormal(newf.points), orignorm) < 0)
      std::swap(newf.points[1], newf.points[2]);
    frags.push_back(newf);

    newf.points[1] = newcorner;
    newf.points[2] = f.points[(corner+2)%3];
    if(scaleFactor(triNormal(newf.points), orignorm) < 0)
      std::swap(newf.points[1], newf.points[2]);
    frags.push_back(newf);

    return 2;
  }

  // if we have a vector pt1->pt2, find triangle edges which intercept
  // this and split into three triangles
  unsigned splitOnEdges(const Fragment& frag, const Vec3& pt1, const Vec3& pt2,
			FragmentVector& frags)
  {
    printf("split on edges\n");

    Vec3 cornerpos[3];
    unsigned cornersect[3];
    unsigned corneridx = 0;

    const Vec3& P2 = pt1;
    const Vec3 V2 = pt2-pt1;

    // pairs of vertexes of triangle
    static const unsigned startidx[3] = {0, 0, 1};
    static const unsigned endidx[3] =   {1, 2, 2};

    for(unsigned i=0; i<3; ++i)
      {
	Vec3 P1 = frag.points[startidx[i]];
	Vec3 V1 = frag.points[endidx[i]]-frag.points[startidx[i]];

	Vec3 c1 = cross(P2-P1, V2);
	Vec3 c2 = cross(V1,V2);
	//float scale = scaleFactor(cross(P2-P1, V2), cross(V1, V2));
	float scale=scaleFactor(c1, c2);
	if(scale>0 && scale<1)
	  {
	    cornersect[corneridx] = i;
	    cornerpos[corneridx++] = P1 + V1*scale;
	  }
      }

    // the line should meet exactly 2 edges
    if(corneridx != 2)
      {
	printf("bailed out on number of overlaps: %i\n", corneridx);
	// we shouldn't get here really
	return 0;
      }

    // this is the point index in common between the edges which are
    // intersected
    unsigned common;
    if(cornersect[0]==0 && cornersect[1]==1)
      common = 0; // point 0 in common
    else if (cornersect[0]==0 && cornersect[1]==2)
      common = 1; // point 1 in common
    else
      common = 2;

    // orignal normal of triangle
    const Vec3 orignorm = triNormal(frag.points);

    // copy fragment and make new triangles
    Fragment newf(frag);

    // this is the triangle to one side of the crossing line
    newf.points[0] = frag.points[common];
    newf.points[1] = cornerpos[0];
    newf.points[2] = cornerpos[1];
    // keep normal in same direction
    if(scaleFactor(triNormal(newf.points), orignorm) < 0)
      std::swap(newf.points[1], newf.points[2]);
    frags.push_back(newf);

    // work out how to split into 2 triangles on other side of crossing
    // line
    // we don't want to "cross" the triangles over each other
    static const unsigned otherpt1[3] = {1, 0, 0};
    static const unsigned otherpt2[3] = {2, 2, 1};

    printf("idxs %i %i\n", otherpt1[common], otherpt2[common]);
    newf.points[0] = cornerpos[0];
    newf.points[1] = frag.points[otherpt1[common]];
    newf.points[2] = frag.points[otherpt2[common]];
    if(scaleFactor(triNormal(newf.points), orignorm) < 0)
      std::swap(newf.points[1], newf.points[2]);
    frags.push_back(newf);

    newf.points[0] = cornerpos[0];
    newf.points[1] = frag.points[otherpt2[common]];
    newf.points[2] = cornerpos[1];
    if(scaleFactor(triNormal(newf.points), orignorm) < 0)
      std::swap(newf.points[1], newf.points[2]);
    frags.push_back(newf);

    return 3;
  }

  unsigned splitTriangle(const Fragment& f, const Vec3& pt1, const Vec3& pt2,
			 FragmentVector& frags)
  {
    // if line is at edge of triangle, no fragment
    if( overlappingLine(f.points[0], f.points[1], pt1, pt2) ||
	overlappingLine(f.points[1], f.points[2], pt1, pt2) ||
	overlappingLine(f.points[2], f.points[0], pt1, pt2) )
      {
	printf("hit sides\n");
	return 0;
      }

    // check if line intercepts a corner and split triangle into 2
    for(unsigned i=0; i<3; ++i)
      {
	if(nearPoints(f.points[i], pt1) || nearPoints(f.points[i], pt2))
	  return splitOnCorner(f, i, frags, pt2-pt1);
      }

    // final case is to split two edges to make 3 triangles
    return splitOnEdges(f, pt1, pt2, frags);
  }

  // end of namespace
}

void splitFragments(const Fragment &f1, const Fragment &f2,
		    FragmentVector& v, unsigned* num1, unsigned* num2)
{
  *num1 = *num2 = 0;
  if(f1.type == Fragment::FR_TRIANGLE && f2.type == Fragment::FR_TRIANGLE)
    {
      bool coplanar = 0;
      Vec3 pt1, pt2;
      if(triangleIntersection(f1.points, f2.points, &coplanar, &pt1, &pt2))
	{
	  printf("intersection!\n");
	  printf(" point 1: %g, %g, %g\n", pt1(0), pt1(1), pt1(2));
	  printf(" point 2: %g, %g, %g\n", pt2(0), pt2(1), pt2(2));
	  // when new fragments are added, then f1/f2 are invalidated
	  Fragment fcpy1(f1);
	  Fragment fcpy2(f2);
	  if(fcpy1.splitcount++ <= 6)
	    *num1 = splitTriangle(fcpy1, pt1, pt2, v);
	  if(fcpy2.splitcount++ <= 6)
	    *num2 = splitTriangle(fcpy2, pt1, pt2, v);
	}
    }
}

// testing routines
// essential for testing whether this actually works
// plot in gnuplot with
// splot "test_XX.dat" lc var with lines

#if 0
Vec3 randPoint()
{
  Vec3 p(rand()*1./RAND_MAX, rand()*1./RAND_MAX, rand()*1./RAND_MAX);
  return p;
}

int main()
{
  Fragment f1, f2;

  unsigned outfi = 0;

  for(unsigned i=0; i<100; ++i)
    {
      f1.points[0] = randPoint();
      f1.points[1] = randPoint();
      f1.points[2] = randPoint();
      f2.points[0] = randPoint();
      f2.points[1] = randPoint();
      f2.points[2] = randPoint();

      // for testing edge/corner hitting
      unsigned idx1 = rand() % 3;
      f2.points[idx1] = f1.points[0];

      Vec3 ipts[2];
      Vec3 p1, p2;
      bool intersect, coplanar;

      intersect = triangleIntersection(f1.points, f2.points,
				       &coplanar, &ipts[0], &ipts[1]);

      if(intersect)
	{
	  char filename[128];
	  sprintf(filename, "test_%i.dat", outfi++);

	  FILE *f;
	  f = fopen(filename, "w");
	  FragmentVector v;
	  fragmentTriangle(f1, ipts[0], ipts[1], v);
	  fragmentTriangle(f2, ipts[0], ipts[1], v);

	  for(unsigned i=0; i!=v.size(); ++i)
	    {
	      for(unsigned j=0; j<4; ++j)
		{
		  const Vec3 &a = v[i].points[j%3];
		  fprintf(f, "%g %g %g %i %i\n", a(0), a(1), a(2), i+1, 1);
		}
	      fputs("\n\n",f);
	    }

	  for(unsigned j=0; j<4; ++j)
	    {
	      const Vec3 &a = f1.points[j%3];
	      fprintf(f,"%g %g %g %i %i\n", a(0), a(1), a(2), v.size()+1, 2);
	    }
	  fputs("\n\n",f);
	  for(unsigned j=0; j<4; ++j)
	    {
	      const Vec3 &a = f2.points[j%3];
	      fprintf(f,"%g %g %g %i %i\n", a(0), a(1), a(2), v.size()+2, 2);
	    }
	  fputs("\n\n",f);
	  fclose(f);

	}

    }

  return 0;
}
#endif
