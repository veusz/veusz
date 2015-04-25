#include <algorithm>
#include <cmath>
#include "mmaths.h"
#include "fragment.h"
#include <QtCore/QPointF>
#include <QtGui/QPolygonF>
#include <QtGui/QPainterPath>
#include <stdio.h>

#define EPS 1e-5

unsigned Fragment::_count = 0;

namespace
{
  inline double trimEpsSqd(double v)
  {
    return std::abs(v) < (EPS*EPS) ? 0 : v;
  }

  // return 0 if f1<f2, else 1 and swap f1, f2
  inline unsigned order(double& f1, double& f2)
  {
    if(f1<f2)
      return 0;
    std::swap(f1, f2);
    return 1;
  }

  // return index of maximum absolute component
  inline unsigned indexMaxAbs(const Vec3& v)
  {
    double max=std::abs(v(0));
    unsigned idx=0;
    double b=std::abs(v(1));
    if(b>max) { max=b; idx=1; }
    double c=std::abs(v(2));
    if(c>max) { max=c; idx=2; }
    return idx;
  }

  inline void printv(const Vec3& a)
  {
    printf("%8g %8g %8g\n", a(0), a(1), a(2));
  }

  // return scale factor a/b between vectors (0 if not scaled)
  double scaleFactor(const Vec3& a, const Vec3& b)
  {
    // find non-zero index
    unsigned idx;
    for(idx=0; idx<3 && std::abs(b(idx))<EPS; ++idx)
      ;
    if(idx==3)
      return 0;

    double ratio = a(idx)/b(idx);
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
    return (std::abs(a(0)-b(0))<EPS && std::abs(a(1)-b(1))<EPS &&
	    std::abs(a(2)-b(2))<EPS);
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
    double x = (b1(idx)-a1(idx))/v1(idx);
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
			    double* isect0, double* isect1,
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

    double tmp1 = D(i0)/(D(i0)-D(i1));
    *isect0 = VV(i0)+(VV(i1)-VV(i0))*tmp1;
    *isectpoint0 = verts[i0] + (verts[i1]-verts[i0])*tmp1;

    double tmp2 = D(i0)/(D(i0)-D(i2));
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
    // printf("U[0] "); printv(U[0]);
    // printf("U[1] "); printv(U[1]);
    // printf("U[2] "); printv(U[2]);

    // printf("V[0] "); printv(V[0]);
    // printf("V[1] "); printv(V[1]);
    // printf("V[2] "); printv(V[2]);

    // compute plane equation of triangle(V0,V1,V2)
    Vec3 N1(cross(V[1]-V[0],V[2]-V[0]));

    double d1 = -dot(N1,V[0]);

    // printf("N1   "); printv(N1);
    // printf("d1   %8g\n", d1);

    // plane equation 1: N1.X+d1=0

    // put U0,U1,U2 into plane equation 1 to compute signed distances to
    // the plane
    Vec3 dotU(trimEpsSqd(dot(N1,U[0])+d1),
	      trimEpsSqd(dot(N1,U[1])+d1),
	      trimEpsSqd(dot(N1,U[2])+d1));

    // printf("dotU "); printv(dotU);

    // same sign on all of them and != 0
    if(dotU(0)*dotU(1)>0 && dotU(0)*dotU(2)>0)
      return 0;

    // compute plane of triangle (U0,U1,U2)
    Vec3 N2(cross(U[1]-U[0], U[2]-U[0]));
    double d2 = -dot(N2, U[0]);
    // plane equation 2: N2.X+d2=0 //

    // printf("N2   "); printv(N2);
    // printf("d2   %8g\n", d2);

    // put V0,V1,V2 into plane equation 2
    Vec3 dotV(trimEpsSqd(dot(N2,V[0])+d2),
	      trimEpsSqd(dot(N2,V[1])+d2),
	      trimEpsSqd(dot(N2,V[2])+d2));

    // printf("dotV "); printv(dotV);

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

    // printf("vp   "); printv(vp);
    // printf("up   "); printv(up);

    // compute interval for triangle 1
    Vec3 isectpointA1, isectpointA2;
    double isect1[2];
    *coplanar = computeISectInterval(V, vp, dotV,
				     &isect1[0], &isect1[1],
				     &isectpointA1, &isectpointA2);
    if(*coplanar)
      {
        // we treat coplanar triangles as not intersecting
        // printf("coplanar\n");
        return 0;
      }

    // compute interval for triangle 2
    Vec3 isectpointB1, isectpointB2;
    double isect2[2];
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
    //printf("split on corner\n");

    // line 1 is f.points[corner] + a*vec
    // line 2 is f.points[corner+1] + b*(f.points[corner+2]-f.points[corner])
    Vec3 p1 = f.points[corner];
    Vec3 p2 = f.points[(corner+1)%3];
    Vec3 V2 = f.points[(corner+2)%3] - p2;

    double a = scaleFactor(cross(p2-p1, V2), cross(vec, V2));
    if(a == 0)
      // triangles don't overlap, except at corner
      return 0;
    Vec3 newcorner = p1 + vec*a;

    // this is the normal to the original triangle
    Vec3 orignorm = triNormal(f.points);

    Fragment newf = f;
    newf.bumpIndex();
    newf.points[0] = f.points[corner];
    newf.points[1] = newcorner;
    newf.points[2] = f.points[(corner+1)%3];
    // swap points if normal is in wrong direction
    if(scaleFactor(triNormal(newf.points), orignorm) < 0)
      std::swap(newf.points[1], newf.points[2]);
    frags.push_back(newf);

    newf.bumpIndex();
    newf.points[1] = newcorner;
    newf.points[2] = f.points[(corner+2)%3];
    if(scaleFactor(triNormal(newf.points), orignorm) < 0)
      std::swap(newf.points[1], newf.points[2]);
    frags.push_back(newf);

    return 2;
  }

  // if we have a vector pt1->pt2, find triangle edges which intercept
  // this and split into three triangles
  unsigned splitOnLine(const Fragment& frag, const Vec3& pt1, const Vec3& pt2,
		       FragmentVector& frags)
  {
    //printf("split on line\n");

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
	//double scale = scaleFactor(cross(P2-P1, V2), cross(V1, V2));
	double scale=scaleFactor(c1, c2);
	if(scale>0 && scale<1)
	  {
	    cornersect[corneridx] = i;
	    cornerpos[corneridx++] = P1 + V1*scale;
	  }
      }

    // the line should meet exactly 2 edges
    if(corneridx != 2)
      {
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
    newf.bumpIndex();
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

    newf.bumpIndex();
    newf.points[0] = cornerpos[0];
    newf.points[1] = frag.points[otherpt1[common]];
    newf.points[2] = frag.points[otherpt2[common]];
    if(scaleFactor(triNormal(newf.points), orignorm) < 0)
      std::swap(newf.points[1], newf.points[2]);
    frags.push_back(newf);

    newf.bumpIndex();
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
	//printf("line hits edge\n");
	return 0;
      }

    // check if line intercepts a corner and split triangle into 2
    for(unsigned i=0; i<3; ++i)
      {
	if(nearPoints(f.points[i], pt1) || nearPoints(f.points[i], pt2))
	  return splitOnCorner(f, i, frags, pt2-pt1);
      }

    // final case is to split two edges to make 3 triangles
    return splitOnLine(f, pt1, pt2, frags);
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

////////////////////////////////////////////////////////
// 2D triangle intersection

namespace
{
  // is a to the left of p1->p2?
  // 1 if left, -1 if right, 0 if colinear
  inline bool SHInside(Vec2 p, Vec2 cp0, Vec2 cp1)
  {
    return (cp1(0)-cp0(0))*(p(1)-cp0(1)) > (cp1(1)-cp0(1))*(p(0)-cp0(0));
  };

  // return whether lines intersect and return intersection point
  inline bool SHIntersection(Vec2 a0, Vec2 a1, Vec2 b0, Vec2 b1, Vec2* res)
  {
    Vec2 da = a0-a1;
    Vec2 db = b0-b1;
    double n1 = cross(a0, a1);
    double n2 = cross(b0, b1);
    double denom = cross(da, db);
    if(denom == 0)
      return 0;
    double idenom = 1/denom;
    *res = db*(n1*idenom) - da*(n2*idenom);
    return 1;
  }

  // Polygon with small number of points
  class Vec2SmallPoly
  {
  public:
    Vec2SmallPoly(unsigned size=0) : count(size) {};
    void push_back(Vec2 v) { points[count++] = v; }
    unsigned size() const { return count; }
    bool empty() const { return count == 0; }
    void clear() { count = 0; }
    Vec2& operator[](unsigned i) { return points[i]; }
    Vec2 operator[](unsigned i) const { return points[i]; }
  private:
    unsigned count;
    Vec2 points[8];
  };

  // Sutherland–Hodgman algorithm for clipping polygon against
  // 2nd polygon. Requires clockwise orientation of points.
  Vec2SmallPoly polyEdgeClip(Vec2SmallPoly inPoly, const Vec2SmallPoly &clipPoly)
  {
    if(clipPoly.empty())
      return inPoly;
    Vec2 cp1 = clipPoly[clipPoly.size()-1];
    for(unsigned ci=0; ci != clipPoly.size() && !inPoly.empty(); ++ci)
      {
	Vec2 cp2 = clipPoly[ci];

	Vec2SmallPoly outPoly;
	Vec2 S = inPoly[inPoly.size()-1];
	for(unsigned si=0; si != inPoly.size(); ++si)
	  {
	    Vec2 E = inPoly[si];
	    if(SHInside(E, cp1, cp2))
	      {
		if(!SHInside(S, cp1, cp2))
		  {
		    Vec2 isect;
		    if(SHIntersection(S, E, cp1, cp2, &isect))
		      outPoly.push_back(isect);
		  }
		outPoly.push_back(E);
	      }
	    else if(SHInside(S, cp1, cp2))
	      {
		Vec2 isect;
		if(SHIntersection(S, E, cp1, cp2, &isect))
		  outPoly.push_back(isect);
	      }
	    S = E;
	  }
	inPoly = outPoly;
	cp1 = cp2;
      }
    return inPoly;
  }

  // is the triangle clockwise (suitable for use in SH clipping)
  inline bool triangleClockwise(const Vec2SmallPoly& pts)
  {
    return (pts[1](0)-pts[0](0))*(pts[2](1)-pts[0](1)) -
      (pts[2](0)-pts[0](0))*(pts[1](1)-pts[0](1)) > 0;
  }

  // given depths for corners of projected triangle tripts
  // interpolate to find average depth of pt[] there
  double interpolateTriangleDepth(const Vec2SmallPoly& pts, const Vec3* tripts)
  {
    // solve pt = p0 + a*(p1-p0) + b*(p2-p0)
    // where p0,1,2 are the points of the triangle

    // We need to find a case where the x and y coordinates of p0/3 are
    // different
    double invx, invy, alower, invalower, sum;
    unsigned baseidx;
    Vec3 p0, p1, p2;

    for(baseidx=0; baseidx<3; ++baseidx)
      {
	p0=tripts[baseidx];
	p1=tripts[(baseidx+1)%3];
	p2=tripts[(baseidx+2)%3];
	if( std::abs(p0(0)-p2(0)) > 1e-5 && std::abs(p0(1)-p2(1)) > 1e-5 )
	  break;
      }
    // we didn't find this - perhaps it's not really a triangle, but
    // return mean depth
    if(baseidx == 3)
      goto error;

    invx = 1/(p2(0)-p0(0));
    invy = 1/(p2(1)-p0(1));
    alower = (p1(0)-p0(0))*invx-(p1(1)-p0(1))*invy;
    if(alower == 0)
      goto error;
    invalower = 1/alower;

    sum = 0;
    for(unsigned i=0; i != pts.size(); ++i)
      {
	double aupper = (pts[i](0)-p0(0))*invx-(pts[i](1)-p0(1))*invy;
	double a = aupper*invalower;
	double b = (pts[i](0)-p0(0) - a*(p1(0)-p0(0)))*invx;
	double depth = p0(2) + a*(p1(2)-p0(2)) + b*(p2(2)-p0(2));
	sum += depth;
      }
    return sum/pts.size();

    // much more efficient than an exception here
  error:
    return (tripts[0](3)+tripts[1](3)+tripts[2](3))*(1/3.f);
  }

  void overlapDepthTriangleTriangle(const Fragment& f1, const Fragment& f2,
				    double* d1, double* d2)
  {
    // copy points to 2D vectors
    Vec2SmallPoly tri1(3);
    for(unsigned i=0; i != 3; ++i)
      {
	tri1[i](0) = f1.proj[i](0); tri1[i](1) = f1.proj[i](1);
      }
    if(!triangleClockwise(tri1))
      std::swap(tri1[0], tri1[1]);

    Vec2SmallPoly tri2(3);
    for(unsigned i=0; i != 3; ++i)
      {
	tri2[i](0) = f2.proj[i](0); tri2[i](1) = f2.proj[i](1);
      }
    if(!triangleClockwise(tri2))
      std::swap(tri2[0], tri2[1]);

    // look for overlap in 2D
    Vec2SmallPoly clipped = polyEdgeClip(tri1, tri2);
    if(! clipped.empty())
      {
	*d1 = interpolateTriangleDepth(clipped, f1.proj);
	*d2 = interpolateTriangleDepth(clipped, f2.proj);
      }
  }
}

// get average depths of intersection in 2D
void overlapDepth(const Fragment& f1, const Fragment& f2,
		  double* d1, double* d2)
{
  if(f1.type == Fragment::FR_TRIANGLE && f2.type == Fragment::FR_TRIANGLE)
    {
      overlapDepthTriangleTriangle(f1, f2, d1, d2);
    }

  // fixme line and triangles
}

namespace
{
  double polyArea(const QPolygonF& p)
  {
    double area = 0;
    for(int i=0; i<p.size(); ++i)
      {
        int n=(i+1) % p.size();
        area += p[i].x()*p[n].y()-p[n].x()*p[i].y();
      }
    return area;
  }

  bool polyDirection(const QPolygonF& p)
  {
    return polyArea(p) > 0;
  }

  // given depths for corners of projected triangle tripts
  // interpolate to find average depth of pt[] there
  void updateTriangleDepths(const Vec3* tripts, Vec3* npts)
  {
    // solve pt = p0 + a*(p1-p0) + b*(p2-p0)
    // where p0,1,2 are the points of the triangle

    // We need to find a case where the x and y coordinates of p0/3 are
    // different
    Vec3 p0, p1, p2;
    unsigned baseidx;
    for(baseidx=0; baseidx<3; ++baseidx)
      {
	p0=tripts[baseidx];
	p1=tripts[(baseidx+1)%3];
	p2=tripts[(baseidx+2)%3];
	if( std::abs(p0(0)-p2(0)) > 1e-5 && std::abs(p0(1)-p2(1)) > 1e-5 )
	  break;
      }
    // we didn't find this - perhaps it's not really a triangle
    if(baseidx == 3)
      return;

    double invx = 1/(p2(0)-p0(0));
    double invy = 1/(p2(1)-p0(1));
    double alower = (p1(0)-p0(0))*invx-(p1(1)-p0(1))*invy;
    if(alower == 0)
      return;
    double invalower = 1/alower;

    for(unsigned i=0; i<3; ++i)
      {
	double aupper = (npts[i](0)-p0(0))*invx-(npts[i](1)-p0(1))*invy;
	double a = aupper*invalower;
	double b = (npts[i](0)-p0(0) - a*(p1(0)-p0(0)))*invx;
	double depth = p0(2) + a*(p1(2)-p0(2)) + b*(p2(2)-p0(2));
        npts[i](2) = rounddp(depth);
      }
  }

  bool findSelfIntersectPoly(const QPolygonF& poly, int* isectpt, int* isectline)
  {
    const int size = poly.size();
    for(int ptidx=0; ptidx<size; ++ptidx)
      {
        QPointF pt = poly[ptidx];
        for(int lidx=0; lidx+2<size; ++lidx)
          {
            QPointF pt1 = poly[(ptidx+lidx+1) % size];
            QPointF pt2 = poly[(ptidx+lidx+2) % size];

            bool hit;
            if( std::abs(pt1.x()-pt2.x()) < EPS )
              {
                // is this on the vertical line
                hit = std::abs(pt.x() - pt1.x()) < EPS;
              }
            else
              {
                double frac = (pt.x()-pt1.x())/(pt2.x()-pt1.x());
                if( frac > EPS && frac < (1-EPS) )
                  {
                    double py = (pt2.y()-pt1.y())*frac + pt1.y();
                    hit = std::abs(pt.y() - py) < EPS;
                  }
                else
                  hit = 0;
              }

            if(hit)
              {
                *isectpt = ptidx;
                *isectline = (ptidx+lidx+1) % size;
                return 1;
              }
          }
      }
    return 0;
  }

  void fixSelfIntersectPolygons(QList<QPolygonF>& polys)
  {
    // check whether any points lie on any lines and split if so
    // this is slow O(N^2), but hopefully N is small
    for(int polyidx=0; polyidx<polys.size();)
      {
        int isectpt, isectline;
        if(findSelfIntersectPoly(polys[polyidx], &isectpt, &isectline))
          {
            QPolygonF& poly = polys[polyidx];

            // split left and right of point
            QPolygonF poly1;
            int pidx = isectpt-1;
            for(;;)
              {
                pidx = (pidx+1) % poly.size();
                poly1 << poly[pidx];
                if(pidx == isectline)
                  break;
              }

            QPolygonF poly2;
            for(;;)
              {
                pidx = (pidx+1) % poly.size();
                poly2 << poly[pidx];
                if(pidx == isectpt)
                  break;
              }

            polys[polyidx] = poly1;
            polys.push_back(poly2);
          }
        else
          ++polyidx;
      }
  }

  void fixDuplicatePoints(QPolygonF& poly)
  {
    QPointF lastp=poly.front();
    for(int i=poly.size()-1; i>=0; --i)
      {
        QPointF thisp=poly[i];
        if(std::abs(lastp.x()-thisp.x())<EPS &&
           std::abs(lastp.y()-thisp.y())<EPS)
          poly.remove(i);
        lastp=thisp;
      }
  }

  unsigned addPainterPathAsTris(FragmentVector& frags,
                                const Fragment templ,
                                const QPainterPath& pp)
  {
    QList<QPolygonF> polylist(pp.toSubpathPolygons());
    fixSelfIntersectPolygons(polylist);

    unsigned ct = 0;
    for(int i=0; i<polylist.size(); ++i)
      {
        QPolygonF& poly = polylist[i];
        fixDuplicatePoints(poly);

        for(int i=0; i<poly.size(); ++i)
          printf("%g %g, ", poly[i].x(), poly[i].y());
        printf("\n");

        // this is the ear clipping algorithm
        bool dirn = polyDirection(poly);
        int i = 0;
        while(poly.size() > 2)
          {
            printf("split %i, %i\n", i, poly.size());

            QPolygonF tri;
            tri << poly[i % poly.size()]
                << poly[(i+1) % poly.size()]
                << poly[(i+2) % poly.size()];

            bool ok = poly.size() == 3;
            // is triangle in correct direction and does it not
            // contain any other points in the polygon
            if(polyDirection(tri) == dirn)
              {
                ok = 1;
                // check no vertex in the polygon is inside
                for(int o=3; o<poly.size(); ++o)
                  {
                    if(tri.containsPoint(poly[(i+o) % poly.size()],
                                         Qt::OddEvenFill))
                      {
                        ok = 0; break;
                      }
                  }
              }

            if(ok)
              {
                Fragment f(templ);
                f.splitcount++;
                f.bumpIndex();
                for(unsigned j=0; j<3; ++j)
                  f.proj[j] = Vec3(rounddp(tri[j].x()), rounddp(tri[j].y()), 0);
                printf("Add: %g %g, %g %g, %g %g\n",
                       tri[0].x(), tri[0].y(),
                       tri[1].x(), tri[1].y(),
                       tri[2].x(), tri[2].y());
                updateTriangleDepths(templ.proj, f.proj);
                frags.push_back(f);
                poly.remove((i+1) % poly.size());
                ++ct;
              }
            else
              ++i;
          }
      } // list of polygons
    return ct;
  }

  void splitOn2DOverlap_tri_tri(FragmentVector& frags,
                                unsigned idx1, unsigned idx2,
                                unsigned* n1, unsigned* n2)
  {
    QPainterPath pp1;
    QPainterPath pp2;
    {
      const Fragment& f1 = frags[idx1];
      QPolygonF p1(3);
      printf("frag1\n");
      for(unsigned i=0; i<3; ++i)
        {
          p1[i] = QPointF(f1.proj[i](0), f1.proj[i](1));
          printf("%g %g\n", p1[i].x(), p1[i].y());
        }
      pp1.addPolygon(p1);
      pp1.closeSubpath();

      const Fragment& f2 = frags[idx2];
      QPolygonF p2(3);
      printf("frag2\n");
      for(unsigned i=0; i<3; ++i)
        {
          p2[i] = QPointF(f2.proj[i](0), f2.proj[i](1));
          printf("%g %g\n", p1[i].x(), p1[i].y());
        }
      pp2.addPolygon(p2);
      pp2.closeSubpath();
    }

    QPainterPath ppu(pp1.intersected(pp2));
    if(ppu.isEmpty())
      return;

    QPainterPath pp1diff = pp1.subtracted(ppu);
    QPainterPath pp2diff = pp2.subtracted(ppu);
    if(pp1diff.isEmpty() && pp2diff.isEmpty())
      return;

    *n1 = addPainterPathAsTris(frags, frags[idx1], ppu);
    if(! pp1diff.isEmpty())
      {
        *n1 += addPainterPathAsTris(frags, frags[idx1], pp1diff);
      }

    *n2 = addPainterPathAsTris(frags, frags[idx2], ppu);
    if(! pp2diff.isEmpty())
      {
        *n2 += addPainterPathAsTris(frags, frags[idx2], pp2diff);
      }

    printf("n1=%i n2=%i\n", *n1, *n2);
  }

};


// do the fragments overlap, if so make new projected fragments
// return number added for idx1 and idx2 as n1 and n2
// (these were added to the end of the fragments vector)
void splitOn2DOverlap(FragmentVector& fragments,
                      unsigned idx1, unsigned idx2,
                      unsigned* n1, unsigned* n2)
{
  if(fragments[idx1].type == Fragment::FR_TRIANGLE &&
     fragments[idx2].type == Fragment::FR_TRIANGLE)
    {
      splitOn2DOverlap_tri_tri(fragments, idx1, idx2, n1, n2);
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

  for(unsigned i=0; i<1000; ++i)
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
          printf("intersection\n");
	}

    }

  return 0;
}
#endif

#if 0
int main()
{
  Vec3 a[3];
  Vec3 b[3];

a[0](0)=-1.84210526315789491214e-01; a[0](1)=-1.40727850317379221678e-06; a[0](2)=-3.70710678118514724844e+00;
a[1](0)=-1.84210526315789491214e-01; a[1](1)=3.72148131669521042841e-02; a[1](2)=-3.66989070887427137890e+00;
a[2](0)=-1.31578947368421073083e-01; a[2](1)=-1.40727850317379221678e-06; a[2](2)=-3.70710678118514724844e+00;


b[0](0)=-1.84210526315789491214e-01; b[0](1)=3.72148131669521042841e-02; b[0](2)=-3.66989070887427137890e+00;
b[1](0)=-1.84210526315789491214e-01; b[1](1)=7.44310336124073823605e-02; b[1](2)=-3.63267463656339506528e+00;
b[2](0)=-1.31578947368421073083e-01; b[2](1)=3.72148131669521042841e-02; b[2](2)=-3.66989070887427137890e+00;


  bool coplanar;
  Vec3 pt1, pt2;

  bool retn = triangleIntersection(a, b, &coplanar, &pt1, &pt2);

  printf("retn %i\n", retn);

  printv(pt1);
  printv(pt2);
}
#endif


#if 1
int main()
{
  FragmentVector v;
  Fragment f;
  f.type = Fragment::FR_TRIANGLE;

  f.proj[0] = Vec3(0.3, 0.3, 1);
  f.proj[1] = Vec3(1, 0., 0.5);
  f.proj[2] = Vec3(0.5, 1, 1);
  v.push_back(f);

  f.proj[0] = Vec3(0, 0, 0);
  f.proj[1] = Vec3(0, 1, 0.5);
  f.proj[2] = Vec3(1, 0.5, 1);
  v.push_back(f);

  unsigned n1=0;
  unsigned n2=0;

  splitOn2DOverlap(v, 0, 1, &n1, &n2);

  for(unsigned i=0; i!=v.size(); ++i)
    {
      double dx=0;//rand()*0.04/RAND_MAX;
      double dy=0;//rand()*0.04/RAND_MAX;

      for(unsigned j=0; j!=4; ++j)
        printf("%g %g %g %i\n", v[i].proj[j%3](0)+dx, v[i].proj[j%3](1)+dy,
               v[i].proj[j%3](2), i);
      printf("nan nan nan\n");
    }

  return 0;
}
#endif
