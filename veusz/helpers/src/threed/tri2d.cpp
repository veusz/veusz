#include <algorithm>
#include <bitset>

#include <iostream>


#include "mmaths.h"
#include "fixedvector.h"
#include "tri2d.h"

//typedef std::vector FixedVector;

//#include <vector>
//template<class T, unsigned N> class FixedVector : public std::vector<T> {};

// ignore differences in doubles < this
// maximum number of points/nodes in polygons
#define NODE_BITS 4 // Max of 16 nodes (0..2**NODE_BITS-1)
#define MAXNODES (1<<NODE_BITS)
#define MAXLINKS 32
#define MAXTRIANGLES 32

#define SNAPDIST 1e-8
#define CLOSEDIST 1e-8
#define EPS 1e-12
#define ISCALE 1000

namespace
{

// inline double round4(double d)
// {
//   //return d;
//   return round(d*ISCALE)*(1./ISCALE);
// }

enum isect { LINE_NOOVERLAP, LINE_CROSS, LINE_OVERLAP };

// Do the two line segments p1->p2, q1->q2 cross or overlap?
// return LINE_NOOVERLAP if no overlap
//        LINE_CROSS if they cross somewhere
//        LINE_OVERLAP if they lie on top of each other partially
// if posn != 0, return crossing position if LINE_CROSS
// if LINE_OVERLAP the two end points of overlap are returned in posn and posn2
// Assumes that the line segments are finite.

isect calcLine2DIntersect(Vec2 p1, Vec2 p2, Vec2 q1, Vec2 q2, Vec2* posn=0, Vec2* posn2=0)
{
  Vec2 dp = p2-p1;
  Vec2 dq = q2-q1;
  Vec2 dpq = p1-q1;
  double denom = cross(dp, dq);

  //std::cout << "denom " << denom << '\n';

  // parallel vectors or points below
  if(std::abs(denom) < EPS)
    {
      if( std::abs(cross(dp, dpq)) > EPS || std::abs(cross(dq, dpq)) > EPS )
        return LINE_NOOVERLAP;

      // colinear segments - do they overlap?
      double u0, u1;
      Vec2 dpq2 = p2-q1;
      if(std::abs(dq(0)) > std::abs(dq(1)))
        {
          u0 = dpq(0)*(1/dq(0));
          u1 = dpq2(0)*(1/dq(0));
        }
      else
        {
          u0 = dpq(1)*(1/dq(1));
          u1 = dpq2(1)*(1/dq(1));
        }

      if(u0 > u1)
        std::swap(u0, u1);

      if( u0>(1+EPS) || u1<-EPS )
        return LINE_NOOVERLAP;

      u0 = std::max(u0, 0.);
      u1 = std::min(u1, 1.);
      if(posn != 0)
        *posn = q1 + dq*u0;
      if( std::abs(u0-u1) < EPS )
        return LINE_CROSS;
      if(posn2 != 0)
        *posn2 = q1 + dq*u1;
      return LINE_OVERLAP;
    }

  double s = cross(dq, dpq)*(1/denom);
  if(s < -EPS || s > (1+EPS))
    return LINE_NOOVERLAP;
  double t = cross(dp, dpq)*(1/denom);
  if(t < -EPS || t > (1+EPS))
    return LINE_NOOVERLAP;

  if(posn != 0)
    *posn = p1 + dp * std::max(std::min(s, 1.), 0.);

  return LINE_CROSS;
}

// which half of line is point on (0 if it is on the line)
inline int ptsign(Vec2 p1, Vec2 p2, Vec2 p3)
{
  double v = (p1(0)-p3(0))*(p2(1)-p3(1)) - (p2(0)-p3(0))*(p1(1)-p3(1));
  return std::abs(v)<EPS ? 0 : (v<0 ? -1 : 1);
}

// is pt inside triangle v1,v2,v2?
// http://stackoverflow.com/questions/2049582/how-to-determine-a-point-
// in-a-triangle
// returns false if the point lies on the edge
bool point_in_triangle(Vec2 pt, Vec2 v1, Vec2 v2, Vec2 v3)
{
  int s1 = ptsign(pt, v1, v2);
  int s2 = ptsign(pt, v2, v3);
  if(s1 != s2)
    return 0;
  int s3 = ptsign(pt, v3, v1);
  return s2 == s3;
}

// this class is used for the nodes of the graph
typedef Vec2 Node;

// joined Nodes are called a Link
// Link stores indices two nodes within a sorted integer value
// allowing easy comparison
struct Link
{
  Link(unsigned n1, unsigned n2)
    : _val(std::min(n1,n2) | std::max(n1,n2) << NODE_BITS)
  {}
  Link() {}

  bool operator ==(const Link& other) const
  {
    return _val == other._val;
  }
  bool operator !=(const Link& other) const
  {
    return _val != other._val;
  }
  bool operator<(const Link& other) const
  {
    return _val < other._val;
  }
  unsigned node1() const { return _val & ((1<<NODE_BITS)-1); }
  unsigned node2() const { return _val >> NODE_BITS; }

  friend class LinkSet;

private:
  unsigned _val;
};

// type for a list of Nodes which make up the graph
typedef FixedVector<Node,MAXNODES> Nodes;

// the Graph is a list of Links
typedef FixedVector<Link,MAXLINKS> Graph;

// a set of links: to avoid repeating calculations
// initially empty
struct LinkSet
{
  // add the link to the set
  void add(Link link)
  {
    _set[link._val] = 1;
  }
  // does the link exist
  bool includes(Link link)
  {
    return _set[link._val];
  }

private:
  std::bitset< (1<<(2*NODE_BITS)) > _set;
};

// are two points close?
inline bool close_pts(const Vec2& a, const Vec2& b)
{
  //std::cout << std::abs(a(0)-b(0)) << ' ' << std::abs(a(1)-b(1)) << '\n';
  return std::abs(a(0)-b(0))<CLOSEDIST && std::abs(a(1)-b(1))<CLOSEDIST;
}

// return existing node or add a new one corresponding to pt
inline unsigned getOrAddNode(Nodes& nodes, Vec2 pt)
{
  for(unsigned i=0; i<nodes.size(); ++i)
    {
      if(close_pts(nodes[i], pt))
        return i;
    }
  // assert(nodes.size() < MAXNODES);
  //std::cout << "add " << pt(0) << " " << pt(1) << "\n";
  nodes.push_back(pt);
  return nodes.size()-1;
}

// snap points to lines
void snap_points(const Vec2* lpts, unsigned nlpts, Vec2* pts, unsigned npts)
{
  for(unsigned lineidx=0; lineidx<nlpts; ++lineidx)
    {
      const Vec2 lpt1 = lpts[lineidx];
      const Vec2 lpt2 = lpts[(lineidx+1) % nlpts];
      for(unsigned ptidx=0; ptidx<npts; ++ptidx)
        {
          Vec2 pt = pts[ptidx];

          // check for bounds
          if(pt(0) < std::min(lpt1(0), lpt2(0)) || pt(0) > std::max(lpt1(0), lpt2(0)) ||
             pt(1) < std::min(lpt1(1), lpt2(1)) || pt(1) > std::max(lpt1(1), lpt2(1)))
            continue;

          if( std::abs(lpt1(0)-lpt2(0)) > std::abs(lpt1(1)-lpt2(1)) )
            {
              double grad = (lpt2(1)-lpt1(1))/(lpt2(0)-lpt1(0));
              double ypt = (pt(0)-lpt1(0))*grad + lpt1(1);
              if( std::abs(ypt-pt(1)) < SNAPDIST )
                {
                  //std::cout << "snapping " << ypt << " " << pt(1) << '\n';
                  pts[ptidx](1) = ypt;
                }
            }
          else
            {
              double grad = (lpt2(0)-lpt1(0))/(lpt2(1)-lpt1(1));
              double xpt = (pt(1)-lpt1(1))*grad + lpt1(0);
              if( std::abs(xpt-pt(0)) < SNAPDIST )
                {
                  //std::cout << "snapping " << xpt << " " << pt(0) << '\n';
                  pts[ptidx](0) = xpt;
                }
            }
        }
    }
}

// convert an input polygon (pts*npts) to Graph and Nodes objects
void poly2graph(Graph& outgraph, Nodes& nodes, const Vec2* pts, unsigned npts)
{
  unsigned lastnode=0; // init gets rid of warning :-(
  for(unsigned pi=0; pi!=(npts+1); ++pi)
    {
      Vec2 pt=pts[pi % npts];

      // find/add node: this is highly inefficient for large numbers of N
      unsigned newnode = getOrAddNode(nodes, pt);

      // create link between this and next point
      if(pi != 0 && lastnode != newnode)
        outgraph.push_back( Link(lastnode, newnode) );
      lastnode = newnode;
    }
}

// this is hard work: break Link given by linkidx by points posn1, posn2
// which are along link (or may be at ends)
void insert_overlapped_nodes(Graph& graph, Nodes& nodes, unsigned linkidx,
                             const Vec2& posn1, const Vec2& posn2)
{
  Link li = graph[linkidx];
  Vec2 vlink = nodes[li.node2()]-nodes[li.node1()];
  Vec2 vposn1 = posn1 - nodes[li.node1()];
  Vec2 vposn2 = posn2 - nodes[li.node1()];

  // work out fraction along that breaks occur
  double frac1, frac2;
  if(std::abs(vlink(0)) > std::abs(vlink(1)))
    {
      frac1 = vposn1(0) * (1/vlink(0));
      frac2 = vposn2(0) * (1/vlink(0));
    }
  else
    {
      frac1 = vposn1(1) * (1/vlink(1));
      frac2 = vposn2(1) * (1/vlink(1));
    }
  if(frac1 > frac2)
    std::swap(frac1, frac2);

  //std::cout << "fracs " << frac1 << " " << frac2 << "\n";

  // construct valid links
  unsigned linkct = 0;
  Link links[3];

  unsigned newn1 = getOrAddNode(nodes, nodes[li.node1()]+vlink*frac1);
  if(newn1 != li.node1())
    links[linkct++] = Link(li.node1(), newn1);

  //std::cout << "pt " << nodes[newn1](0) << " " << nodes[newn1](1) << "\n";

  unsigned newn2 = getOrAddNode(nodes, nodes[li.node1()]+vlink*frac2);
  if(newn2 != newn1)
    links[linkct++] = Link(newn1, newn2);

  if(li.node2() != newn2)
    links[linkct++] = Link(newn2, li.node2());

  // insert into graph, replacing existing link
  if(linkct > 0)
    {
      graph[linkidx] = links[0];
      for(unsigned i=1; i<linkct; ++i)
        graph.push_back(links[i]);
    }
}

void identify_split_nodes(Graph& graph1, Graph& graph2, Nodes &nodes)
{
  //std::cout << "identify split nodes\n";
  Vec2 posn1, posn2;
  for(unsigned i1=0; i1<graph1.size(); ++i1)
    {
      for(unsigned i2=0; i2<graph2.size() ; ++i2)
      {
        const Link l1 = graph1[i1];
        const Link l2 = graph2[i2];
        if(l1 == l2)
          continue;

        // std::cout
        //   << nodes[l1.node1()](0) << ' ' << nodes[l1.node1()](1) << ", "
        //   << nodes[l1.node2()](0) << ' ' << nodes[l1.node2()](1) << "  "
        //   << nodes[l2.node1()](0) << ' ' << nodes[l2.node1()](1) << ", "
        //   << nodes[l2.node2()](0) << ' ' << nodes[l2.node2()](1) << "\n";

        isect isectv = calcLine2DIntersect
          (nodes[l1.node1()], nodes[l1.node2()],
           nodes[l2.node1()], nodes[l2.node2()],
           &posn1, &posn2);

        if(isectv == LINE_CROSS)
          {
            //std::cout << "x " << posn1(0) << ' ' << posn1(1) << '\n';
            unsigned newnode = getOrAddNode(nodes, posn1);
            if(newnode != l1.node1() && newnode != l1.node2())
              {
                graph1[i1] = Link(l1.node1(), newnode);
                graph1.push_back(Link(newnode, l1.node2()));
              }
            if(newnode != l2.node1() && newnode != l2.node2())
              {
                graph2[i2] = Link(l2.node1(), newnode);
                graph2.push_back(Link(newnode, l2.node2()));
              }
          }
        else if(isectv == LINE_OVERLAP)
          {
            //std::cout << "ol1 " << posn1(0) << ' ' << posn1(1) << '\n';
            //std::cout << "ol2 " << posn2(0) << ' ' << posn2(1) << '\n';
            insert_overlapped_nodes(graph1, nodes, i1, posn1, posn2);
            insert_overlapped_nodes(graph2, nodes, i2, posn1, posn2);
          }
      }
    }
}

// merge ingraph into outgraph
void merge_graph(Graph& outgraph, Nodes& nodes, const Graph& tomerge)
{
  //std::cout << "merge graph\n";
  // stage1: calculate intersection of every Link with every Link to
  // build up node list to include every intersection
  Graph mergegraph(tomerge);
  identify_split_nodes(outgraph, mergegraph, nodes);

  // stage2: merge all the links
  for(unsigned i=0; i<mergegraph.size(); ++i)
    outgraph.push_back(mergegraph[i]);
}

// represent a triangle with nodes n1,n2,n3
// this can be compared to other triangles
class Triangle
{
public:
  Triangle() {}
  Triangle(unsigned n1, unsigned n2, unsigned n3)
  {
    unsigned low = (n1<n2 && n1<n3) ? n1 : (n2<n3 ? n2 : n3);  //min
    unsigned hi  = (n1>n2 && n1>n3) ? n1 : (n2>n3 ? n2 : n3);  //max
    unsigned med = (n1!=low && n1!=hi) ? n1 : ((n2!=low && n2!=hi) ? n2 : n3); //mid
    _val = low | med << NODE_BITS | hi << (NODE_BITS*2);
  }
  bool operator ==(const Triangle& other) const
  {
    return _val == other._val;
  }
  unsigned n1() const { return _val & ((1<<NODE_BITS)-1); }
  unsigned n2() const { return (_val >> NODE_BITS) & ((1<<NODE_BITS)-1); }
  unsigned n3() const { return _val >> (2*NODE_BITS); }

private:
  unsigned _val;
};


typedef FixedVector<Triangle, MAXTRIANGLES> TriangleVec;

// is this triangle already in the list?
inline bool triangle_exists(const TriangleVec& v, Triangle t)
{
  for(unsigned i=0; i<v.size(); ++i)
    if( v[i] == t )
      return 1;
  return 0;
}

// compute area of triangle
inline double triangle_area(Vec2 p1, Vec2 p2, Vec2 p3)
{
  return 0.5*std::abs(cross(p1,p2) + cross(p2,p3) + cross(p3,p1));
}

// check whether there are no other links crossing or overlapping this one
// returns isect value, depending on whether other lines overlap or cross
bool check_invalid_link(const Graph& graph, const Nodes& nodes, Link link)
{
  Vec2 posn;

  unsigned n1=link.node1();
  unsigned n2=link.node2();

  for(unsigned i=0; i<graph.size(); ++i)
    {
      if(graph[i] != link)
        {
          unsigned ln1 = graph[i].node1();
          unsigned ln2 = graph[i].node2();

          isect isectv =
            calcLine2DIntersect(nodes[n1], nodes[n2], nodes[ln1], nodes[ln2], &posn);

          if(isectv == LINE_OVERLAP)
            return 1;
          else if(isectv == LINE_CROSS)
            {
              // cross point is not one of the end points
              if(!close_pts(posn, nodes[n1]) && !close_pts(posn, nodes[n2]) &&
                 !close_pts(posn, nodes[ln1]) && !close_pts(posn, nodes[ln2]))
                return 1;
            }
        }
    }
  return 0;
}

// convert Graph and Nodes into a list of output triangles
void make_triangles(Graph& graph, const Nodes& nodes, TriangleVec &triangles)
{
  // don't double-check links by keeping track of what's already
  // included in two sets
  LinkSet invalid_links;
  LinkSet valid_links;
  for(unsigned li=0; li<graph.size(); ++li)
    valid_links.add(graph[li]);

  // iterate over a starting link in the graph
  for(unsigned li=0; li<graph.size(); ++li)
    {
      unsigned n1 = graph[li].node1();
      unsigned n2 = graph[li].node2();

      // iterate over possible 3rd Nodes
      for(unsigned oni=0; oni<nodes.size(); ++oni)
        {
          if(n1 == oni || n2 == oni)
            continue;

          Triangle tri = Triangle(n1, n2, oni);

          // skip existing triangles
          if(triangle_exists(triangles, tri))
            continue;

          // check whether the links are allowed.
          // if there's a link in the graph already, then it's ok.
          // otherwise check whether it would cross an existing link.
          // bad links are cached, as are valid links
          Link l1(n1, oni);
          if(!valid_links.includes(l1))
            {
              if(invalid_links.includes(l1))
                continue;
              if(check_invalid_link(graph, nodes, l1))
                {
                  invalid_links.add(l1);
                  continue;
                }
            }
          Link l2(n2, oni);
          if(!valid_links.includes(l2))
            {
              if(invalid_links.includes(l2))
                continue;
              if(check_invalid_link(graph, nodes, l2))
                {
                  invalid_links.add(l2);
                  continue;
                }
            }

          // add new triangle
          triangles.push_back(tri);

          // add new link if required
          if(!valid_links.includes(l1))
            {
              graph.push_back(l1);
              valid_links.add(l1);
            }
          if(!valid_links.includes(l2))
            {
              graph.push_back(l2);
              valid_links.add(l2);
            }
        }
    }


}

// look for nodes inside triangles and don't include those triangles
void filter_triangles(const TriangleVec& tin, const Nodes& nodes, TriangleVec& tout)
{
  for(unsigned t=0; t<tin.size(); ++t)
    {
      Triangle tri(tin[t]);
      bool bad=0;
      for(unsigned n=0; n<nodes.size(); ++n)
        {
          if( n == tri.n1() || n == tri.n2() || n == tri.n3() )
            continue;

          if(point_in_triangle(nodes[n], nodes[tri.n1()], nodes[tri.n2()],
                               nodes[tri.n3()]))
            {
              bad=1; break;
            }
        }
      if(!bad)
        tout.push_back(tri);
    }
}

// void dumpgraph(const Graph& g)
// {
//   std::cout << "Graph " << g.size() << " (\n";
//   for(unsigned i=0; i!=g.size(); ++i)
//     std::cout << ' ' << g[i].node1()
//               << ' ' << g[i].node2()
//               << '\n';
//   std::cout << ")\n";
// }

  // area any of the points of tri1 in tri2?
  // return true if no overlap
  // following doesn't work if points are on edges!
  // bool check_no_overlap(const Triangle2D& a, const Triangle2D& b)
  // {
  //   for(unsigned i=0; i<3; i++)
  //     {
  //       if(point_in_triangle(a[i], b[0], b[1], b[2]))
  //         {
  //           if(!close_pts(a[i], b[0]) &&
  //              !close_pts(a[i], b[1]) &&
  //              !close_pts(a[i], b[2]))
  //             return 0;
  //         }
  //     }
  //   return 1;
  // }

} // namespace

bool clip_triangles_2d(const Triangle2D& tri1, const Triangle2D& tri2,
                       std::vector<Triangle2D>& tris_both,
                       std::vector<Triangle2D>& tris1,
                       std::vector<Triangle2D>& tris2)
{

  // for(unsigned i=0; i<4; i++)
  //   std::cout << tri1[i%3](0) << ' ' << tri1[i%3](1) << '\n';
  // std::cout << "nan nan\n";
  // for(unsigned i=0; i<4; i++)
  //   std::cout << tri2[i%3](0) << ' ' << tri2[i%3](1) << '\n';

  Triangle2D tri1_snap(tri1);
  Triangle2D tri2_snap(tri2);
  snap_points(tri1_snap.pts, 3, tri2_snap.pts, 3);
  snap_points(tri2_snap.pts, 3, tri1_snap.pts, 3);

  // std::cout << "snap\n";
  // for(unsigned i=0; i<4; i++)
  //   std::cout << tri1[i%3](0) << ' ' << tri1[i%3](1) << '\n';
  // std::cout << "nan nan\n";
  // for(unsigned i=0; i<4; i++)
  //   std::cout << tri2[i%3](0) << ' ' << tri2[i%3](1) << '\n';

  // std::cout << triangle_area(tri1_snap) << ' ' << triangle_area(tri2_snap) << '\n';

  // snapping might make triangle disappear
  if(triangle_area(tri1_snap)<1e-8 || triangle_area(tri2_snap)<1e-8)
    return 0;

  // std::cout << "tri1 cross: "
  //           << cross(tri1[0]-tri1[1], tri1[0]-tri1[2]) << ' '
  //           << cross(tri1[0]-tri1[1], tri1[1]-tri1[2]) << ' '
  //           << cross(tri1[0]-tri1[2], tri1[1]-tri1[2]) << '\n';
  // std::cout << "tri2 cross: "
  //           << cross(tri2[0]-tri2[1], tri2[0]-tri2[2]) << ' '
  //           << cross(tri2[0]-tri2[1], tri2[1]-tri2[2]) << ' '
  //           << cross(tri2[0]-tri2[2], tri2[1]-tri2[2]) << '\n';

  Nodes nodes;
  Graph g1;
  poly2graph(g1, nodes, tri1_snap.pts, 3);
  Graph g2;
  poly2graph(g2, nodes, tri2_snap.pts, 3);
  merge_graph(g1, nodes, g2);

  TriangleVec triangles_pre;
  make_triangles(g1, nodes, triangles_pre);
  TriangleVec triangles;
  filter_triangles(triangles_pre, nodes, triangles);

  for(unsigned i=0; i<triangles.size(); ++i)
    {
      Triangle t(triangles[i]);
      Triangle2D tri(nodes[t.n1()], nodes[t.n2()], nodes[t.n3()]);

      Vec2 cenpt = (tri[0]+tri[1]+tri[2])*(1./3.);
      bool intri1 = point_in_triangle(cenpt, tri1_snap[0],
                                      tri1_snap[1], tri1_snap[2]);
      bool intri2 = point_in_triangle(cenpt, tri2_snap[0],
                                      tri2_snap[1], tri2_snap[2]);

      // std::cout
      //   << intri1 << ' ' << intri2 << ' '
      //   << tri[0](0) << ',' << tri[1](0) << ","
      //   << tri[2](0) << ' ' << tri[0](1) << ","
      //   << tri[1](1) << ',' << tri[2](1) << "\n";

      if(intri1 && intri2)
        tris_both.push_back(tri);
      if(intri1)
        tris1.push_back(tri);
      if(intri2)
        tris2.push_back(tri);
    }

  return 1;
}

#if 0
int main()
{
  Triangle2D tri1(Vec2(-0.2317, 0.074085),
                  Vec2(-0.231844, 0.0656526),
                  Vec2(-0.246207, 0.0865563));

  Triangle2D tri2(Vec2(-0.246274, 0.0867),
                  Vec2(-0.231679, 0.0750825),
                  Vec2(-0.231558, 0.0806746));

  std::vector<Triangle2D> tris1, tris2, tris_both;
  clip_triangles_2d(tri1, tri2, tris_both, tris1, tris2);


  return 0;
}
#endif
