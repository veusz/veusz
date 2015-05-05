#include <algorithm>
#include <bitset>
#include <cmath>

#include <cstdlib>
#include <iostream>
#include <fstream>

#include "mmaths.h"
#include "fixedvector.h"

//typedef std::vector FixedVector;

//#include <vector>
//template<class T, unsigned N> class FixedVector : public std::vector<T> {};

// ignore differences in doubles < this
#define EPS 1e-8
// maximum number of points/nodes in polygons
#define NODE_BITS 4 // Max of 16 nodes (0..2**NODE_BITS-1)
#define MAXNODES (1<<NODE_BITS)
#define MAXLINKS 32
#define MAXTRIANGLES 32

enum isect { LINE_NOOVERLAP, LINE_CROSS, LINE_OVERLAP };

// Do the two line segments p1->p2, q1->q2 cross or overlap?
// return LINE_NOOVERLAP if no overlap
//        LINE_CROSS if they cross somewhere
//        LINE_OVERLAP if they lie on top of each other partially
// if posn != 0, return crossing position if LINE_CROSS
// if LINE_OVERLAP the two end points of overlap are returned in posn and posn2
// Assumes that the line segments are finite.
// Ignores intersections inside EPS of the start and end points
isect calcLine2DIntersect(Vec2 p1, Vec2 p2, Vec2 q1, Vec2 q2, Vec2* posn=0, Vec2* posn2=0)
{
  Vec2 dp = p2-p1;
  Vec2 dq = q2-q1;
  Vec2 dpq = p1-q1;
  double denom = cross(dp, dq);

  // parallel vectors or points below
  if(std::abs(denom) < EPS)
    {
      if( std::abs(cross(dp, dpq)) > EPS || std::abs(cross(dq, dpq)) > EPS )
        return LINE_NOOVERLAP;

      // colinear segments - do they overlap?
      double u0, u1;
      Vec2 dpq2 = p2-q1;
      if(std::abs(dq(0)) > EPS)
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

      if( u0>(1-EPS) || u1<EPS )
        return LINE_NOOVERLAP;

      u0 = std::max(u0, 0.);
      u1 = std::min(u1, 1.);
      if(posn != 0)
        *posn = q1 + dq*u0;
      if( std::abs(u0-u1) < EPS )
        // not sure this can really happen, unless the u0/u1
        // constraint above is lifted
        return LINE_CROSS;
      if(posn2 != 0)
        *posn2 = q1 + dq*u1;
      return LINE_OVERLAP;
    }

  double s = cross(dq, dpq)*(1/denom);
  if(s < EPS || s > (1-EPS))
    return LINE_NOOVERLAP;
  double t = cross(dp, dpq)*(1/denom);
  if(t < EPS || t > (1-EPS))
    return LINE_NOOVERLAP;

  if(posn != 0)
    *posn = p1 + dp*s;

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
bool pointInTriangle(Vec2 pt, Vec2 v1, Vec2 v2, Vec2 v3)
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

// are two vectors close?
inline bool close(const Vec2& a, const Vec2& b)
{
  return std::abs(a(0)-b(0))<EPS && std::abs(a(1)-b(1))<EPS;
}

// return existing node or add a new one corresponding to pt
inline unsigned getOrAddNode(Nodes& nodes, Vec2 pt)
{
  for(unsigned i=0; i<nodes.size(); ++i)
    {
      if(close(nodes[i], pt))
        return i;
    }
  nodes.push_back(pt);
  return nodes.size()-1;
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

#if 0
// this is hard work: break Link given by linkidx by points posn1, posn2
// which are along link (or may be at ends)
void insert_overlapped_nodes(Graph& graph, Nodes& nodes, unsigned linkidx,
                             const Vec2& posn1, const Vec2& posn2)
{
  // this is harder, as we to break the lines once or twice
  Link li = graph[linkidx];
  Vec2 vlink = nodes[li.node2()]-nodes[li.node1()];
  Vec2 vposn1 = posn1 - nodes[li.node1()];
  Vec2 vposn2 = posn2 - nodes[li.node1()];
  // work out fraction along that breaks occur
  double frac1, frac2;
  if(std::abs(vlink(0)) > EPS)
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

  // ignore frac1
  if(frac1 < EPS || frac1 >= (1-EPS))
    {
      // frac1 is invalid, so shift frac2 into it
      frac1 = frac2; frac2 = -1;
    }

  std::cout << "fracs: " << frac1 << ' ' << frac2 << '\n';

  unsigned newn1;
  if(frac1 > EPS && frac1 < (1-EPS))
    {
      newn1 = getOrAddNode(nodes, nodes[li.node1()]+vlink*frac1);
      graph[linkidx] = Link(li.node1(), newn1);
      std::cout << "add one a\n";
    }
  else
    {
      // both fracs are invalid
      return;
    }

  if(frac2 > EPS && frac2 < (1-EPS) && std::abs(frac1-frac2) > EPS)
    {
      // two break points
      unsigned newn2 = getOrAddNode(nodes, nodes[li.node1()]+vlink*frac2);
      graph.push_back(Link(newn1, newn2));
      graph.push_back(Link(newn2, li.node2()));
      std::cout << "add two\n";
    }
  else
    {
      // one break point
      graph.push_back(Link(newn1, li.node2()));
      std::cout << "add 1b\n";
    }
}
#endif

void identify_nodes(const Graph& graph1, const Graph& graph2, Nodes &nodes)
{
  Vec2 posn1, posn2;
  for(unsigned i1=0; i1<graph1.size(); ++i1)
    {
      const Link l1 = graph1[i1];
      for(unsigned i2=0; i2<graph2.size(); ++i2)
      {
        const Link l2 = graph2[i2];
        if(l1 == l2)
          continue;

        isect isectv = calcLine2DIntersect
          (nodes[l1.node1()], nodes[l1.node2()],
           nodes[l2.node1()], nodes[l2.node2()],
           &posn1, &posn2);

        if(isectv == LINE_CROSS)
          {
            getOrAddNode(nodes, posn1);
          }
        else if(isectv == LINE_OVERLAP)
          {
            getOrAddNode(nodes, posn1);
            getOrAddNode(nodes, posn2);
          }
      }
    }
}

// is point on line (excluding endpoints)?
inline bool point_on_line(const Vec2 &p1, const Vec2 &p2, const Vec2 &p)
{
  Vec2 dp12 = p2-p1;
  Vec2 dp = p-p1;

  if( std::abs(dp12(0)) > std::abs(dp12(1)) )
    {
      double ratio = dp(0)/dp12(0);
      if(ratio < EPS || ratio > (1-EPS))
        return 0;
      return std::abs(dp12(1)*ratio-dp(1)) < EPS;
    }
  else
    {
      double ratio = dp(1)/dp12(1);
      if(ratio < EPS || ratio > (1-EPS))
        return 0;
      return std::abs(dp12(0)*ratio-dp(0)) < EPS;
    }
}

void break_links(Graph& graph, const Nodes& nodes)
{
  for(unsigned lidx=0; lidx<graph.size(); ++lidx)
    {
      for(unsigned nidx=0; nidx<nodes.size(); ++nidx)
        {
          // graph[lidx] may change below!
          unsigned n1 = graph[lidx].node1();
          unsigned n2 = graph[lidx].node2();

          // node is on link already
          if(nidx == n1 || nidx == n2)
            continue;

          Vec2 p1 = nodes[n1];
          Vec2 p2 = nodes[n2];

          if(point_on_line(p1, p2, nodes[nidx]))
            {
              graph[lidx] = Link(n1, nidx);
              graph.push_back(Link(nidx, n2));
            }
        }
    }
}


// merge ingraph into outgraph
void merge_graph(Graph& outgraph, Nodes& nodes, const Graph& mergegraph)
{
  // stage1: calculate intersection of every Link with every Link to
  // build up node list to include every intersection
  identify_nodes(outgraph, mergegraph, nodes);

  // stage2: merge all the links
  for(unsigned i=0; i<mergegraph.size(); ++i)
    outgraph.push_back(mergegraph[i]);

  // stage3: check whether nodes lie along Links and break accordingly
  break_links(outgraph, nodes);
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
  unsigned n1=link.node1();
  unsigned n2=link.node2();

  for(unsigned i=0; i<graph.size(); ++i)
    {
      if(graph[i] != link)
        {
          unsigned ln1 = graph[i].node1();
          unsigned ln2 = graph[i].node2();
          isect isectv =
            calcLine2DIntersect(nodes[n1], nodes[n2], nodes[ln1], nodes[ln2]);

          if(isectv != LINE_NOOVERLAP)
            return 1;
        }
    }
  return 0;
}

// convert Graph and Nodes into a list of output triangles
void makeTriangles(Graph& graph, const Nodes& nodes, TriangleVec &triangles)
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

          if(pointInTriangle(nodes[n], nodes[tri.n1()], nodes[tri.n2()], nodes[tri.n3()]))
            {
              bad=1; break;
            }
        }
      if(!bad)
        tout.push_back(tri);
    }
}

void dumpgraph(const Graph& g)
{
  std::cout << "Graph " << g.size() << " (\n";
  for(unsigned i=0; i!=g.size(); ++i)
    std::cout << ' ' << g[i].node1()
              << ' ' << g[i].node2()
              << '\n';
  std::cout << ")\n";
}

int main()
{

  Vec2 p1[3];
  Vec2 p2[3];

  for(unsigned ct=0; ct<100000; ++ct)
    {
      //std::cout << ct << '\n';

      for(unsigned i=0; i<3; ++i)
        {
          p1[i](0) = std::rand()*(1./RAND_MAX);
          p1[i](1) = std::rand()*(1./RAND_MAX);
          p2[i](0) = std::rand()*(1./RAND_MAX);
          p2[i](1) = std::rand()*(1./RAND_MAX);
        }
      if(std::rand()*(1./RAND_MAX) < 0.1)
        {
          unsigned i1 = std::rand() % 3;
          unsigned i2 = std::rand() % 3;
          p1[i1] = p2[i2];
        }
      if(std::rand()*(1./RAND_MAX) < 0.1)
        {
          unsigned i1 = std::rand() % 3;
          unsigned i2 = std::rand() % 3;
          p1[i1] = p2[i2];
        }

      Nodes nodes;
      Graph g1;
      poly2graph(g1, nodes, p1, 3);
      Graph g2;
      poly2graph(g2, nodes, p2, 3);
      merge_graph(g1, nodes, g2);
      TriangleVec triangles_pre;
      makeTriangles(g1, nodes, triangles_pre);

      TriangleVec triangles;
      filter_triangles(triangles_pre, nodes, triangles);

      double a1=0, a2=0;
      for(unsigned i=0; i<triangles.size(); ++i)
        {
          Triangle t(triangles[i]);
          Vec2 pt((nodes[t.n1()](0)+nodes[t.n2()](0)+nodes[t.n3()](0))*(1./3.),
                  (nodes[t.n1()](1)+nodes[t.n2()](1)+nodes[t.n3()](1))*(1./3.));

          double area = triangle_area(nodes[t.n1()], nodes[t.n2()], nodes[t.n3()]);
          if(pointInTriangle(pt, p1[0], p1[1], p1[2]))
            {
              a1 += area;
            }
          if(pointInTriangle(pt, p2[0], p2[1], p2[2]))
            {
              a2 += area;
            }
        }

      double a1tot = triangle_area(p1[0], p1[1], p1[2]);
      double a2tot = triangle_area(p2[0], p2[1], p2[2]);

      if((std::abs(a1-a1tot)>1e-4) || std::abs(a2-a2tot)>1e-4)
        {
          std::cout << "num " << ct << '\n';

          std::ofstream f("tris_in.dat");
          f << p1[0](0) << ' ' << p1[0](1) << '\n'
            << p1[1](0) << ' ' << p1[1](1) << '\n'
            << p1[2](0) << ' ' << p1[2](1) << '\n'
            << p1[0](0) << ' ' << p1[0](1) << '\n'
            << "nan nan\n"
            << p2[0](0) << ' ' << p2[0](1) << '\n'
            << p2[1](0) << ' ' << p2[1](1) << '\n'
            << p2[2](0) << ' ' << p2[2](1) << '\n'
            << p2[0](0) << ' ' << p2[0](1) << '\n';


          std::ofstream tris("tris_out.dat");
          for(unsigned i=0; i<triangles.size(); ++i)
            {
              Triangle t = triangles[i];
              tris << "# Tri " << t.n1() << ' ' << t.n2() << ' ' << t.n3() << '\n'
                   << nodes[t.n1()](0) << ' ' << nodes[t.n1()](1) << '\n'
                   << nodes[t.n2()](0) << ' ' << nodes[t.n2()](1) << '\n'
                   << nodes[t.n3()](0) << ' ' << nodes[t.n3()](1) << '\n'
                   << nodes[t.n1()](0) << ' ' << nodes[t.n1()](1) << '\n'
                   << "nan nan\n";
            }
          std::cout << "Nodes " << nodes.size() << '\n';
          for(unsigned i=0; i!=nodes.size(); ++i)
            std::cout << i << ' ' << nodes[i](0) << ' ' << nodes[i](1) << '\n';
          std::cout << '\n';

          dumpgraph(g1);

          std::cout << '\n';
          for(unsigned i=0; i<triangles.size(); ++i)
            {
              Triangle t = triangles[i];
              std::cout << t.n1() << ' ' << t.n2() << ' ' << t.n3() << '\n';
            }

          std::cout << "area 1 " << a1 << ' ' << a1tot << '\n';
          std::cout << "area 2 " << a2 << ' ' << a2tot << '\n';

          break;

        }

    }

  return 0;
}


#if 0

int main()
{
  Vec2 p1[3];
  Vec2 p2[3];
  // p1[0] = Vec2(0,0);
  // p1[1] = Vec2(0,1);
  // p1[2] = Vec2(1,0);
  // p2[1] = Vec2(0.1,0.1);
  // p2[0] = Vec2(0.1,1.1);
  // p2[2] = Vec2(1.1,0.1);

  // buggy
  p1[0] = Vec2(0,0);
  p1[1] = Vec2(0,1);
  p1[2] = Vec2(1,0);
  p2[1] = Vec2(0.3,0.3);
  p2[0] = Vec2(0.3,0.8);
  p2[2] = Vec2(0.8,0.3);

  {
    std::ofstream f("tris_in.dat");
    f << p1[0](0) << ' ' << p1[0](1) << '\n'
      << p1[1](0) << ' ' << p1[1](1) << '\n'
      << p1[2](0) << ' ' << p1[2](1) << '\n'
      << p1[0](0) << ' ' << p1[0](1) << '\n'
      << "nan nan\n"
      << p2[0](0) << ' ' << p2[0](1) << '\n'
      << p2[1](0) << ' ' << p2[1](1) << '\n'
      << p2[2](0) << ' ' << p2[2](1) << '\n'
      << p2[0](0) << ' ' << p2[0](1) << '\n';
  }

  Nodes nodes;
  Graph g1;
  poly2graph(g1, nodes, p1, 3);
  dumpgraph(g1);
  Graph g2;
  poly2graph(g2, nodes, p2, 3);
  dumpgraph(g2);

  merge_graph(g1, nodes, g2);
  std::cout << "size1 " << g1.size() << '\n';
  std::cout << "size2 " << g1.size() << '\n';

  dumpgraph(g1);

  std::cout << "Nodes " << nodes.size() << '\n';
  for(unsigned i=0; i!=nodes.size(); ++i)
    std::cout << nodes[i](0) << ' ' << nodes[i](1) << '\n';

  //scangraph(g1, nodes);

  makeTriangles(g1, nodes);

  return 0;
}

#endif
