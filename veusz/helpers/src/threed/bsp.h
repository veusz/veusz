// -*-c++-*-

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

#ifndef BSP_H
#define BSP_H

#include <vector>
#include "fragment.h"

typedef std::vector<unsigned> IdxVector;

struct BSPNode
{
  IdxVector fragidxs;
  BSPNode* less;
  BSPNode* more;
};

BSPNode* buildBSPTree(FragmentVector& fragvec);
void deleteBSPTree(BSPNode *node);

class BSPTreeIterate
{
public:
  BSPTreeIterate(FragmentVector& _vec)
    : vec(_vec)
  {}
  void operator()(Fragment& f) {};

  void iterate(BSPNode *root)
  {
    if(root->less)
      iterate(root->less);

    for(unsigned i=0, s=root->fragidxs.size(); i<s; ++i)
      operator()(vec[root->fragidxs[i]]);

    if(root->more)
      iterate(root->more);
  }

  FragmentVector& vec;
};

inline void buildBSPDepthList(BSPNode* node, IdxVector& out)
{
  if(node->less)
    buildBSPDepthList(node->less, out);

  for(unsigned i=0, s=node->fragidxs.size(); i<s; ++i)
    out.push_back(node->fragidxs[i]);

  if(node->more)
    buildBSPDepthList(node->more, out);
}


#endif
