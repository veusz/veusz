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

#define EMPTY_BSP_IDX (std::numeric_limits<unsigned>::max())

struct BSPRecord
{
  BSPRecord()
    : minfragidxidx(0), nfrags(0),
      frontidx(EMPTY_BSP_IDX), backidx(EMPTY_BSP_IDX)
  {
  }

  // fragments stored in this node, in terms of the index to an array
  // of indexes, frag_idxs
  unsigned minfragidxidx, nfrags;
  // indices in bsp_recs to the BSPRecord items in front and behind
  unsigned frontidx, backidx;
};

// This class defines a specialised Binary Space Paritioning (BSP)
// buliding routine. 3D space is split recursively by planes to
// separate objects into front and back entries. The idea is to only
// use the BSP tree _once_, which is unlike normal uses of BSP. It is
// used to create a robust back->front ordering for a particular
// viewing direction. To avoid lots of dynamic memory allocation and
// to reduce overheads, the nodes in the BSP tree are stored in a
// vector.

class BSPBuilder
{
public:
  // construct the BSP tree from the fragments given and a particular
  // viewing direction
  BSPBuilder(FragmentVector& fragvec, Vec3 viewdirn);

  // return a vector of fragment indexes in drawing order
  IdxVector getFragmentIdxs(const FragmentVector& fragvec) const;

  // the nodes in the tree
  std::vector<BSPRecord> bsp_recs;
  // vector of indices to the fragments vector
  IdxVector frag_idxs;
};


#endif
