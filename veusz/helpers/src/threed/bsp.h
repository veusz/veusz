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

  unsigned minfragidxidx, nfrags;
  unsigned frontidx, backidx;
};

class BSPBuilder
{
public:
  BSPBuilder(FragmentVector& fragvec);
  IdxVector getFragmentIdxs() const;

  std::vector<BSPRecord> bsp_recs;
  IdxVector frag_idxs;
};


#endif
