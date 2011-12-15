//    Copyright (C) 2011 Jeremy S. Sanders
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

#include "numpyfuncs.h"
#include <limits>
#include "isnan.h"

void binData(const Numpy1DObj& indata, int binning,
	     bool average,
	     int* numoutbins, double** outdata)
{
  int size = indata.dim / binning;
  if( indata.dim % binning != 0 )
    ++size;

  *numoutbins = size;
  double *out = new double[size];
  *outdata = out;

  double sum = 0.;
  int ct = 0;
  for(int i = 0 ; i < indata.dim; ++i)
    {
      // include new data
      if ( isFinite( indata(i) ) )
	{
	  sum += indata(i);
	  ct += 1;
	}

      // every bin or at end of array
      if ( i % binning == binning-1 || i == indata.dim-1 )
	{
	  if( ct == 0 )
	    {
	      out[i / binning] = std::numeric_limits<double>::quiet_NaN();
	    }
	  else
	    {
	      if( average )
		out[i / binning] = sum / ct;
	      else
		out[i / binning] = sum;
	    }
	  sum = 0;
	  ct = 0;
	}
    }
}

