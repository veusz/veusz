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

namespace {
  template <class T> inline T min(T a, T b)
  {
    return (a<b) ? a : b;
  }
}

void binData(const Numpy1DObj& indata, int binning,
	     bool average,
	     int* numoutbins, double** outdata)
{
  // round up output size
  int size = indata.dim / binning;
  if( indata.dim % binning != 0 )
    ++size;

  // create output array
  *numoutbins = size;
  double *out = new double[size];
  *outdata = out;

  // do binning
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

void rollingAverage(const Numpy1DObj& indata,
		    const Numpy1DObj* weights,
		    int width,
		    int* numoutbins, double** outdata)
{
  // round up output size
  int size = indata.dim;
  if( weights != 0 )
    size = min( weights->dim, size );

  // create output array
  *numoutbins = size;
  double *out = new double[size];
  *outdata = out;

  for(int i = 0 ; i < size; ++i)
    {
      double ct = 0.;
      double sum = 0.;

      // iterate over rolling width
      for(int di = -width; di <= width; ++di)
	{
	  const int ri = di+i;
	  if ( ri >= 0 && ri < size && isFinite(indata(ri)) )
	    {
	      if( weights != 0 )
		{
		  // weighted average
		  const double w = (*weights)(ri);
		  if( isFinite(w) )
		    {
		      ct += w;
		      sum += w*indata(ri);
		    }
		}
	      else
		{
		  // standard average
		  ct += 1;
		  sum += indata(ri);
		}
	    }
	}

      if( ct != 0. )
	out[i] = sum / ct;
      else
	out[i] = std::numeric_limits<double>::quiet_NaN();
    }
}
