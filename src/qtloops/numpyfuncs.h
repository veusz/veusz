//    Copyright (C) 2011 Jeremy S. Sanders
//    Email: Jeremy Sanders <jeremy@jeremysanders.net>
//
//    This file is part of Veusz.
//
//    Veusz is free software: you can redistribute it and/or modify it
//    under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 2 of the License, or
//    (at your option) any later version.
//
//    Veusz is distributed in the hope that it will be useful, but
//    WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
//    General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with Veusz. If not, see <https://www.gnu.org/licenses/>.
//
//////////////////////////////////////////////////////////////////////////////

#ifndef NUMPYFUNCS_HH
#define NUMPYFUNCS_HH

#include "qtloops_helpers.h"

// bin up data given by factor. If average is True, then divide by number
// of elements in bins
void binData(const Numpy1DObj& indata, int binning,
	     bool average,
	     int* numoutbins, double** outdata);


// rolling average calculation
// weights is an optional weighting array
void rollingAverage(const Numpy1DObj& indata,
		    const Numpy1DObj* weights,
		    int width,
		    int* numoutbins, double** outdata);

#endif
