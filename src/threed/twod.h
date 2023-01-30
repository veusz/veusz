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

#ifndef TWOD_H
#define TWOD_H

#include "mmaths.h"

enum ISect { LINE_NOOVERLAP, LINE_CROSS, LINE_OVERLAP }; 

// Do the two line segments p1->p2, q1->q2 cross or overlap?
// return LINE_NOOVERLAP if no overlap
//        LINE_CROSS if they cross somewhere
//        LINE_OVERLAP if they lie on top of each other partially
// if posn != 0, return crossing position if LINE_CROSS
// if LINE_OVERLAP the two end points of overlap are returned in posn and posn2
// Assumes that the line segments are finite.

ISect twodLineIntersect(Vec2 p1, Vec2 p2, Vec2 q1, Vec2 q2,
                        Vec2* posn=0, Vec2* posn2=0);

// clip 2D polygon by a 2nd polygon (must be clockwise polygons)
Vec2Vector twodPolyEdgeClip(Vec2Vector inPoly, const Vec2Vector& clipPoly);

// area of polygon (+ve -> clockwise)
double twodPolyArea(const Vec2Vector& poly);

// ensure polygon is clockwise
void twodPolyMakeClockwise(Vec2Vector* poly);

// does line cross polygon? (make sure poly is defined clockwise)
bool twodLineIntersectPolygon(Vec2 p1, Vec2 p2, const Vec2Vector& poly);

#endif
