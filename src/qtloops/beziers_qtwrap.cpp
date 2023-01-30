//    Copyright (C) 2010 Jeremy S. Sanders
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

#include "beziers.h"
#include "beziers_qtwrap.h"

QPolygonF bezier_fit_cubic_single( const QPolygonF& data, double error )
{
  QPolygonF out(4);
  const int retn = sp_bezier_fit_cubic(out.data(), data.data(),
				       data.count(), error);
  if( retn >= 0 )
    return out;
  else
    return QPolygonF();
}

QPolygonF bezier_fit_cubic_multi( const QPolygonF& data, double error,
				  unsigned max_beziers )
{
  QPolygonF out(4*max_beziers);
  const int retn = sp_bezier_fit_cubic_r(out.data(), data.data(),
					 data.count(), error,
					 max_beziers);

  if( retn >= 0 )
    {
      // get rid of unused points
      if( retn*4 < out.count() )
	out.remove( retn*4, out.count()-retn*4 );
      return out;
    }
  else
    return QPolygonF();
}

QPolygonF bezier_fit_cubic_tight(const QPolygonF& data, double looseness)
{
 /**
  * MS Excel-like cubic Bezier fitting formulated by Brian Murphy.
  * (http://www.xlrotor.com/Smooth_curve_bezier_example_file.zip)
  * 
  * 4 bezier control points (ctrls[0]-ctrl[3]) are created for each line
  * segment. Positions of ctrls are determined by 4 nearest data points
  * (pt0-pt3) with following rules:
  * ctrls[0]: same position as pt0.
  * ctrls[1]: on a line through pt1 parallel to pt0-pt2,
  *           at a distance from pt1 = f1 * |pt0-pt2|
  * ctrls[2]: on a line through pt2 parallel to pt1-pt3,
  *           at a distance from pt2 = f2 * |pt1-pt3|
  * ctrls[3]: same position as pt3
  * The magic numbers (f1 and f2) are determined by length ratio of the
  * 3 line segments and "looseness" with some additional rules.
  * looseness: artificial parameter to control "tension" of the Bezier
  *            curve. Larger value gives more curved connection.
  *            In MS Excell, this value is set as "0.5".
  */
  const int len = data.size();
  if (len < 2) {
    return QPolygonF();
  }else if (len == 2) {
    QPolygonF bezier_ctrls(4);
    bezier_ctrls << data[0] << data[0] << data[1] << data[1];
    return bezier_ctrls;
  }else{
    QPolygonF bezier_ctrls(4 * (len - 1));
    for (int i = 1; i < len; i++) {
      QPolygonF ctrls(4);
      QPointF pt0;
      QPointF pt1 = data[i-1];
      QPointF pt2 = data[i];
      QPointF pt3;
      ctrls[0] = pt1;
      ctrls[3] = pt2;
      double f1;
      double f2;
      if (i == 1) {
        pt0 = data[i-1];
        pt3 = data[i+1];
        f1 = looseness / 1.5;
        f2 = looseness / 3.0;
      }else if  (i == len - 1) {
        pt0 = data[i-2];
        pt3 = data[i];
        f1 = looseness / 3.0;
        f2 = looseness / 1.5;
      }else{
        pt0 = data[i-2];
        pt3 = data[i+1];
        f1 = looseness / 3.0;
        f2 = looseness / 3.0;
      }
      double d02 = QLineF(pt0, pt2).length();
      double d12 = QLineF(pt1, pt2).length();
      double d13 = QLineF(pt1, pt3).length();
      bool b1 = d02 < d12 * 3.0;
      bool b2 = d13 < d12 * 3.0;
      if (!(b1 && b2)) {
        f1 = d12 / d02 / 2.0;
        f2 = d12 / d13 / 2.0;
        if (b1) {
          f1 = f2;
        }
        if (b2) {
          f2 = f1;
        }
      }
      ctrls[1] = pt1 + (pt2 - pt0) * f1;
      ctrls[2] = pt2 + (pt1 - pt3) * f2;
      bezier_ctrls += ctrls;
    }
    return bezier_ctrls;
  }
}
