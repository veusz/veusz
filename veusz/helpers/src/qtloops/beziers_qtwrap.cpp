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

