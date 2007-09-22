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

