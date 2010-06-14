// -*- mode: C++; -*-

#ifndef SP_BEZIERS_H
#define SP_BEZIERS_H

/*
 * An Algorithm for Automatically Fitting Digitized Curves
 * by Philip J. Schneider
 * from "Graphics Gems", Academic Press, 1990
 *
 * Authors:
 *   Philip J. Schneider
 *   Lauris Kaplinski <lauris@ximian.com>
 *
 * Copyright (C) 1990 Philip J. Schneider
 * Copyright (C) 2001 Lauris Kaplinski and Ximian, Inc.
 *
 * Released under GNU GPL
 */

/* Bezier approximation utils */

// Modified to be based around QPointF by Jeremy Sanders (2007)

#include <QPointF>

QPointF bezier_pt(unsigned const degree, QPointF const V[], double const t);

int sp_bezier_fit_cubic(QPointF bezier[], QPointF const *data,
			int len, double error);

int sp_bezier_fit_cubic_r(QPointF bezier[], QPointF const data[],
			  int len, double error,
			  unsigned max_beziers);

int sp_bezier_fit_cubic_full(QPointF bezier[], int split_points[],
			     QPointF const data[], int len,
			     QPointF const &tHat1, QPointF const &tHat2,
			     double error, unsigned max_beziers);

QPointF sp_darray_left_tangent(QPointF const d[], unsigned const len);
QPointF sp_darray_left_tangent(QPointF const d[], unsigned const len,
			       double const tolerance_sq);
QPointF sp_darray_right_tangent(QPointF const d[], unsigned const length,
				double const tolerance_sq);


#endif /* SP_BEZIERS_H */
