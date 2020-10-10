#    Copyright (C) 2020 Jeremy S. Sanders
#    Email: Jeremy Sanders <jeremy@jeremysanders.net>
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
##############################################################################

from __future__ import division
import numpy as N

from .. import qtall as qt
from .. import document
from .. import setting
from .. import utils
from .nonorthgraph import NonOrthGraph
from ..helpers import qtloops

try:
    from astropy import wcs
    from astropy.io.fits import Header
    havewcs = True
except ImportError:
    havewcs = False

def _(text, disambiguation=None, context='Project'):
    return qt.QCoreApplication.translate(context, text, disambiguation)

projections = {
    'AZP': 'zenithal/azimuthal perspective (AZP)',
    'SZP': 'slant zenithal perspective (SZP)',
    'TAN': 'gnomonic (TAN)',
    'STG': 'stereographic (STG)',
    'SIN': '(slant) orthographic (SIN)',
    'ARC': 'azimuthal equidistant (ARC)',
#    'ZPN': 'azimuthal polynomial (ZPN)', # doesn't work
    'ZEA': 'azimuthal equal area (ZEA)',
    'AIR': 'Airy (AIR)',
    'CYP': 'cylindrical perspective (CYP)',
    'CEA': 'cylindrical equal area (CEA)',
    'CAR': 'equirectangular (CAR)',
    'MER': 'Mercator (MER)',
    'COP': 'conic perspective (COP)', # doesn't work
    'COE': 'conic equal area (COE)', # doesn't work
    'COD': 'conic equidistant (COD)', # doesn't work
    'COO': 'conic orthomorphic (COO)', # doesn't work
    'SFL': 'sanson-flamsteed (SFL)',
    'PAR': 'parabolic (PAR)',
    'MOL': 'Mollweide (MOL)',
    'AIT': 'Hammer-Aitoff (AIT)',
    'BON': 'Bonne (BON)', # doesn't work
    'PCO': 'polyconic (PCO)',
    'TSC': 'tangential spherical cube (TSC)',
    'CSC': 'COBE quad spherical cube (CSC)',
    'QSC': 'quad spherical cube (QSC)',
    'HPX': 'HEALPix (HPX)',
    'XPH': 'HEALPix polar (XPH)',
}

params = {
    'AZP': [('Distance (µ)', 0), ('Tilt (γ; deg)',0)],
    'SZP': [('Distance (µ)', 0), ('Longitude (φ_c; deg)', 0), ('Latitude (θ_c; deg)', 90)],
    'TAN': [],
    'STG': [],
    'SIN': [('Slant η' ,0), ('Slant ξ', 0)],
    'ARC': [],
    'ZEA': [],
    'AIR': [('Latitude (θ_b; deg)', 90)],
    'CYP': [('Distance (µ)', 1), ('Cylinder radius (λ)', 1)],

}


class GridLine(setting.Line):
    '''Grid in plot.'''

    def __init__(self, name, **args):
        setting.Line.__init__(self, name, **args)
        steps = [1,2,5,10,15,30,45,60,90,180]
        self.add( setting.FloatChoice(
            'latitudeStep', steps, 30,
            descr = _('Latitude grid step'),
            usertext=_('Latitude step')), 0 )
        self.add( setting.FloatChoice(
            'longitudeStep', steps, 30,
            descr = _('Longitude grid step'),
            usertext=_('Longitude step')), 1 )
        self.get('color').newDefault('grey')


class Project(NonOrthGraph):
    '''Projection plotter'''

    typename = 'proj'
    allowusercreation = True
    description = _('Projection')

    @classmethod
    def addSettings(klass, s):
        '''Construct list of settings.'''
        NonOrthGraph.addSettings(s)

        inv_projections = {v:k for k,v in projections.items()}
        sortnames = tuple(sorted(inv_projections, key=lambda x: x.lower()))
        sortcodes = tuple((inv_projections[x] for x in sortnames))

        s.add( setting.Choice(
            'projection',
            sortcodes,
            'AIT',
            uilist=sortnames,
            descr=_('Projection'),
            usertext=_('Projection')) )
        s.add( setting.Float(
            'ref_lon',
            0,
            descr=_('Central longitude (deg)'),
            usertext=_('Longitude')) )
        s.add( setting.Float(
            'ref_lat',
            0,
            descr=_('Central latitude (deg)'),
            usertext=_('Latitude')) )
        s.add( setting.Float(
            'fov',
            90.,
            descr=_('Vertical field of view (deg)'),
            usertext=_('FoV')) )
        s.add( setting.Float(
            'projparam',
            0,
            descr=_('Projection parameter angle (deg)'),
            usertext=_('Parameter')) )

        s.add( GridLine(
            'SpokeLine',
            descr=_('Grid line'),
            usertext=_('Grid line')), pixmap='settings_axisgridlines')

    def __init__(self, parent, name=None):
        NonOrthGraph.__init__(self, parent, name=name)
        self._cache_key = None
        self._cache_wcs = None
        self._scale = 0
        self._ox = self._oy = 0
        self._valid_mask = None

    def _checkWCSCache(self):
        s = self.settings
        proj = s.projection
        refpos = (s.ref_lon, s.ref_lat)
        fov = s.fov
        param = s.projparam

        k = (proj, refpos, fov, param)
        if self._cache_key == k or not havewcs:
            return
        self._cache_key = k

        hdr = Header()
        projcode = proj
        hdr['CTYPE1'] = 'RA---%s' % projcode
        hdr['CTYPE2'] = 'DEC--%s' % projcode
        hdr['CRVAL1'] = refpos[0]
        hdr['CRVAL2'] = refpos[1]
        hdr['CRPIX1'] = 0
        hdr['CRPIX2'] = 0
        hdr['CDELT1'] = fov/100
        hdr['CDELT2'] = fov/100
        hdr['CUNIT1'] = 'deg'
        hdr['CUNIT2'] = 'deg'
        hdr['PV2_1'] = param
        hdr['EQUINOX'] = 2000.0

        try:
            self._cache_wcs = wcs.WCS(hdr)
        except Exception:
            self._cache_wcs = None

    def graphToPlotCoords(self, coorda, coordb):
        '''Convert coordinates to linear plot coordinates'''

        if not havewcs or self._cache_wcs is None:
            return N.nan*coorda, N.nan*coordb

        try:
            x, y = self._cache_wcs.wcs_world2pix(coorda, coordb, 0)
        except Exception:
            return N.nan*coorda, N.nan*coordb

        x = x*0.01*self._scale + self._ox
        y = y*-0.01*self._scale + self._oy
        return x, y

    def breakLines(self, posn, px, py):
        """Break into places where a coordinate jumps by more than
        half the graph width."""
        xw = posn[2]-posn[0]
        yw = posn[3]-posn[1]
        return utils.breakCoordsOnJump(px, py, 0.5*xw, 0.5*yw)

    def computeMask(self, bounds):
        if self._cache_wcs is None:
            self._valid_mask = None
            return

        yg, xg = N.meshgrid(
            N.arange(int(bounds[1]), int(bounds[3])+1),
            N.arange(int(bounds[0]), int(bounds[2])+1),
        )
        ys = (yg-self._oy)*(1/(-0.01*self._scale))
        xs = (xg-self._ox)*(1/( 0.01*self._scale))
        lon, lat = self._cache_wcs.wcs_pix2world(xs.ravel(), ys.ravel(), 0)

        valid = N.reshape(N.isfinite(lat+lon), yg.shape)
        self._valid_mask = valid

    def drawGrid(self, painter, bounds):
        if self._cache_wcs is None:
            return

        maskpolys = qtloops.traceBitmap(self._valid_mask.T.astype(N.intc))
        for poly in maskpolys:
            polyf = qt.QPolygonF(poly)
            polyf.translate(int(bounds[0]), int(bounds[1]))
            painter.drawPolygon(polyf)

        clip = qt.QRectF(
            bounds[0], bounds[1], bounds[2]-bounds[0], bounds[3]-bounds[1])
        painter.save()
        painter.setClipRect(clip)
        xj = 0.25*(bounds[2]-bounds[0])
        yj = 0.25*(bounds[3]-bounds[1])

        def drawlines(x, y):
            for xr, yr in zip(x, y):
                for xn, yn in utils.breakCoordsOnNans(xr, yr):
                    for xs, ys in utils.breakCoordsOnJump(xn, yn, xj, yj):
                        poly = qt.QPolygonF()
                        qtloops.addNumpyToPolygonF(poly, xs, ys)
                        qtloops.plotClippedPolyline(painter, clip, poly)

        # draw lines of latitude
        lons = N.arange(-180,180+1,5, dtype=N.float64)
        lons[0] = -179.99
        lons[-1] = 179.99
        lats = N.array([-89.999, -60, -30, 0, 30, 60, 89.999])
        mlon, mlat = N.meshgrid(lons, lats)
        x, y = self.graphToPlotCoords(mlon.ravel(), mlat.ravel())
        x = x.reshape(mlon.shape)
        y = y.reshape(mlat.shape)
        drawlines(x, y)

        # draw lines of longitude
        lons = N.arange(-180,180+1,30, dtype=N.float64)
        lons[0] = -179.999
        lons[-1] = 179.999
        lats = N.linspace(-90, 90, 90+1, dtype=N.float64)
        lats[0] = -89.999
        lats[-1] = 89.999
        mlon, mlat = N.meshgrid(lons, lats)
        x, y = self.graphToPlotCoords(mlon.ravel(), mlat.ravel())
        x = x.reshape(mlon.shape)
        y = y.reshape(mlat.shape)
        drawlines(x.T, y.T)

        painter.restore()

    def drawGraph(self, painter, bounds, datarange, outerbounds=None):
        '''Plot graph area and axes.'''

        self._scale = (bounds[3]-bounds[1])*0.5
        self._ox = 0.5*(bounds[0]+bounds[2])
        self._oy = 0.5*(bounds[1]+bounds[3])

        self._checkWCSCache()
        self.computeMask(bounds)

        self.drawGrid(painter, bounds)


document.thefactory.register(Project)
