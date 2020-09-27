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
from .nonorthgraph import NonOrthGraph

try:
    from astropy import wcs
    from astropy.io.fits import Header
    havewcs = True
except ImportError:
    havewcs = False

def _(text, disambiguation=None, context='Project'):
    return qt.QCoreApplication.translate(context, text, disambiguation)

projections = {
    'AZP': 'azimuthal perspective (AZP)',
    'SZP': 'slant zenithal perspective (SZP)',
    'TAN': 'gnomonic (TAN)',
    'STG': 'stereographic (STG)',
    'SIN': 'orthographic (SIN)',
    'ARC': 'azimuthal equidistant (ARC)',
    'ZPN': 'azimuthal polynomial (ZPN)',
    'ZEA': 'azimuthal equal area (ZEA)',
    'AIR': 'Airy (AIR)',
    'CYP': 'cylindrical perspective (CYP)',
    'CEA': 'cylindrical equal area (CEA)',
    'CAR': 'equirectangular (CAR)',
    'MER': 'Mercator (MER)',
    'COP': 'conic perspective (COP)',
    'COE': 'conic equal area (COE)',
    'COD': 'conic equidistant (COD)',
    'COO': 'conic orthomorphic (COO)',
    'SFL': 'sanson-flamsteed (SFL)',
    'PAR': 'parabolic (PAR)',
    'MOL': 'Mollweide (MOL)',
    'AIT': 'Hammer-Aitoff (AIT)',
    'BON': 'Bonne (BON)',
    'PCO': 'polyconic (PCO)',
    'TSC': 'tangential spherical cube (TSC)',
    'CSC': 'COBE quad spherical cube (CSC)',
    'QSC': 'quad spherical cube (QSC)',
    'HPX': 'HEALPix (HPX)',
    'XPH': 'HEALPix polar (XPH)',
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

    def _checkWCSCache(self):
        s = self.settings
        proj = s.projection
        refpos = (s.ref_lon, s.ref_lat)
        fov = s.fov

        k = (proj, refpos, fov)
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
        breakpos = N.where(
            (N.abs(px[1:]-px[:-1]) > 0.5*xw) |
            (N.abs(py[1:]-py[:-1]) > 0.5*yw))[0]
        if len(breakpos) == 0:
            # no breaks
            out = ((px, py),)
        else:
            # split
            out = []
            last = 0
            for brk in breakpos:
                out.append((px[last:brk+1], py[last:brk+1]))
                last = brk+1
            out.append((px[last:], py[last:]))
        return out

    def drawGrid(self, painter, bounds):

    def drawGraph(self, painter, bounds, datarange, outerbounds=None):
        '''Plot graph area and axes.'''

        self._scale = (bounds[3]-bounds[1])*0.5
        self._ox = 0.5*(bounds[0]+bounds[2])
        self._oy = 0.5*(bounds[1]+bounds[3])

        self._checkWCSCache()




document.thefactory.register(Project)
