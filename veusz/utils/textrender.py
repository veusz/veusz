# textrender.py
# module to render text, tries to understand a basic LateX-like syntax

#    Copyright (C) 2003 Jeremy S. Sanders
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
###############################################################################

import math
import re

import numpy as N

from .. import qtall as qt
from . import points

#from ..helpers import qtmml
from ..helpers import recordpaint
from ..helpers.qtloops import RotatedRectangle

def _(text, disambiguation=None, context='TextRender'):
    """Translate text."""
    return qt.QCoreApplication.translate(context, text, disambiguation)

# this definition is monkey-patched when veusz is running in self-test
# mode as we need to hack the metrics - urgh
FontMetrics = qt.QFontMetricsF

# lookup table for special symbols
symbols = {
    # escaped characters
    r'\_': '_',
    r'\^': '^',
    r'\{': '{',
    r'\}': '}',
    r'\[': '[',
    r'\]': ']',
    r'\backslash' : '\u005c',

    # operators
    r'\pm': '\u00b1',
    r'\mp': '\u2213',
    r'\times': '\u00d7',
    r'\cdot': '\u22c5',
    r'\ast': '\u2217',
    r'\star': '\u22c6',
    r'\deg': '\u00b0',
    r'\divide': '\u00f7',
    r'\dagger': '\u2020',
    r'\ddagger': '\u2021',
    r'\cup': '\u22c3',
    r'\cap': '\u22c2',
    r'\uplus': '\u228e',
    r'\vee': '\u22c1',
    r'\wedge': '\u22c0',
    r'\nabla': '\u2207',
    r'\lhd': '\u22b2',
    r'\rhd': '\u22b3',
    r'\unlhd': '\u22b4',
    r'\unrhd': '\u22b5',

    r'\oslash': '\u2298',
    r'\odot': '\u2299',
    r'\oplus': '\u2295',
    r'\ominus': '\u2296',
    r'\otimes': '\u2297',

    r'\diamond': '\u22c4',
    r'\bullet': '\u2022',
    r'\AA': '\u212b',
    r'\sqrt': '\u221a',
    r'\propto': '\u221d',
    r'\infty': '\u221e',
    r'\int': '\u222b',
    r'\leftarrow': '\u2190',
    r'\Leftarrow': '\u21d0',
    r'\uparrow': '\u2191',
    r'\rightarrow': '\u2192',
    r'\to': '\u2192',
    r'\Rightarrow': '\u21d2',
    r'\downarrow': '\u2193',
    r'\leftrightarrow': '\u2194',
    r'\Leftrightarrow': '\u21d4',
    r'\circ': '\u25cb',
    r'\ell': '\u2113',

    # relations
    r'\le': '\u2264',
    r'\ge': '\u2265',
    r'\neq': '\u2260',
    r'\sim': '\u223c',
    r'\ll': '\u226a',
    r'\gg': '\u226b',
    r'\doteq': '\u2250',
    r'\simeq': '\u2243',
    r'\subset': '\u2282',
    r'\supset': '\u2283',
    r'\approx': '\u2248',
    r'\asymp': '\u224d',
    r'\subseteq': '\u2286',
    r'\supseteq': '\u2287',
    r'\sqsubset': '\u228f',
    r'\sqsupset': '\u2290',
    r'\sqsubseteq': '\u2291',
    r'\sqsupseteq': '\u2292',
    r'\in': '\u2208',
    r'\ni': '\u220b',
    r'\equiv': '\u2261',
    r'\prec': '\u227a',
    r'\succ': '\u227b',
    r'\preceq': '\u227c',
    r'\succeq': '\u227d',
    r'\bowtie': '\u22c8',
    r'\vdash': '\u22a2',
    r'\dashv': '\u22a3',
    r'\models': '\u22a7',
    r'\perp': '\u22a5',
    r'\parallel': '\u2225',
    r'\umid': '\u2223',

    # lower case greek letters
    r'\alpha': '\u03b1',
    r'\beta': '\u03b2',
    r'\gamma': '\u03b3',
    r'\delta': '\u03b4',
    r'\epsilon': '\u03b5',
    r'\zeta': '\u03b6',
    r'\eta': '\u03b7',
    r'\theta': '\u03b8',
    r'\iota': '\u03b9',
    r'\kappa': '\u03ba',
    r'\lambda': '\u03bb',
    r'\mu': '\u03bc',
    r'\nu': '\u03bd',
    r'\xi': '\u03be',
    r'\omicron': '\u03bf',
    r'\pi': '\u03c0',
    r'\rho': '\u03c1',
    r'\stigma': '\u03c2',
    r'\sigma': '\u03c3',
    r'\tau': '\u03c4',
    r'\upsilon': '\u03c5',
    r'\phi': '\u03c6',
    r'\chi': '\u03c7',
    r'\psi': '\u03c8',
    r'\omega': '\u03c9',

    # upper case greek letters
    r'\Alpha': '\u0391',
    r'\Beta': '\u0392',
    r'\Gamma': '\u0393',
    r'\Delta': '\u0394',
    r'\Epsilon': '\u0395',
    r'\Zeta': '\u0396',
    r'\Eta': '\u0397',
    r'\Theta': '\u0398',
    r'\Iota': '\u0399',
    r'\Kappa': '\u039a',
    r'\Lambda': '\u039b',
    r'\Mu': '\u039c',
    r'\Nu': '\u039d',
    r'\Xi': '\u039e',
    r'\Omicron': '\u039f',
    r'\Pi': '\u03a0',
    r'\Rho': '\u03a1',
    r'\Sigma': '\u03a3',
    r'\Tau': '\u03a4',
    r'\Upsilon': '\u03a5',
    r'\Phi': '\u03a6',
    r'\Chi': '\u03a7',
    r'\Psi': '\u03a8',
    r'\Omega': '\u03a9',

    # hebrew
    r'\aleph': '\u05d0',
    r'\beth': '\u05d1',
    r'\daleth': '\u05d3',
    r'\gimel': '\u2137',

    # more symbols
    '\\AE'              : '\xc6',
    '\\Angle'           : '\u299c',
    '\\Bumpeq'          : '\u224e',
    '\\Cap'             : '\u22d2',
    '\\Colon'           : '\u2237',
    '\\Cup'             : '\u22d3',
    '\\DH'              : '\xd0',
    '\\DJ'              : '\u0110',
    '\\Digamma'         : '\u03dc',
    '\\Koppa'           : '\u03de',
    '\\L'               : '\u0141',
    '\\LeftDownTeeVector': '\u2961',
    '\\LeftDownVectorBar': '\u2959',
    '\\LeftRightVector' : '\u294e',
    '\\LeftTeeVector'   : '\u295a',
    '\\LeftTriangleBar' : '\u29cf',
    '\\LeftUpDownVector': '\u2951',
    '\\LeftUpTeeVector' : '\u2960',
    '\\LeftUpVectorBar' : '\u2958',
    '\\LeftVectorBar'   : '\u2952',
    '\\Lleftarrow'      : '\u21da',
    '\\Longleftarrow'   : '\u27f8',
    '\\Longleftrightarrow': '\u27fa',
    '\\Longrightarrow'  : '\u27f9',
    '\\Lsh'             : '\u21b0',
    '\\NG'              : '\u014a',
    '\\NestedGreaterGreater': '\u2aa2',
    '\\NestedLessLess'  : '\u2aa1',
    '\\O'               : '\xd8',
    '\\OE'              : '\u0152',
    '\\ReverseUpEquilibrium': '\u296f',
    '\\RightDownTeeVector': '\u295d',
    '\\RightDownVectorBar': '\u2955',
    '\\RightTeeVector'  : '\u295b',
    '\\RightTriangleBar': '\u29d0',
    '\\RightUpDownVector': '\u294f',
    '\\RightUpTeeVector': '\u295c',
    '\\RightUpVectorBar': '\u2954',
    '\\RightVectorBar'  : '\u2953',
    '\\RoundImplies'    : '\u2970',
    '\\Rrightarrow'     : '\u21db',
    '\\Rsh'             : '\u21b1',
    '\\RuleDelayed'     : '\u29f4',
    '\\Sampi'           : '\u03e0',
    '\\Stigma'          : '\u03da',
    '\\Subset'          : '\u22d0',
    '\\Supset'          : '\u22d1',
    '\\TH'              : '\xde',
    '\\UpArrowBar'      : '\u2912',
    '\\UpEquilibrium'   : '\u296e',
    '\\Uparrow'         : '\u21d1',
    '\\Updownarrow'     : '\u21d5',
    '\\VDash'           : '\u22ab',
    '\\Vdash'           : '\u22a9',
    '\\Vert'            : '\u2016',
    '\\Vvdash'          : '\u22aa',
    '\\aa'              : '\xe5',
    '\\ae'              : '\xe6',
    '\\allequal'        : '\u224c',
    '\\amalg'           : '\u2a3f',
    '\\angle'           : '\u2220',
    '\\approxeq'        : '\u224a',
    '\\approxnotequal'  : '\u2246',
    '\\aquarius'        : '\u2652',
    '\\aries'           : '\u2648',
    '\\arrowwaveright'  : '\u219d',
    '\\backepsilon'     : '\u03f6',
    '\\backprime'       : '\u2035',
    '\\backsim'         : '\u223d',
    '\\backsimeq'       : '\u22cd',
    '\\barwedge'        : '\u2305',
    '\\because'         : '\u2235',
    '\\between'         : '\u226c',
    '\\bigcap'          : '\u22c2',
    '\\bigcirc'         : '\u25ef',
    '\\bigcup'          : '\u22c3',
    '\\bigtriangledown' : '\u25bd',
    '\\bigtriangleup'   : '\u25b3',
    '\\blacklozenge'    : '\u29eb',
    '\\blacksquare'     : '\u25aa',
    '\\blacktriangle'   : '\u25b4',
    '\\blacktriangledown': '\u25be',
    '\\blacktriangleleft': '\u25c2',
    '\\blacktriangleright': '\u25b8',
    '\\boxdot'          : '\u22a1',
    '\\boxminus'        : '\u229f',
    '\\boxplus'         : '\u229e',
    '\\boxtimes'        : '\u22a0',
    '\\bumpeq'          : '\u224f',
    '\\cancer'          : '\u264b',
    '\\capricornus'     : '\u2651',
    '\\cdots'           : '\u22ef',
    '\\circeq'          : '\u2257',
    '\\circlearrowleft' : '\u21ba',
    '\\circlearrowright': '\u21bb',
    '\\circledS'        : '\u24c8',
    '\\circledast'      : '\u229b',
    '\\circledcirc'     : '\u229a',
    '\\circleddash'     : '\u229d',
    '\\clockoint'       : '\u2a0f',
    '\\clwintegral'     : '\u2231',
    '\\complement'      : '\u2201',
    '\\cong'            : '\u2245',
    '\\coprod'          : '\u2210',
    '\\curlyeqprec'     : '\u22de',
    '\\curlyeqsucc'     : '\u22df',
    '\\curlyvee'        : '\u22ce',
    '\\curlywedge'      : '\u22cf',
    '\\curvearrowleft'  : '\u21b6',
    '\\curvearrowright' : '\u21b7',
    '\\dblarrowupdown'  : '\u21c5',
    '\\ddddot'          : '\u20dc',
    '\\dddot'           : '\u20db',
    '\\dh'              : '\xf0',
    '\\diagup'          : '\u2571',
    '\\digamma'         : '\u03dd',
    '\\div'             : '\xf7',
    '\\divideontimes'   : '\u22c7',
    '\\dj'              : '\u0111',
    '\\doteqdot'        : '\u2251',
    '\\dotplus'         : '\u2214',
    '\\downdownarrows'  : '\u21ca',
    '\\downharpoonleft' : '\u21c3',
    '\\downharpoonright': '\u21c2',
    '\\downslopeellipsis': '\u22f1',
    '\\eighthnote'      : '\u266a',
    '\\eqcirc'          : '\u2256',
    '\\eqslantgtr'      : '\u2a96',
    '\\eqslantless'     : '\u2a95',
    '\\estimates'       : '\u2259',
    '\\eth'             : '\u01aa',
    '\\exists'          : '\u2203',
    '\\fallingdotseq'   : '\u2252',
    '\\flat'            : '\u266d',
    '\\forall'          : '\u2200',
    '\\forcesextra'     : '\u22a8',
    '\\frown'           : '\u2322',
    '\\gemini'          : '\u264a',
    '\\geq'             : '\u2265',
    '\\geqq'            : '\u2267',
    '\\geqslant'        : '\u2a7e',
    '\\gnapprox'        : '\u2a8a',
    '\\gneq'            : '\u2a88',
    '\\gneqq'           : '\u2269',
    '\\gnsim'           : '\u22e7',
    '\\greaterequivlnt': '\u2273',
    '\\gtrapprox'       : '\u2a86',
    '\\gtrdot'          : '\u22d7',
    '\\gtreqless'       : '\u22db',
    '\\gtreqqless'      : '\u2a8c',
    '\\gtrless'         : '\u2277',
    '\\guillemotleft'   : '\xab',
    '\\guillemotright'  : '\xbb',
    '\\guilsinglleft'   : '\u2039',
    '\\guilsinglright'  : '\u203a',
    '\\hermitconjmatrix': '\u22b9',
    '\\homothetic'      : '\u223b',
    '\\hookleftarrow'   : '\u21a9',
    '\\hookrightarrow'  : '\u21aa',
    '\\hslash'          : '\u210f',
    '\\i'               : '\u0131',
    '\\intercal'        : '\u22ba',
    '\\jupiter'         : '\u2643',
    '\\k'               : '\u0328',
    '\\l'               : '\u0142',
    '\\langle'          : '\u2329',
    '\\lazysinv'        : '\u223e',
    '\\lceil'           : '\u2308',
    '\\ldots'           : '\u2026',
    '\\leftarrowtail'   : '\u21a2',
    '\\leftharpoondown' : '\u21bd',
    '\\leftharpoonup'   : '\u21bc',
    '\\leftleftarrows'  : '\u21c7',
    '\\leftrightarrows' : '\u21c6',
    '\\leftrightharpoons': '\u21cb',
    '\\leftrightsquigarrow': '\u21ad',
    '\\leftthreetimes'  : '\u22cb',
    '\\leo'             : '\u264c',
    '\\leq'             : '\u2264',
    '\\leqq'            : '\u2266',
    '\\leqslant'        : '\u2a7d',
    '\\lessapprox'      : '\u2a85',
    '\\lessdot'         : '\u22d6',
    '\\lesseqgtr'       : '\u22da',
    '\\lesseqqgtr'      : '\u2a8b',
    '\\lessequivlnt'    : '\u2272',
    '\\lessgtr'         : '\u2276',
    '\\lfloor'          : '\u230a',
    '\\libra'           : '\u264e',
    '\\llcorner'        : '\u231e',
    '\\lmoustache'      : '\u23b0',
    '\\lnapprox'        : '\u2a89',
    '\\lneq'            : '\u2a87',
    '\\lneqq'           : '\u2268',
    '\\lnot'            : '\xac',
    '\\lnsim'           : '\u22e6',
    '\\longleftarrow'   : '\u27f5',
    '\\longleftrightarrow': '\u27f7',
    '\\longmapsto'      : '\u27fc',
    '\\longrightarrow'  : '\u27f6',
    '\\looparrowleft'   : '\u21ab',
    '\\looparrowright'  : '\u21ac',
    '\\lozenge'         : '\u25ca',
    '\\lrcorner'        : '\u231f',
    '\\ltimes'          : '\u22c9',
    '\\male'            : '\u2642',
    '\\mapsto'          : '\u21a6',
    '\\measuredangle'   : '\u2221',
    '\\mercury'         : '\u263f',
    '\\mho'             : '\u2127',
    '\\mid'             : '\u2223',
    '\\mkern1mu'        : '\u200a',
    '\\mkern4mu'        : '\u205f',
    '\\multimap'        : '\u22b8',
    '\\nLeftarrow'      : '\u21cd',
    '\\nLeftrightarrow' : '\u21ce',
    '\\nRightarrow'     : '\u21cf',
    '\\nVDash'          : '\u22af',
    '\\nVdash'          : '\u22ae',
    '\\natural'         : '\u266e',
    '\\nearrow'         : '\u2197',
    '\\neptune'         : '\u2646',
    '\\nexists'         : '\u2204',
    '\\ng'              : '\u014b',
    '\\nleftarrow'      : '\u219a',
    '\\nleftrightarrow' : '\u21ae',
    '\\nmid'            : '\u2224',
    '\\nolinebreak'     : '\u2060',
    '\\notgreaterless'  : '\u2279',
    '\\notlessgreater'  : '\u2278',
    '\\nparallel'       : '\u2226',
    '\\nrightarrow'     : '\u219b',
    '\\ntriangleleft'   : '\u22ea',
    '\\ntrianglelefteq' : '\u22ec',
    '\\ntriangleright'  : '\u22eb',
    '\\ntrianglerighteq': '\u22ed',
    '\\nvDash'          : '\u22ad',
    '\\nvdash'          : '\u22ac',
    '\\nwarrow'         : '\u2196',
    '\\o'               : '\xf8',
    '\\oe'              : '\u0153',
    '\\oint'            : '\u222e',
    '\\openbracketleft' : '\u301a',
    '\\openbracketright': '\u301b',
    '\\original'        : '\u22b6',
    '\\partial'         : '\u2202',
    '\\perspcorrespond' : '\u2a5e',
    '\\pisces'          : '\u2653',
    '\\pitchfork'       : '\u22d4',
    '\\pluto'           : '\u2647',
    '\\precapprox'      : '\u2ab7',
    '\\preccurlyeq'     : '\u227c',
    '\\precedesnotsimilar': '\u22e8',
    '\\precnapprox'     : '\u2ab9',
    '\\precneqq'        : '\u2ab5',
    '\\prod'            : '\u220f',
    '\\quarternote'     : '\u2669',
    '\\rangle'          : '\u232a',
    '\\rbrace'          : '}',
    '\\rceil'           : '\u2309',
    '\\recorder'        : '\u2315',
    '\\rfloor'          : '\u230b',
    '\\rightangle'      : '\u221f',
    '\\rightanglearc'   : '\u22be',
    '\\rightarrowtail'  : '\u21a3',
    '\\rightharpoondown': '\u21c1',
    '\\rightharpoonup'  : '\u21c0',
    '\\rightleftarrows' : '\u21c4',
    '\\rightleftharpoons': '\u21cc',
    '\\rightmoon'       : '\u263e',
    '\\rightrightarrows': '\u21c9',
    '\\rightsquigarrow' : '\u21dd',
    '\\rightthreetimes' : '\u22cc',
    '\\risingdotseq'    : '\u2253',
    '\\rmoustache'      : '\u23b1',
    '\\rtimes'          : '\u22ca',
    '\\sagittarius'     : '\u2650',
    '\\saturn'          : '\u2644',
    '\\scorpio'         : '\u264f',
    '\\searrow'         : '\u2198',
    '\\setminus'        : '\u2216',
    '\\sharp'           : '\u266f',
    '\\smile'           : '\u2323',
    '\\sphericalangle'  : '\u2222',
    '\\sqcap'           : '\u2293',
    '\\sqcup'           : '\u2294',
    '\\sqrint'          : '\u2a16',
    '\\square'          : '\u25a1',
    '\\ss'              : '\xdf',
    '\\starequal'       : '\u225b',
    '\\subseteqq'       : '\u2ac5',
    '\\subsetneq'       : '\u228a',
    '\\subsetneqq'      : '\u2acb',
    '\\succapprox'      : '\u2ab8',
    '\\succcurlyeq'     : '\u227d',
    '\\succnapprox'     : '\u2aba',
    '\\succneqq'        : '\u2ab6',
    '\\succnsim'        : '\u22e9',
    '\\sum'             : '\u2211',
    '\\supseteqq'       : '\u2ac6',
    '\\supsetneq'       : '\u228b',
    '\\supsetneqq'      : '\u2acc',
    '\\surd'            : '\u221a',
    '\\surfintegral'    : '\u222f',
    '\\swarrow'         : '\u2199',
    '\\taurus'          : '\u2649',
    '\\textTheta'       : '\u03f4',
    '\\textasciiacute'  : '\xb4',
    '\\textasciibreve'  : '\u02d8',
    '\\textasciicaron'  : '\u02c7',
    '\\textasciidieresis': '\xa8',
    '\\textasciigrave'  : '`',
    '\\textasciimacron' : '\xaf',
    '\\textasciitilde'  : '~',
    '\\textbrokenbar'   : '\xa6',
    '\\textbullet'      : '\u2022',
    '\\textcent'        : '\xa2',
    '\\textcopyright'   : '\xa9',
    '\\textcurrency'    : '\xa4',
    '\\textdagger'      : '\u2020',
    '\\textdaggerdbl'   : '\u2021',
    '\\textdegree'      : '\xb0',
    '\\textdollar'      : '$',
    '\\textdoublepipe'  : '\u01c2',
    '\\textemdash'      : '\u2014',
    '\\textendash'      : '\u2013',
    '\\textexclamdown'  : '\xa1',
    '\\texthvlig'       : '\u0195',
    '\\textnrleg'       : '\u019e',
    '\\textonehalf'     : '\xbd',
    '\\textonequarter'  : '\xbc',
    '\\textordfeminine' : '\xaa',
    '\\textordmasculine': '\xba',
    '\\textparagraph'   : '\xb6',
    '\\textperiodcentered': '\u02d9',
    '\\textpertenthousand': '\u2031',
    '\\textperthousand' : '\u2030',
    '\\textphi'         : '\u0278',
    '\\textquestiondown': '\xbf',
    '\\textquotedblleft': '\u201c',
    '\\textquotedblright': '\u201d',
    '\\textquotesingle' : "'",
    '\\textregistered'  : '\xae',
    '\\textsection'     : '\xa7',
    '\\textsterling'    : '\xa3',
    '\\texttheta'       : '\u03b8',
    '\\textthreequarters': '\xbe',
    '\\texttildelow'    : '\u02dc',
    '\\texttimes'       : '\xd7',
    '\\texttrademark'   : '\u2122',
    '\\textturnk'       : '\u029e',
    '\\textvartheta'    : '\u03d1',
    '\\textvisiblespace': '\u2423',
    '\\textyen'         : '\xa5',
    '\\th'              : '\xfe',
    '\\therefore'       : '\u2234',
    '\\tildetrpl'       : '\u224b',
    '\\top'             : '\u22a4',
    '\\triangledown'    : '\u25bf',
    '\\triangleleft'    : '\u25c3',
    '\\trianglelefteq'  : '\u22b4',
    '\\triangleq'       : '\u225c',
    '\\triangleright'   : '\u25b9',
    '\\trianglerighteq' : '\u22b5',
    '\\truestate'       : '\u22a7',
    '\\twoheadleftarrow': '\u219e',
    '\\twoheadrightarrow': '\u21a0',
    '\\ulcorner'        : '\u231c',
    '\\updownarrow'     : '\u2195',
    '\\upharpoonleft'   : '\u21bf',
    '\\upharpoonright'  : '\u21be',
    '\\upslopeellipsis' : '\u22f0',
    '\\upuparrows'      : '\u21c8',
    '\\uranus'          : '\u2645',
    '\\urcorner'        : '\u231d',
    '\\varepsilon'      : '\u025b',
    '\\varkappa'        : '\u03f0',
    '\\varnothing'      : '\u2205',
    '\\varphi'          : '\u03c6',
    '\\varpi'           : '\u03d6',
    '\\varrho'          : '\u03f1',
    '\\varsigma'        : '\u03c2',
    '\\vartriangle'     : '\u25b5',
    '\\vartriangleleft' : '\u22b2',
    '\\vartriangleright': '\u22b3',
    '\\vdots'           : '\u22ee',
    '\\veebar'          : '\u22bb',
    '\\venus'           : '\u2640',
    '\\vert'            : '|',
    '\\verymuchgreater' : '\u22d9',
    '\\verymuchless'    : '\u22d8',
    '\\virgo'           : '\u264d',
    '\\volintegral'     : '\u2230',
    '\\wp'              : '\u2118',
    '\\wr'              : '\u2240',
}

class RenderState:
    """Holds the state of the rendering."""
    def __init__(self, font, painter, x, y, alignhorz,
                 actually_render=True):
        self.font = font
        self.painter = painter
        self.x = x     # current x position
        self.y = y     # current y position
        self.alignhorz = alignhorz
        self.actually_render = actually_render
        self.maxlines = 1 # maximim number of lines drawn

    def fontMetrics(self):
        """Returns font metrics object."""
        return FontMetrics(self.font, self.painter.device())

    def getPixelsPerPt(self):
        """Return number of pixels per point in the rendering."""
        return self.painter.pixperpt

class Part:
    """Represents a part of the text to be rendered, made up of smaller parts."""
    def __init__(self, children):
        self.children = children

    def render(self, state):
        for p in self.children:
            p.render(state)

class PartText(Part):
    """Fundamental bit of text to be rendered: some text."""
    def __init__(self, text):
        self.text = text

    def addText(self, text):
        self.text += text

    def render(self, state):
        """Render some text."""

        width = state.fontMetrics().horizontalAdvance(self.text)

        # actually write the text if requested
        if state.actually_render:
            state.painter.drawText( qt.QPointF(state.x, state.y), self.text )

        # move along, nothing to see
        state.x += width

class PartLines(Part):
    """Render multiple lines."""

    def __init__(self, children):
        Part.__init__(self, children)
        self.widths = []

    def render(self, state):
        """Render multiple lines."""
        # record widths of individual lines
        if not state.actually_render:
            self.widths = []

        height = state.fontMetrics().height()
        inity = state.y
        initx = state.x

        state.y -= height*(len(self.children)-1)

        # iterate over lines (reverse as we draw from bottom up)
        for i, part in enumerate(self.children):
            if state.actually_render and self.widths:
                xwidth = max(self.widths)
                # if we're rendering, use max width to justify line
                if state.alignhorz < 0:
                    # left alignment
                    state.x = initx
                elif state.alignhorz == 0:
                    # centre alignment
                    state.x = initx + (xwidth - self.widths[i])*0.5
                elif state.alignhorz > 0:
                    # right alignment
                    state.x = initx + (xwidth - self.widths[i])
            else:
                # if not, just left justify to get widths
                state.x = initx

            # render the line itself
            part.render(state)

            # record width if we're not rendering
            if not state.actually_render:
                self.widths.append( state.x - initx )
            # move up a line
            state.y += height

        # move on x posn
        if self.widths:
            state.x = initx + max(self.widths)
        else:
            state.x = initx
        state.y = inity
        # keep track of number of lines rendered
        state.maxlines = max(state.maxlines, len(self.children))

class PartSuperScript(Part):
    """Represents superscripted part."""
    def render(self, state):
        font = state.font
        painter = state.painter

        # change text height
        oldheight = state.fontMetrics().height()
        size = font.pointSizeF()
        font.setPointSizeF(size*0.6)
        painter.setFont(font)

        # set position
        oldy = state.y
        state.y -= oldheight*0.4

        # draw children
        Part.render(self, state)

        # restore font and position
        state.y = oldy
        font.setPointSizeF(size)
        painter.setFont(font)

class PartFrac(Part):
    """"A fraction, do latex \frac{a}{b}."""

    def render(self, state):
        if len(self.children) != 2:
            return

        font = state.font
        painter = state.painter

        # make font half size
        size = font.pointSizeF()
        font.setPointSizeF(size*0.5)
        painter.setFont(font)

        # keep track of width above and below line
        if not state.actually_render:
            self.widths = []

        initx = state.x
        inity = state.y

        # render bottom of fraction
        if state.actually_render and len(self.widths) == 2:
            # centre line
            state.x = initx + (max(self.widths) - self.widths[0])*0.5
        self.children[1].render(state)
        if not state.actually_render:
            # get width if not rendering
            self.widths.append(state.x - initx)

        # render top of fraction
        m = state.fontMetrics()
        state.y -= (m.ascent() + m.descent())
        if state.actually_render and len(self.widths) == 2:
            # centre line
            state.x = initx + (max(self.widths) - self.widths[1])*0.5
        else:
            state.x = initx
        self.children[0].render(state)
        if not state.actually_render:
            self.widths.append(state.x - initx)

        state.x = initx + max(self.widths)
        state.y = inity

        # restore font
        font.setPointSizeF(size)
        painter.setFont(font)
        height = state.fontMetrics().ascent()

        # draw line between lines with 0.5pt thickness
        painter.save()
        pen = painter.pen()
        painter.setPen( qt.QPen(
            painter.pen().brush(), state.getPixelsPerPt()*0.5) )
        painter.setPen(pen)

        painter.drawLine(
            qt.QPointF(initx, inity-height/2.),
            qt.QPointF(initx+max(self.widths), inity-height/2.) )

        painter.restore()

class PartSubScript(Part):
    """Represents subscripted part."""
    def render(self, state):
        font = state.font

        # change text height
        size = font.pointSizeF()
        font.setPointSizeF(size*0.6)
        state.painter.setFont(font)

        # set position
        oldy = state.y
        state.y += state.fontMetrics().descent()

        # draw children
        Part.render(self, state)

        # restore font and position
        state.y = oldy
        font.setPointSizeF(size)
        state.painter.setFont(font)

class PartMultiScript(Part):
    """Represents multiple parts with the same starting x, e.g. a combination of
       super- and subscript parts."""
    def render(self, state):
        oldx = state.x
        newx = oldx
        for p in self.children:
            state.x = oldx
            p.render(state)
            newx = max([state.x, newx])
        state.x = newx

    def append(self, p):
        self.children.append(p)

class PartItalic(Part):
    """Represents italic part."""
    def render(self, state):
        font = state.font

        font.setItalic( not font.italic() )
        state.painter.setFont(font)

        Part.render(self, state)

        font.setItalic( not font.italic() )
        state.painter.setFont(font)

class PartBold(Part):
    """Represents bold part."""
    def render(self, state):
        font = state.font

        font.setBold( not font.bold() )
        state.painter.setFont(font)

        Part.render(self, state)

        font.setBold( not font.bold() )
        state.painter.setFont(font)

class PartUnderline(Part):
    """Represents underlined part."""
    def render(self, state):
        font = state.font

        font.setUnderline( not font.underline() )
        state.painter.setFont(font)

        Part.render(self, state)

        font.setUnderline( not font.underline() )
        state.painter.setFont(font)

class PartFont(Part):
    """Change font name in part."""
    def __init__(self, children):
        try:
            self.fontname = children[0].text
        except (AttributeError, IndexError):
            self.fontname = ''
        self.children = children[1:]

    def render(self, state):
        font = state.font
        oldfamily = font.family()
        font.setFamily(self.fontname)
        state.painter.setFont(font)

        Part.render(self, state)

        font.setFamily(oldfamily)
        state.painter.setFont(font)

class PartSize(Part):
    """Change font size in part."""
    def __init__(self, children):
        self.size = None
        self.deltasize = None

        # convert size
        try:
            size = children[0].text.replace('pt', '') # crap code
            if size[:1] in '+-':
                # is a modification of font size
                self.deltasize = float(size)
            else:
                # is an absolute font size
                self.size = float(size)
        except (AttributeError, ValueError, IndexError):
            self.deltasize = 0.

        self.children = children[1:]

    def render(self, state):
        font = state.font
        size = oldsize = font.pointSizeF()

        if self.size:
            # absolute size
            size = self.size
        elif self.deltasize:
            # change of size
            size = max(size+self.deltasize, 0.1)

        font.setPointSizeF(size)
        state.painter.setFont(font)

        Part.render(self, state)

        font.setPointSizeF(oldsize)
        state.painter.setFont(font)

class PartBar(Part):
    """Draw a bar over text."""

    def render(self, state):
        initx = state.x

        # draw material under bar
        Part.render(self, state)

        # draw line over text with 0.5pt thickness
        painter = state.painter
        height = state.fontMetrics().ascent()

        painter.save()
        penw = state.getPixelsPerPt()*0.5
        painter.setPen( qt.QPen(painter.pen().brush(), penw) )
        painter.drawLine(
            qt.QPointF(initx, state.y-height+penw),
            qt.QPointF(state.x, state.y-height+penw))
        painter.restore()

class PartHat(Part):
    """Draw a hat over text."""

    def render(self, state):
        initx = state.x

        # draw material under bar
        Part.render(self, state)

        # draw line over text with 0.5pt thickness
        painter = state.painter
        height = state.fontMetrics().ascent()

        painter.save()
        penw = state.getPixelsPerPt()*0.5
        painter.setPen( qt.QPen(painter.pen().brush(), penw) )
        hatheight = min((state.x-initx)/2, height / 3.)
        painter.drawLine(
            qt.QPointF(initx, state.y-height+penw),
            qt.QPointF((initx+state.x)/2, state.y-height+penw-hatheight))
        painter.drawLine(
            qt.QPointF((initx+state.x)/2, state.y-height+penw-hatheight),
            qt.QPointF(state.x, state.y-height+penw))
        painter.restore()

class PartDot(Part):
    """Draw a dot over text."""

    def render(self, state):
        initx = state.x

        # draw material under bar
        Part.render(self, state)

        # draw circle over text with 1pt radius
        painter = state.painter
        height = state.fontMetrics().ascent()

        painter.save()
        circsize = state.getPixelsPerPt()
        painter.setBrush( qt.QBrush(painter.pen().color()) )
        painter.setPen( qt.QPen(qt.Qt.PenStyle.NoPen) )

        x = 0.5*(initx + state.x)
        y = state.y-height + circsize
        painter.drawEllipse( qt.QRectF(
            qt.QPointF(x-circsize,y-circsize),
            qt.QPointF(x+circsize,y+circsize)) )
        painter.restore()

class PartDDot(Part):
    """Draw a double dot over text."""

    def render(self, state):
        initx = state.x

        # draw material under bar
        Part.render(self, state)

        # draw circle over text with 1pt radius
        painter = state.painter
        height = state.fontMetrics().ascent()

        painter.save()
        circsize = state.getPixelsPerPt()
        painter.setBrush( qt.QBrush(painter.pen().color()) )
        painter.setPen( qt.QPen(qt.Qt.PenStyle.NoPen) )

        x1 = initx + 0.25*(state.x-initx)
        x2 = initx + 0.75*(state.x-initx)
        y = state.y-height + circsize
        for x in x1, x2:
            painter.drawEllipse( qt.QRectF(
                qt.QPointF(x-circsize,y-circsize),
                qt.QPointF(x+circsize,y+circsize)) )
        painter.restore()

class PartTilde(Part):
    """Draw a tilde ~ over text."""

    def render(self, state):

        initx = state.x

        # draw material under tilde
        Part.render(self, state)

        # change text height
        font = state.font
        size = font.pointSizeF()
        height = state.fontMetrics().capHeight()

        font.setPointSizeF(size*0.7)
        state.painter.setFont(font)

        # set x and y positions for tilde
        tildew = state.fontMetrics().horizontalAdvance('~')

        if not font.italic() or (state.x-initx) >= 2*tildew:
            over_pos = state.x - (state.x - initx)*0.5 - tildew*0.5
        else:
            over_pos = state.x - tildew

        # paint tilde over a text
        state.painter.drawText(qt.QPointF(over_pos, state.y - height), '~')

        font.setPointSizeF(size)
        state.painter.setFont(font)

class PartMarker(Part):
    """Draw a marker symbol."""

    def render(self, state):
        painter = state.painter
        size = state.fontMetrics().ascent()

        painter.save()
        pen = painter.pen()
        pen.setWidthF( state.getPixelsPerPt() * 0.5 )
        painter.setPen(pen)

        try:
            points.plotMarker(
                painter, state.x + size/2.,
                state.y - size/2.,
                self.children[0].text, size*0.3)
        except ValueError:
            pass

        painter.restore()

        state.x += size

class PartColor(Part):
    def __init__(self, children):
        try:
            self.colorname = children[0].text
        except (AttributeError, IndexError):
            self.colorname = ''
        self.children = children[1:]

    def render(self, state):
        painter = state.painter
        pen = painter.pen()
        oldcolor = pen.color()

        pen.setColor( painter.docColor(self.colorname) )
        painter.setPen(pen)

        Part.render(self, state)

        pen.setColor(oldcolor)
        painter.setPen(pen)

# a dict of latex commands, the part object they correspond to,
# and the number of arguments
part_commands = {
    '^': (PartSuperScript, 1),
    '_': (PartSubScript, 1),
    r'\italic': (PartItalic, 1),
    r'\emph': (PartItalic, 1),
    r'\bold': (PartBold, 1),
    r'\underline': (PartUnderline, 1),
    r'\textbf': (PartBold, 1),
    r'\textit': (PartItalic, 1),
    r'\font': (PartFont, 2),
    r'\size': (PartSize, 2),
    r'\frac': (PartFrac, 2),
    r'\bar': (PartBar, 1),
    r'\overline': (PartBar, 1),
    r'\hat': (PartHat, 1),
    r'\dot': (PartDot, 1),
    r'\ddot': (PartDDot, 1),
    r'\wtilde': (PartTilde, 1),
    r'\marker': (PartMarker, 1),
    r'\color': (PartColor, 2),
}

# split up latex expression into bits
splitter_re = re.compile(r'''
(
\\[A-Za-z]+[ ]* |   # normal latex command
\\[\[\]{}_^] |      # escaped special characters
\\\\ |              # line end
\{ |                # begin block
\} |                # end block
\^ |                # power
_                   # subscript
)
''', re.VERBOSE)

def latexEscape(text):
    """Escape any special characters in LaTex-like code."""
    # \\ is converted a unicode-special character and replaced again
    # with the latex code later, to avoid its parts being replaced in
    # the next step
    text = text.replace('\\', '\ue000')
    # replace _, ^, {, }, [ and ] with escaped versions
    text = re.sub(r'([_\^\[\]\{\}])', r'\\\1', text)
    text = text.replace('\ue000', '{\\backslash}')
    return text

def makePartList(text):
    """Make list of parts from text"""
    parts = []
    parents = [parts]

    def doAdd(p):
        """Add the part at the correct level."""
        parents[-1].append(p)
        return p

    for p in splitter_re.split(text):
        if p[:1] == '\\':
            # we may need to drop excess spaces after \foo commands
            ps = p.rstrip()
            if ps in symbols:
                # it will become a symbol, so preserve whitespace
                doAdd(ps)
                if ps != p:
                    doAdd(p[len(ps)-len(p):])
            else:
                # add as possible command, so drop excess whitespace
                doAdd(ps)
        elif p == '{':
            # add a new level
            parents.append( doAdd([]) )
        elif p == '}':
            if len(parents) > 1:
                parents.pop()
        elif p:
            # if not blank, keep it
            doAdd(p)
    return parts

def makePartTree(partlist):
    """Make a tree of parts from the part list."""

    lines = []
    itemlist = []
    length = len(partlist)

    def addText(text):
        """Try to merge consecutive text items for better rendering."""
        if itemlist and isinstance(itemlist[-1], PartText):
            itemlist[-1].addText(text)
        else:
            itemlist.append( PartText(text) )

    i = 0
    while i < length:
        p = partlist[i]
        if p == r'\\':
            lines.append( Part(itemlist) )
            itemlist = []
        elif isinstance(p, str):
            if p in symbols:
                addText(symbols[p])
            elif p in part_commands:
                klass, numargs = part_commands[p]
                if numargs == 1 and len(partlist) > i+1 and isinstance(partlist[i+1], str):
                    # coerce a single argument to a partlist so that things
                    # like "A^\dagger" render correctly without needing
                    # curly brackets
                    partargs = [makePartTree([partlist[i+1]])]
                else:
                    partargs = [makePartTree(k) for k in partlist[i+1:i+numargs+1]]

                if (p == '^' or p == '_'):
                    if len(itemlist) > 0 and (
                        isinstance(itemlist[-1], PartSubScript) or
                        isinstance(itemlist[-1], PartSuperScript) or
                        isinstance(itemlist[-1], PartMultiScript)):
                        # combine sequences of multiple sub-/superscript parts into
                        # a MultiScript item so that a single text item can have
                        # both super and subscript indicies
                        # e.g. X^{(q)}_{i}
                        if isinstance(itemlist[-1], PartMultiScript):
                            itemlist.append( klass(partargs) )
                        else:
                            itemlist[-1] = PartMultiScript([itemlist[-1], klass(partargs)])
                    else:
                        itemlist.append( klass(partargs) )
                else:
                    itemlist.append( klass(partargs) )
                i += numargs
            else:
                addText(p)
        else:
            itemlist.append( makePartTree(p) )
        i += 1
    # remaining items
    lines.append( Part(itemlist) )

    if len(lines) == 1:
        # single line, so optimize (itemlist == lines[0] still)
        if len(itemlist) == 1:
            # try to flatten any excess layers
            return itemlist[0]
        else:
            return lines[0]
    else:
        return PartLines(lines)

class _Renderer:
    """Different renderer types based on this."""

    def __init__(
            self, painter, font, x, y, text,
            alignhorz = -1, alignvert = -1, angle = 0,
            usefullheight = False,
            doc = None):

        self.painter = painter
        self.font = font
        self.alignhorz = alignhorz
        self.alignvert = alignvert
        self.angle = angle
        self.usefullheight = usefullheight
        self.doc = doc

        # x and y are the original coordinates
        # xi and yi are adjusted for alignment
        self.x = self.xi = x
        self.y = self.yi = y
        self.calcbounds = None

        self._initText(text)

    def _initText(self, text):
        """Override this to set up renderer with text."""

    def ensureInBox(self, minx = -32767, maxx = 32767,
                    miny = -32767, maxy = 32767, extraspace = False):
        """Adjust position of text so that it is within this box."""

        if self.calcbounds is None:
            self.getBounds()

        cb = self.calcbounds

        # add a small amount of extra room if requested
        if extraspace:
            self.painter.setFont(self.font)
            l = FontMetrics(
                self.font,
                self.painter.device()).height()*0.2
            miny += l

        # twiddle positions and bounds
        if cb[2] > maxx:
            dx = cb[2] - maxx
            self.xi -= dx
            cb[2] -= dx
            cb[0] -= dx

        if cb[0] < minx:
            dx = minx - cb[0]
            self.xi += dx
            cb[2] += dx
            cb[0] += dx

        if cb[3] > maxy:
            dy = cb[3] - maxy
            self.yi -= dy
            cb[3] -= dy
            cb[1] -= dy

        if cb[1] < miny:
            dy = miny - cb[1]
            self.yi += dy
            cb[3] += dy
            cb[1] += dy

    def getDimensions(self):
        """Get the (w, h) of the bounding box."""

        if self.calcbounds is None:
            self.getBounds()
        cb = self.calcbounds
        return (cb[2]-cb[0]+1, cb[3]-cb[1]+1)

    def _getWidthHeight(self):
        """Calculate the width and height of rendered text.

        Return totalwidth, totalheight, dy
        dy is a descent to add, to include in the alignment, if wanted
        """

    def getTightBounds(self):
        """Get bounds in form of rotated rectangle."""

        largebounds = self.getBounds()

        totalwidth, totalheight, dy = self._getWidthHeight()

        return RotatedRectangle(
            0.5*(largebounds[0]+largebounds[2]),
            0.5*(largebounds[1]+largebounds[3]),
            totalwidth,
            totalheight+dy,
            self.angle * math.pi / 180.)

    def getBounds(self):
        """Get bounds in standard version."""

        if self.calcbounds is not None:
            return self.calcbounds

        totalwidth, totalheight, dy = self._getWidthHeight()

        # in order to work out text position, we rotate a bounding box
        # in fact we add two extra points to account for descent if reqd
        tw = totalwidth / 2
        th = totalheight / 2
        coordx = N.array( [-tw,  tw,  tw, -tw, -tw,    tw   ] )
        coordy = N.array( [ th,  th, -th, -th,  th+dy, th+dy] )

        # rotate angles by theta
        theta = -self.angle * (math.pi / 180.)
        c = math.cos(theta)
        s = math.sin(theta)
        newx = coordx*c + coordy*s
        newy = coordy*c - coordx*s

        # calculate bounding box
        newbound = (newx.min(), newy.min(), newx.max(), newy.max())

        # use rotated bounding box to find position of start text posn
        if self.alignhorz < 0:
            xr = ( self.x, self.x+(newbound[2]-newbound[0]) )
            self.xi += (newx[0] - newbound[0])
        elif self.alignhorz > 0:
            xr = ( self.x-(newbound[2]-newbound[0]), self.x )
            self.xi += (newx[0] - newbound[2])
        else:
            xr = ( self.x+newbound[0], self.x+newbound[2] )
            self.xi += newx[0]

        # y alignment
        # adjust y by these values to ensure proper alignment
        if self.alignvert < 0:
            yr = ( self.y + (newbound[1]-newbound[3]), self.y )
            self.yi += (newy[0] - newbound[3])
        elif self.alignvert > 0:
            yr = ( self.y, self.y + (newbound[3]-newbound[1]) )
            self.yi += (newy[0] - newbound[1])
        else:
            yr = ( self.y+newbound[1], self.y+newbound[3] )
            self.yi += newy[0]

        self.calcbounds = [xr[0], yr[0], xr[1], yr[1]]
        return self.calcbounds

class _StdRenderer(_Renderer):
    """Standard rendering class."""

    # expresions in brackets %{{ }}% are evaluated
    exprexpansion = re.compile(r'%\{\{(.+?)\}\}%')

    def _initText(self, text):

        # expand any expressions in the text
        delta = 0
        for m in self.exprexpansion.finditer(text):
            expanded = self._expandExpr(m.group(1))
            text = text[:delta+m.start()] + expanded + text[delta+m.end():]
            delta += len(expanded) - (m.end()-m.start())

        # make internal tree
        partlist = makePartList(text)
        self.parttree = makePartTree(partlist)

    def _expandExpr(self, expr):
        """Expand expression."""
        if self.doc is None:
            return _("* Evaluation not supported here *")
        else:
            expr = expr.strip()
            try:
                comp = self.doc.evaluate.compileCheckedExpression(expr)
                if comp is None:
                    return _("* Evaluation error *")

                return str(eval(comp, self.doc.evaluate.context))
            except Exception as e:
                return _("* Evaluation error: %s *") % latexEscape(str(e))

    def _getWidthHeight(self):
        """Get size of box around text."""

        # work out total width and height
        self.painter.setFont(self.font)

        # work out height of box, and
        # make the bounding box a bit bigger if we want to include descents

        state = RenderState(
            self.font, self.painter, 0, 0,
            self.alignhorz,
            actually_render = False)

        fm = state.fontMetrics()

        if self.usefullheight:
            totalheight = fm.ascent()
            dy = fm.descent()
        else:
            if self.alignvert == 0:
                # if want vertical centering, better to centre around middle
                # of typical letter (i.e. where strike position is)
                #totalheight = fm.strikeOutPos()*2
                totalheight = fm.boundingRectChar('0').height()
            else:
                # if top/bottom alignment, better to use maximum letter height
                totalheight = fm.ascent()
            dy = 0

        # work out width
        self.parttree.render(state)
        totalwidth = state.x
        # add number of lines for height
        totalheight += fm.height()*(state.maxlines-1)

        return totalwidth, totalheight, dy

    def render(self):
        """Render the text."""

        if self.calcbounds is None:
            self.getBounds()

        state = RenderState(
            self.font, self.painter,
            self.xi, self.yi,
            self.alignhorz)

        # if the text is rotated, change the coordinate frame
        if self.angle != 0:
            self.painter.save()
            self.painter.translate( qt.QPointF(state.x, state.y) )
            self.painter.rotate(self.angle)
            state.x = 0
            state.y = 0

        # actually paint the string
        self.painter.setFont(self.font)
        self.parttree.render(state)

        # restore coordinate frame if text was rotated
        if self.angle != 0:
            self.painter.restore()

        # caller might want this information
        return self.calcbounds

class _MmlRenderer(_Renderer):
    """MathML renderer."""

    def _initText(self, text):
        """Setup MML document and draw it in recording paint device."""

        self.error = ''
        self.size = qt.QSize(1, 1)

        self.mmldoc = doc = qtmml.QtMmlDocument()
        try:
            self.mmldoc.setContent(text)
        except ValueError as e:
            self.mmldoc = None
            self.error = _('Error interpreting MathML: %s\n') % str(e)
            return

        # this is pretty horrible :-(

        # We write the mathmml document to a RecordPaintDevice device
        # at the same DPI as the screen, because the MML code breaks
        # for other DPIs. We then repaint the output to the real
        # device, scaling to make the size correct.

        screendev = qt.QApplication.desktop()
        self.record = recordpaint.RecordPaintDevice(
            1024, 1024, screendev.logicalDpiX(), screendev.logicalDpiY())

        rpaint = qt.QPainter(self.record)
        # painting code relies on these attributes of the painter
        rpaint.pixperpt = screendev.logicalDpiY() / 72.
        rpaint.scaling = 1.0

        # Upscale any drawing by this factor, then scale back when
        # drawing. We have to do this to get consistent output at
        # different zoom factors (I hate this code).
        upscale = 5.

        doc.setFontName( qtmml.QtMmlWidget.NormalFont, self.font.family() )

        ptsize = self.font.pointSizeF()
        if ptsize < 0:
            ptsize = self.font.pixelSize() / self.painter.pixperpt

        doc.setBaseFontPointSize(int(ptsize * upscale))

        # the output will be painted finally scaled
        self.drawscale = (
            self.painter.dpi / screendev.logicalDpiY()
            / upscale )
        self.size = doc.size() * self.drawscale

        doc.paint(rpaint, qt.QPoint(0, 0))
        rpaint.end()

    def _getWidthHeight(self):
        return self.size.width(), self.size.height(), 0

    def render(self):
        """Render the text."""

        if self.calcbounds is None:
            self.getBounds()

        p = self.painter
        p.save()
        if self.mmldoc is not None:
            p.translate(self.xi, self.yi)
            p.rotate(self.angle)
            # is drawn from bottom of box, not top
            p.translate(0, -self.size.height())
            p.scale(self.drawscale, self.drawscale)
            self.record.play(p)
        else:
            # display an error - must be a better way to do this
            p.setFont(qt.QFont())
            p.setPen(qt.QPen(qt.QColor("red")))
            p.drawText(
                qt.QRectF(self.xi, self.yi, 200, 200),
                qt.Qt.AlignmentFlag.AlignLeft | qt.Qt.AlignmentFlag.AlignTop | qt.Qt.TextFlag.TextWordWrap,
                self.error )
        p.restore()

        return self.calcbounds

# identify mathml text
mml_re = re.compile(r'^\s*<math.*</math\s*>\s*$', re.DOTALL)

def Renderer(painter, font, x, y, text,
             alignhorz = -1, alignvert = -1, angle = 0,
             usefullheight = False,
             doc = None):

    """Return an appropriate Renderer object depending on the text.
    This looks like a class name, because it was a class originally.

    painter is the painter to draw on
    font is the starting font to use
    x and y are the x and y positions to draw the text at
    alignhorz = (-1, 0, 1) for (left, centre, right) alignment
    alignvert = (-1, 0, 1) for (above, centre, below) alignment
    angle is the angle to draw the text at
    usefullheight means include descenders in calculation of height
    of text
    doc is a Document for evaluating any expressions

    alignment is in the painter frame, not the text frame
    """

    if mml_re.match(text):
        r = _MmlRenderer
    else:
        r = _StdRenderer

    return r(
        painter, font, x, y, text,
        alignhorz=alignhorz, alignvert=alignvert,
        angle=angle, usefullheight=usefullheight,
        doc=doc
    )
