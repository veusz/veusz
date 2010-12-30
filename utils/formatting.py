#    Copyright (C) 2010 Jeremy S. Sanders
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

# $Id$

import re
import math

import dates
import veusz.qtall as qt4

_formaterror = 'FormatError'

# a format statement in a string
_format_re = re.compile(r'%([-#0-9 +.hlL]*?)([diouxXeEfFgGcrs%])')

def localeFormat(totfmt, args, locale=None):
    """Format using fmt statement fmt, qt QLocale object locale and
    arguments to formatting args.

    * arguments are not supported in this formatting, nor is using
    a dict to supply values for statement
    """

    # substitute all format statements with string format statements
    newfmt = _format_re.sub("%s", totfmt)
    
    # do formatting separately for all statements
    strings = []
    i = 0
    for f in _format_re.finditer(totfmt):
        code = f.group(2)
        if code == '%':
            s = '%'
        else:
            try:
                s = f.group() % args[i]
                i += 1
            except IndexError:
                raise TypeError, "Not enough arguments for format string"
            if locale is not None and code in 'eEfFgG':
                s = s.replace('.', qt4.QString(locale.decimalPoint()))

        strings.append(s)

    if i != len(args):
        raise TypeError, "Not all arguments converted during string formatting"

    return newfmt % tuple(strings)

def formatSciNotation(num, formatargs, locale=None):
    """Format number into form X \times 10^{Y}.
    This function trims trailing zeros and decimal point unless a formatting
    argument is supplied

    This is similar to the %e format string
    formatargs is the standard argument in a format string to control the
    number of decimal places, etc.

    locale is a QLocale object
    """

    # create an initial formatting string
    if formatargs:
        format = '%' + formatargs + 'e'
    else:
        format = '%.10e'

    # try to format the number
    # this may be user-supplied data, so don't crash hard by returning
    # useless output
    try:
        text = format % num
    except:
        return _formaterror

    # split around the exponent
    leader, exponent = text.split('e')

    # strip off trailing decimal point and zeros if no format args
    if not formatargs:
        leader = '%.10g' % float(leader)

    # trim off leading 1
    if leader == '1' and not formatargs:
        leader = ''
    else:
        # the unicode string is a small space, multiply and small space
        leader += u'\u00d7'

    # do substitution of decimals
    if locale is not None:
        leader = leader.replace('.', qt4.QString(locale.decimalPoint()))

    return '%s10^{%i}' % (leader, int(exponent))

def formatGeneral(num, fmtarg, locale=None):
    """General formatting which switches from normal to scientic
    notation."""
    
    a = abs(num)
    # manually choose when to switch from normal to scientific
    # as the default isn't very good
    if a >= 1e4 or (a < 1e-2 and a > 1e-110):
        retn = formatSciNotation(num, fmtarg, locale=locale)
    else:
        if fmtarg:
            f = '%' + fmtarg + 'g'
        else:
            f = '%.10g'

        try:
            retn = f % num
        except TypeError:
            retn = _formaterror

        if locale is not None:
            retn = retn.replace('.', qt4.QString(locale.decimalPoint()))
    return retn

engsuffixes = ( 'y', 'z', 'a', 'f', 'p', 'n',
                u'\u03bc', 'm', '', 'k', 'M', 'G',
                'T', 'P', 'E', 'Z', 'Y' )

def formatEngineering(num, fmtarg, locale=None):
    """Engineering suffix format notation using SI suffixes."""

    if num != 0.:
        logindex = math.log10( abs(num) ) / 3.

        # for numbers < 1 round down suffix
        if logindex < 0. and (int(logindex)-logindex) > 1e-6:
            logindex -= 1

        # make sure we don't go out of bounds
        logindex = min( max(logindex, -8),
                        len(engsuffixes) - 9 )

        suffix = engsuffixes[ int(logindex) + 8 ]
        val = num / 10**( int(logindex) *3)
    else:
        suffix = ''
        val = num

    text = ('%' + fmtarg + 'g%s') % (val, suffix)
    if locale is not None:
        text = text.replace('.', qt4.QString(locale.decimalPoint()))
    return text

# catch general veusz formatting expression
_formatRE = re.compile(r'%([^A-Za-z]*)(VDVS|VD.|V.|[A-Za-z])')

def formatNumber(num, format, locale=None):
    """ Format a number in different ways.

    format is a standard C format string, with some additions:
     %Ve    scientific notation X \times 10^{Y}
     %Vg    switches from normal notation to scientific outside 10^-2 to 10^4
     %VE    engineering suffix option

     %VDx   date formatting, where x is one of the arguments in 
            http://docs.python.org/lib/module-time.html in the function
            strftime
    """

    while True:
        # repeatedly try to do string format
        m = _formatRE.search(format)
        if not m:
            break

        # argument and type of formatting
        farg, ftype = m.groups()

        # special veusz formatting
        if ftype[:1] == 'V':
            # special veusz formatting
            if ftype == 'Ve':
                out = formatSciNotation(num, farg, locale=locale)
            elif ftype == 'Vg':
                out = formatGeneral(num, farg, locale=locale)
            elif ftype == 'VE':
                out = formatEngineering(num, farg, locale=locale)
            elif ftype[:2] == 'VD':
                d = dates.floatToDateTime(num)
                # date formatting (seconds since start of epoch)
                if ftype[:4] == 'VDVS':
                    # special seconds operator
                    out = ('%'+ftype[4:]+'g') % (d.second+d.microsecond*1e-6)
                else:
                    # use date formatting
                    try:
                        out = d.strftime(str('%'+ftype[2:]))
                    except ValueError:
                        out = _formaterror
            else:
                out = _formaterror

            # replace hyphen with true - and small space
            out = out.replace('-', u'\u2212')

        else:
            # standard C formatting
            try:
                out = localeFormat('%' + farg + ftype, (num,), locale=locale)
            except:
                out = _formaterror

        format = format[:m.start()] + out + format[m.end():]

    return format
