#    Copyright (C) 2005 Jeremy S. Sanders
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
#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
###############################################################################

# $Id$

import sys

# get modules relative to above directory
sys.path.append('../')

import unittest
import setting

# test setting routines

class TestSettings(unittest.TestCase):
    def setUp(self):
        pass

    def testStr(self):
        s = setting.Str('test', 'foobar')
        self.assertEqual(s.getName(), 'test')
        self.assertEqual(s.get(), 'foobar')

        s.set('plugh')
        self.assertEqual(s.get(), 'plugh')

        s.set( s.fromText('hi') )
        self.assertEqual(s.get(), 'hi')

        self.assertRaises(setting.InvalidType, s.set, 10)

    def testBool(self):
        s = setting.Bool('test', True)
        self.assertEqual(s.getName(), 'test')
        self.assertEqual(s.get(), True)

        s.set(False)
        self.assertEqual(s.get(), False)
        s.set(1)
        self.assertEqual(s.get(), True)

        self.assertRaises(setting.InvalidType, s.set, 'foo')
        self.assertRaises(setting.InvalidType, s.set, 3.14)

        s.set( s.fromText('tRuE') )
        self.assertEqual(s.get(), True)
        s.set( s.fromText('0') )
        self.assertEqual(s.get(), False)

    def testInt(self):
        s = setting.Int('test', 42)
        self.assertEqual(s.get(), 42)
        s.set(89)
        self.assertEqual(s.get(), 89)
        self.assertEqual(s.getName(), 'test')

        self.assertEqual(s.toText(), '89')

        s.set( s.fromText('43') )
        self.assertEqual(s.get(), 43)

        self.assertRaises(setting.InvalidType, s.set, 'foo')
        self.assertRaises(setting.InvalidType, s.fromText, 'foo')

    def testFloat(self):
        # it's a wonder this works with FP accuarcy
        s = setting.Float('test', 42.)
        self.assertEqual(s.get(), 42.)
        s.set(89.)
        self.assertEqual(s.get(), 89.)
        self.assertEqual(s.getName(), 'test')

        self.assertEqual(s.toText(), '89.0')

        s.set( s.fromText('43') )
        self.assertEqual(s.get(), 43.0)

        self.assertRaises(setting.InvalidType, s.fromText, 'foo')

    def testIntOrAuto(self):
        s = setting.IntOrAuto('test', 42)
        self.assertEqual(s.get(), 42)
        s.set(89)
        self.assertEqual(s.get(), 89)
        self.assertEqual(s.getName(), 'test')

        self.assertEqual(s.toText(), '89')

        s.set( s.fromText('43') )
        self.assertEqual(s.get(), 43)

        self.assertRaises(setting.InvalidType, s.fromText, 'foo')

        s.set('auto')
        self.assertEqual( s.get(), 'Auto' )
        s.set('Auto')
        self.assertEqual( s.get(), 'Auto' )

        s.set( s.fromText('AuTo') )
        self.assertEqual(s.get(), 'Auto')

    def testFloatOrAuto(self):
        # it's a wonder this works with FP accuarcy
        s = setting.FloatOrAuto('test', 42.)
        self.assertEqual(s.get(), 42.)
        s.set(89.)
        self.assertEqual(s.get(), 89.)
        self.assertEqual(s.getName(), 'test')

        self.assertEqual(s.toText(), '89.0')

        s.set( s.fromText('43') )
        self.assertEqual(s.get(), 43.0)

        self.assertRaises(setting.InvalidType, s.fromText, 'foo')

        s.set('auto')
        self.assertEqual( s.get(), 'Auto' )
        s.set('Auto')
        self.assertEqual( s.get(), 'Auto' )

        s.set( s.fromText('AuTo') )
        self.assertEqual(s.get(), 'Auto')

    def testDistance(self):
        s = setting.Distance('test', '3pt')
        self.assertEqual(s.get(), '3pt')

        s.set('1 inch')
        s.set('3 cm')
        s.set('5.3%')
        s.set('3/4')
        s.set('3.3mm')

        s.set( s.fromText('3.5mm') )
        self.assertEqual(s.get(), '3.5mm')

        self.assertRaises(setting.InvalidType, s.set, 'foo')
        self.assertRaises(setting.InvalidType, s.set, '5 lightyears')
        self.assertRaises(setting.InvalidType, s.fromText, '42 hours')

    def testChoice(self):
        choices = ['alpha', 'beta', 'gamma', 'delta']
        s = setting.Choice('test', choices, 'alpha')
        self.assertEqual(s.get(), 'alpha')

        s.set('beta')
        self.assertEqual(s.get(), 'beta')

        s.set( s.fromText('gamma') )
        self.assertEqual(s.get(), 'gamma')

        self.assertRaises(setting.InvalidType, s.set, 'foo')
        self.assertRaises(setting.InvalidType, s.fromText, 'bar')

    def testSettings(self):

        ss = setting.Settings('my settings')

        ss.add( setting.Int('dna', 42) )

if __name__ == '__main__':
    unittest.main()

