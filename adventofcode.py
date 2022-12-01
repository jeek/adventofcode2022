"""Advent of Code 2021."""

import unittest
from collections import defaultdict
from copy import copy
from itertools import product
from heapq import heappop, heappush

def problem01(inputfile="01.input", part=1):
    """Problem #1."""
    with open(inputfile) as file:
        return 0

TESTDATA = [
    ["Problem_01", problem01, 1, 0, 0, 0, 0]
]

class TestSequence(unittest.TestCase):
    """Passthrough case. Tests added in main."""


def test_generator(i, j):
    """Simple test generator."""

    def test(self):
        self.assertEqual(i, j)
    return test


if __name__ == '__main__':
    for t in TESTDATA:
        setattr(TestSequence, 'test_%s' % t[0] + "_A1",
                test_generator(t[1](inputfile=("0" + str(t[2]))[-2:]+".test",
                                    part=1), t[3]))
        setattr(TestSequence, 'test_%s' % t[0] + "_A2",
                test_generator(t[1](inputfile=("0" + str(t[2]))[-2:]+".test",
                                    part=2), t[4]))
        setattr(TestSequence, 'test_%s' % t[0] + "_B1",
                test_generator(t[1](inputfile=("0" + str(t[2]))[-2:]+".input",
                                    part=1), t[5]))
        setattr(TestSequence, 'test_%s' % t[0] + "_B2",
                test_generator(t[1](inputfile=("0" + str(t[2]))[-2:]+".input",
                                    part=2), t[6]))
    unittest.main(verbosity=2)