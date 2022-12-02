"""Advent of Code 2022."""

import unittest
from collections import defaultdict
from copy import copy
from itertools import product, permutations
from heapq import heappop, heappush

def problem01(inputfile="01.input", part=1):
    """Problem #1."""
    return sum(sorted(list(map(lambda x: sum(int(i) for i in x.split("\n")), open(inputfile).read().split("\n\n"))))[-(part*2-1):])

def problem02(inputfile="02.input", part=1):
    """Problem #2."""
    score = 0
    for line in [i for i in open(inputfile).read().split("\n")]:
        them, me = line.split(" ")
        if part == 2:
            me = ["ZXY","XYZ","YZX"]["XYZ".index(me)]["ABC".index(them)]
        score += " XYZ".index(me) + (6 if "ABC".index(them)=="YZX".index(me) else 3 if "ABC".index(them)=="XYZ".index(me) else 0)
    return score

def problem02a(inputfile="02.input", part=1):
    """Problem #2, alternate solution."""
    return sum([["B X","C Y","A Z","A X","B Y","C Z","C X","A Y","B Z"],["B X","C X","A X","A Y","B Y","C Y","C Z","A Z","B Z"]][part-1].index(line)+1 for line in open(inputfile).read().split("\n"))

def problem02b(inputfile="02.input", part=1):
    """Problem #2, alternate solution."""
    return sum([[9,1,5,6,7,2,3,4,8],[9,1,5,7,2,6,8,3,4]][part-1][(ord(line[0])%3)*3+ord(line[2])%3] for line in open(inputfile).read().split("\n"))

def problem02c(inputfile="02.input", part=1):
    """Problem #2, alternate solution."""
    return sum([69,420,27,7,41,42,22,19,33,8,46,45,14,35,21,28,4,44,18,36].index(part*ord(line[0])*ord(line[2])%47)//2 for line in open(inputfile).read().split("\n"))

def problem02d(inputfile="02.input", part=1):
    """Problem #2, alternate solution."""
    return sum([*map(lambda line:[*map(lambda x:ord(x)%47,'OZJ6XYEBP7]\=RDK3[AS')].index(part*ord(line[0])*ord(line[2])%47)//2,open(inputfile).read().splitlines())])

TESTDATA = [
    ["Problem_01", problem01, 1, 24000, 45000, 68802, 205370],
    ["Problem_02", problem02, 2, 15, 12, 11150, 8295],
    ["Problem_02a", problem02a, 2, 15, 12, 11150, 8295],
    ["Problem_02b", problem02b, 2, 15, 12, 11150, 8295],
    ["Problem_02c", problem02c, 2, 15, 12, 11150, 8295],
    ["Problem_02d", problem02d, 2, 15, 12, 11150, 8295],
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