"""Advent of Code 2022."""

import unittest
import string
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

def problem02e(inputfile="02.input", part=1):
    """Problem #2, alternate solution."""
    return sum(1+("BXCXAXAYBYCYCZAZBZBXCYAZAXBYCZCXAYBZ".find(line[0]+line[2],(part%2)*18)%18)//2 for line in open(inputfile).read().split("\n"))

def problem02f(inputfile="02.input", part=1):
    """Problem #2, alternate solution, based on Shiiyu's."""
    return sum([*map(lambda x:(((4+x[1]-x[0])%3*3)if part==1 else((2+x[0]+x[1])%3+x[1]*2))+x[1]+1,[*map(lambda line:[(ord(line[0])+4)%23,(ord(line[2])+4)%23],open(inputfile).read().splitlines())])])

def problem03(inputfile="03.input", part=1):
    """Problem #3."""
    data = open(inputfile).read().split("\n")
    total = 0
    if part == 1:
        for i in data:
            for j in set(i[:len(i)//2]) & set(i[len(i)//2:]):
                total += ("0"+string.ascii_letters).index(j)
        return total
    i = 0
    while i < len(data):
        for j in (set(data[i]) & set(data[i+1]) & set(data[i+2])):
            total += ("0"+string.ascii_letters).index(j)
        i += 3
    return total

def problem03a(inputfile="03.input", part=1):
    """Problem #3, alternate solution."""
    return (sum([*map(lambda x:("0"+string.ascii_letters).index(list(x)[0]),[*map(lambda line:set(line[:len(line)//2])&set(line[len(line)//2:]),open(inputfile).read().split("\n"))])]))if part==1 else(sum([*map(lambda data:sum([("0"+string.ascii_letters).index(list(set(data[i-i%3])&set(data[i-i%3+1])&set(data[i-i%3+2]))[0])for i in range(len(data))])//3,[open(inputfile).read().split("\n")])]))

def problem03b(inputfile="03.input", part=1):
    """Problem #3, alternate solution."""
    data = open(inputfile).read().split("\n")
    total = 0
    if part == 1:
        for i in data:
            for j in set(i[:len(i)//2]) & set(i[len(i)//2:]):
                total += (ord(j)-96)%58
        return total
    i = 0
    while i < len(data):
        for j in (set(data[i]) & set(data[i+1]) & set(data[i+2])):
            total += (ord(j)-96)%58
        i += 3
    return total

def problem03c(inputfile="03.input", part=1):
    """Problem #3, alternate solution."""
    return (sum([*map(lambda x:(ord(list(x)[0])-96)%58,[*map(lambda line:set(line[:len(line)//2])&set(line[len(line)//2:]),open(inputfile).read().split("\n"))])]))if part==1 else(sum([*map(lambda data:sum([(ord(list(set(data[i-i%3])&set(data[i-i%3+1])&set(data[i-i%3+2]))[0])-96)%58 for i in range(len(data))])//3,[open(inputfile).read().split("\n")])]))

def problem03d(inputfile="03.input", part=1):
    """Problem #3, alternate solution."""
    return sum([((ord(list(j)[0])-96)%58) for j in([*map(lambda x:list(map(lambda y:((list(y)[0])),list(x))),list([*map(lambda data:list([0,[set(data[i][:len(data[i])//2])&set(data[i][len(data[i])//2:])],[set(data[i-i%3])&set(data[i-i%3+1])&set(data[i-i%3+2])]][part] for i in range(len(data))),[open(inputfile).read().split("\n")])]))][0])])//[0,1,3][part]

TESTDATA = [
    ["Problem_01", problem01, 1, 24000, 45000, 68802, 205370],
    ["Problem_02", problem02, 2, 15, 12, 11150, 8295],
    ["Problem_02a", problem02a, 2, 15, 12, 11150, 8295],
    ["Problem_02b", problem02b, 2, 15, 12, 11150, 8295],
    ["Problem_02c", problem02c, 2, 15, 12, 11150, 8295],
    ["Problem_02d", problem02d, 2, 15, 12, 11150, 8295],
    ["Problem_02e", problem02e, 2, 15, 12, 11150, 8295],
    ["Problem_02f", problem02f, 2, 15, 12, 11150, 8295],
    ["Problem_03", problem03, 3, 157, 70, 7980, 2881],
    ["Problem_03a", problem03a, 3, 157, 70, 7980, 2881],
    ["Problem_03b", problem03b, 3, 157, 70, 7980, 2881],
    ["Problem_03c", problem03c, 3, 157, 70, 7980, 2881],
    ["Problem_03d", problem03d, 3, 157, 70, 7980, 2881],
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