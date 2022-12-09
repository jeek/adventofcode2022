"""Advent of Code 2022."""

import unittest
import string
from collections import defaultdict
from copy import copy
from itertools import product, permutations, islice, repeat, tee, combinations
from heapq import heappop, heappush
import re
from functools import reduce

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

def problem04(inputfile="04.input", part=1):
    """Problem #4."""
    data = open(inputfile).read().split("\n")
    total = [0,0]
    for i in range(0,len(data)):
        left, right = data[i].split(",")
        left = set(range(int(left.split("-")[0]), 1 + int(left.split("-")[1])))
        right = set(range(int(right.split("-")[0]), 1 + int(right.split("-")[1])))
        if left | right in [left, right]:
            total[0] += 1
        if len(left & right):
            total[1] += 1
    return total[part-1]

def problem04a(inputfile="04.input", part=1):
    """Problem #4, alternate solution."""
    return([*map(lambda x:(x[0]|x[1]in[x[0],x[1]],len(x[0]&x[1])>0)[part-1],[*map(lambda x:[set(range(int(x[0][0]),int(x[0][1])+1)),set(range(int(x[1][0]),int(x[1][1])+1))],[*map(lambda x:[j.split("-")for j in x.split(",")],open(inputfile).read().split("\n"))])])].count(True))

def problem04b(inputfile="04.input", part=1):
    """Problem #4, alternate solution."""
    return [*map(lambda x:[x[0][1]>=x[1][0],(x[0][0]<=x[1][0] and x[0][1]>=x[1][1])or(x[0][0]>=x[1][0] and x[0][1]<=x[1][1])][part%2],[*map(lambda x:sorted([[int(j) for j in i.split("-")] for i in x.split(",")],key=lambda y:y[0]),open(inputfile).read().split("\n"))])].count(True)

def problem04c(inputfile="04.input", part=1):
    """Problem #4, alternate solution."""
    return [[y[0][1]>=y[1][0],(y[0][0]<=y[1][0] and y[0][1]>=y[1][1])or(y[0][0]>=y[1][0] and y[0][1]<=y[1][1])][part%2] for y in [sorted([[int(x[0][0]),int(x[0][1])],[int(x[1][0]),int(x[1][1])]]) for x in [(j[0].split("-"),j[1].split("-")) for j in (i.split(",") for i in open(inputfile).read().split())]]].count(True)

def problem04d(inputfile="04.input", part=1):
    """Problem #4, alternate solution."""
    return [[(c>b)|(a>d),((c<a)|(b<d))&((a<c)|(d<b))][part%2]for a,b,c,d in[[int(k) for k in re.split(r"[,-]",i)]for i in open(inputfile).read().split()]].count(False)

def problem05(inputfile="05.input", part=1):
    """Problem #5."""
    data1, data2 = open(inputfile).read().split("\n\n")
    columns = [[] for i in range(round(len(data1)/4))]
    for k in data1.split("\n"):
        for j in range(0, len(k), 4):
            if "[" in k[j:j+4]:
                columns[j//4].insert(0,k[j+1])
    for line in data2.split("\n"):
        _, i, _, j, _, k = line.split(" ")
        i, j, k = (int(m) for m in [i, j, k])
        if part == 1:
            for l in range(i):
                columns[k-1].append(columns[j-1].pop())
        else:
            temp = []
            for l in range(i):
                temp.append(columns[j-1].pop())
            for l in range(i):
                columns[k-1].append(temp.pop())
    return "".join((i[-1] if len(i)>0 else "") for i in columns)

def problem05a(inputfile="05.input", part=1):
    """Problem #5, alternate solution."""
    return "".join([i[-1] for i in reduce(lambda x, y: [(x[i] if (int(y.split(" ")[3])-1 != i and int(y.split(" ")[5])-1 != i) else x[i] + x[int(y.split(" ")[3])-1][::-1][:int(y.split(" ")[1])][::-1 if part == 2 else 1] if (int(y.split(" ")[3])-1) != i and (int(y.split(" ")[5])-1) == i else x[i][:-int(y.split(" ")[1])]) for i in range(len(x))], [[list(filter(lambda x: x != " ", i[1:])) for i in list(list(i) for i in list(list(x)[::-1] for x in zip(*[list(i) for i in open(inputfile).read().split("\n\n")[0].split("\n")]))) if list(list(i) for i in list(list(x)[::-1] for x in zip(*[list(i) for i in open(inputfile).read().split("\n\n")[0].split("\n")]))).index(i)%4==1]] + open(inputfile).read().split("\n\n")[1].split("\n"))])

def problem06(inputfile="06.input", part=1):
    """Problem #6."""
    return list((part*10-6 + min([i for i in range(len(data)-3) if part*10-6==len(set(data[i:i+part*10-6]))])) for data in open(inputfile).read().split("\n"))

def nwise(iterator, n):
    answer = tee(iterator, n)
    for i in range(n):
        for j in range(i):
            next(answer[i], None)
    return zip(*answer)

def problem06a(inputfile="06.input", part=1):
    """Problem #6, alternate solution."""
    return [list(map(lambda x: len(x)==len(set(x)), list(nwise(data, [4,14][part-1])))).index(True) + [4,14][part-1] for data in open(inputfile).read().split("\n")]

def problem06b(inputfile="06.input", part=1):
    """Problem #6, alternate solution."""
    answer = []
    datalines = open(inputfile).read().split("\n")
    for data in datalines:
        i = 0
        while i < len(data):
            good = True
            j = i
            while good and j < i + [4,14][part-1]:
                k = j + 1
                while good and k < i + [4,14][part-1]:
                    if data[j] == data[k]:
                        good = False
                    k += 1
                j += 1
            if good:
                answer.append(i + [4,14][part-1])
                i = len(data)
            i += 1
    return answer

def problem06c(inputfile="06.input", part=1):
    """Problem #6, alternate solution."""
    datalines = open(inputfile).read().split("\n")
    z = 0
    answer = []
    for data in datalines:
        z += 1
        for i in range(len(data)):
            if len(list(j for j in combinations(data[i:i+[4,14][part-1]], 2) if j[0] == j[1])) == 0:
                if len(answer) < z:
                    answer.append(i + [4,14][part-1])
            i += 1
    return answer

def problem07(inputfile="07.input", part=1):
    """Problem #7."""
    data = open(inputfile).read().split("\n")
    path = ""
    files = {}
    dirs = {}
    total = 0
    for i in data:
        if i[:3] != "dir":
            if i[:4] == "$ cd":
                if i == "$ cd /":
                    path = "/"
                else:
                    if i == "$ cd ..":
                        path = "/".join(path.split("/")[:-1]).replace("//", "/")
                    else:
                        path = (path + "/" + i[5:]).replace("//", "/")
            else:
                if i != "$ ls":
                    a, b = i.split(" ")
                    files[path + b] = int(a)
                    temp = path
                    if path not in dirs:
                        dirs[path] = 0
                    dirs["/"] += int(a)
                    if path != "/":
                        while len(temp) > 1:
                            if temp not in dirs:
                                dirs[temp] = 0
                            dirs[temp] += int(a)
                            temp = temp.split("/")
                            temp.pop()
                            temp = ("/".join(temp)).replace("//", "/")
    for i in dirs:
        if dirs[i] < 100000:
            total += dirs[i]
    if part==1:
        return total
    totaldisk = 70000000 - dirs["/"]
    neededdisk = 30000000
    return min(i for i in dirs.values() if totaldisk + i > neededdisk)

def problem07a(inputfile="07.input", part=1):
    """Problem #7, alternate solution."""
    stack = [[]]
    answers = []
    ind = 1
    for i in open(inputfile).read().split("\n"):
        if i[:4] == "$ cd":
            if i != "$ cd ..":
                stack.append([])
            else:
                answers.append(sum(stack[-1]))
                stack[-1] = stack[-2] + [sum(stack.pop())]
        else:
            if i[:3] != "dir" and i[0] != "$":
                stack[-1].append(int(i.split(" ")[0]))
    while len(stack) > 1:
        answers.append(sum(stack[-1]))
        stack[-1] = stack[-2] + [sum(stack.pop())]
    answers = answers + [sum(stack[0])]
    if part == 1:
        return sum(i for i in answers if i < 100000)
    return min(i for i in answers if 70000000 - max(answers) + i > 30000000)

#def problem07b(inputfile="07.input", part=1):
#    """Problem #7, alternate solution."""
#    print(reduce(lambda x, y: [x[0] + [[]], x[1]] if y[:min(4, len(y))] == "$ cd" and y != "$ cd .." else [x[0][:-1] + [x[0][-1] + [int(y.split(" ")[0])]], x[1]] if re.match(r"\d+ \S+", y) else x if y[:3] == "dir" else [x[0][:-2] + sum(x[0][-1]), x[1] + [sum(x[0][-1])]], [[[[0]],[]]] + open(inputfile).read().split("\n")))
#    return reduce(lambda x, y: [x[0] + [[]], x[1]] if y[:min(4, len(y))] == "$ cd" and y != "$ cd .." else [x[0][:-1] + [x[0][-1] + [int(y.split(" ")[0])]], x[1]] if re.match(r"\d+ \S+", y) else x if y[:3] == "dir" else [x[0][:-2] + sum(x[0][-1]), x[1] + [sum(x[0][-1])]], [[[[0]],[]]] + open(inputfile).read().split("\n"))

def problem08(inputfile="08.input", part=1):
    """Problem #8."""
    data = open(inputfile).read().split("\n")
    visible = []
    for i in data:
        visible.append([])
        for j in i:
            visible[-1].append(0)
    total = 0
    if part == 1:
        for i in range(len(data)):
            temp = list(int(j) for j in data[i])
            best = temp[0]
            visible[i][0] = 1
            for j in range(len(data[i])):
                if temp[j] > best:
                    visible[i][j] = 1
                    best = temp[j]
            temp = list(int(j) for j in data[i])[::-1]
            best = temp[0]
            visible[i][-1] = 1
            for j in range(len(data[i])):
                if temp[j] > best:
                    visible[i][len(data[i])-1-j] = 1
                    best = temp[j]
        data = [[data[j][i] for j in range(len(data))] for i in range(len(data[0])-1,-1,-1)]
        visible = [[visible[j][i] for j in range(len(visible))] for i in range(len(visible[0])-1,-1,-1)]
        for i in range(len(data)):
            temp = list(int(j) for j in data[i])
            best = temp[0]
            visible[i][0] = 1
            for j in range(len(data[i])):
                if temp[j] > best:
                    visible[i][j] = 1
                    best = temp[j]
            temp = list(int(j) for j in data[i])[::-1]
            best = temp[0]
            visible[i][-1] = 1
            for j in range(len(data[i])):
                if temp[j] > best:
                    visible[i][len(data[i])-1-j] = 1
                    best = temp[j]
        return sum(sum(j) for j in visible)
    visible = []
    for i in data:
        visible.append([])
        for j in i:
            visible[-1].append(0)
    best = 0
    for i in range(len(data)):
        for j in range(len(data[i])):
            a, b, c, d = (1 if i != 0 else 0),(1 if i != len(data)-1 else 0),(1 if j != 0 else 0),(1 if j != len(data[i])-1 else 0)
            ii, jj = i-1, j
            while ii > 0 and int(data[i][j]) > int(data[ii][jj]):
                ii -= 1
                a += 1
            ii, jj = i+1, j
            while ii + 1< len(data) and int(data[i][j]) > int(data[ii][jj]):
                ii += 1
                b += 1
            ii, jj = i, j-1
            while jj > 0 and int(data[i][j]) > int(data[ii][jj]):
                jj -= 1
                c += 1
            ii, jj = i, j+1
            while jj + 1 < len(data) and int(data[i][j]) > int(data[ii][jj]):
                jj += 1
                d += 1
            best = max(a * b * c * d, best)
    return best

def problem08a(inputfile="08.input", part=1):
    """Problem #8, alternate solution."""
    data = [[int(i) for i in j] for j in open(inputfile).read().split("\n")]
    visible, finish, best = [[(part-1) for j in range(len(data[0]))] for i in range(len(data))], [sum,max][part-1], 0
    for _, i, j in product(range(len("data")), range(len(data)), range(len(data[0]))):
        if part == 2:
            l, m = i != 0, i - 1
            while m > 0 and data[i][j] > data[m][j]: m, l = m - 1, l + 1
            visible[i][j] *= l
        else: best, visible[i][j] = [best, data[i][j]][j == 0 or data[i][j] > best], data[i][j] > best or j == 0 or visible[i][j]
        if i+1==len(data) and j+1==len(data[0]): data, visible = [[data[j][i] for j in range(len(data))] for i in range(len(data[0])-1,-1,-1)], [[visible[j][i] for j in range(len(visible))] for i in range(len(visible[0])-1,-1,-1)]
    return finish(finish(j) for j in visible)

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
    ["Problem_04", problem04, 4, 2, 4, 602, 891],
    ["Problem_04a", problem04a, 4, 2, 4, 602, 891],
    ["Problem_04b", problem04b, 4, 2, 4, 602, 891],
    ["Problem_04c", problem04c, 4, 2, 4, 602, 891],
    ["Problem_04d", problem04d, 4, 2, 4, 602, 891],
    ["Problem_05", problem05, 5, "CMZ", "MCD", "RLFNRTNFB", "MHQTLJRLB"],
    ["Problem_05a", problem05a, 5, "CMZ", "MCD", "RLFNRTNFB", "MHQTLJRLB"],
    ["Problem_06", problem06, 6, [7,5,6,10,11], [19,23,23,29,26], [1480], [2746]],
    ["Problem_06a", problem06a, 6, [7,5,6,10,11], [19,23,23,29,26], [1480], [2746]],
    ["Problem_06b", problem06b, 6, [7,5,6,10,11], [19,23,23,29,26], [1480], [2746]],
    ["Problem_06c", problem06c, 6, [7,5,6,10,11], [19,23,23,29,26], [1480], [2746]],
    ["Problem_07", problem07, 7, 95437, 24933642, 1297159, 3866390],
    ["Problem_07a", problem07a, 7, 95437, 24933642, 1297159, 3866390],
#    ["Problem_07b", problem07b, 7, 95437, 24933642, 1297159, 3866390],
    ["Problem_08", problem08, 8, 21, 8, 1690, 535680],
    ["Problem_08a", problem08a, 8, 21, 8, 1690, 535680],
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