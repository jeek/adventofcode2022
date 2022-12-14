"""Advent of Code 2022."""

import fractions
import unittest
import string
from collections import defaultdict
from copy import copy
from itertools import product, permutations, islice, repeat, tee, combinations
from heapq import heappop, heappush
import re
from functools import reduce, cmp_to_key
import math as e
from math import log as ln, e as e
#import z3
import networkx as nx

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
            while ii + 1 < len(data) and int(data[i][j]) > int(data[ii][jj]):
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

def problem09(inputfile="09.input", part=1):
    """Problem #9."""
    rope, head, seen = [2, 10][part-1], [[0, 0] for i in range([2, 10][part-1]+1)], set()
    for i, j in [k.split(" ") for k in open(inputfile).read().split("\n")]:
        for _ in range(int(j)):
            head[-1] = {"U":[head[0][0]-2, head[0][1]],"D":[head[0][0]+2, head[0][1]],"L":[head[0][0], head[0][1]-2],"R":[head[0][0],head[0][1]+2]}[i]
            for k in range(rope):
                if ((abs(head[k-1][0]-head[k][0])>1 or abs(head[k-1][1]-head[k][1])>1) and (head[k-1][0]!=head[k][0]) and (head[k-1][1]!=head[k][1])): head[k][0], head[k][1] = head[k][0]+1 if head[k-1][0] > head[k][0] else head[k][0]-1,head[k][1]+1 if head[k-1][1] > head[k][1] else head[k][1]-1
                if (abs(head[k-1][0]-head[k][0])>1): head[k][0] += 1 if head[k-1][0] > head[k][0] else -1
                if (abs(head[k-1][1]-head[k][1])>1): head[k][1] += 1 if head[k-1][1] > head[k][1] else -1
            seen.add(tuple(head[rope-1]))
    return len(seen)

def problem10(inputfile="10.input", part=1):
    """Problem #10."""
    data = [i for i in open(inputfile).read().split("\n")] + [[],[]]
    queue = [[] for i in range(2000)]
    signals = []
    totals = [0,1]
    total = 1
    z = 1
    i = 0
    zz = 19.9
    while i < len(data):
        if data[i] == "noop":
            totals.append(totals[-1])
        if data[i][:4] == "addx":
            totals.append(totals[-1])
            totals.append(totals[-1] + int(data[i][5:]))
        i += 1
 #   print(totals[19:22])
 #   print(totals)
    signals = [totals[i]*i for i in range(len(totals)) if (i + 20)%40==0]
    if part == 1:
        return(sum(signals))
    data = [i for i in open(inputfile).read().split("\n")] + [[],[]]
    queue = [[] for i in range(2000)]
    signals = []
    totals = [0,1]
    total = 1
    z = 1
    i = 0
    zz = 19.9
    while i < len(data):
        if data[i] == "noop":
            totals.append(totals[-1])
        if data[i][:4] == "addx":
            totals.append(totals[-1])
            totals.append(totals[-1] + int(data[i][5:]))
        i += 1
 #   print(totals[19:22])
 #   print(totals)
    signals = [totals[i]*i for i in range(len(totals)) if (i + 20)%40==0]
 #   print(totals)
    screen = [[" " for i in range(40)] for j in range(6)]
#    totals.pop(0)
#    totals.pop(0)
    for i in range(240):
        a, b, c = (i // 3 * 3)%40, (i // 3 * 3 + 1)%40, (i // 3 * 3 + 2)%40
        a,b,c = i%40,i%40-2,i%40-1
#        print(i, a, b, c, totals[i])
        try:
            if totals[i] in [a,b,c]: #]+([totals[i+1]] if i < 39 else [totals[i]]))+([totals[i+2]] if i+1 < 39 else [totals[i]]):
                screen[i//40][i%40]="X"
        except:
            pass
    if inputfile == "10.input":
#        print(screen)
        for i in screen:
            pass
#            print("".join(i))
    return None

def problem11(inputfile="11.input", part=1):
    """Problem #11."""
    m,r=[[[int(i)for i in eval("["+l[1][18:]+"]")],l[2][19:].split(" "),int(l[3][21:]),int(l[4][29:]),int(l[5][30:]),0] for l in[i.split("\n")for i in open(inputfile).read().split("\n\n")]],range
    for _,i in product(r(int(5e6*e**(-ln(2.5e5)/part))),r(len(m))):
        m[i][5],m[i][0]=(m[i][5]+len(m[i][0])),[(((m[i][0][j]if m[i][1][0]=="old"else int(m[i][1][0]))+(m[i][0][j]if m[i][1][2]=="old"else int(m[i][1][2]))if m[i][1][1]=="+"else(m[i][0][j]if m[i][1][0]=="old"else int(m[i][1][0]))*(m[i][0][j]if m[i][1][2]=="old"else int(m[i][1][2])))//(5-2*part))%reduce(lambda x,y:x*y,[k[2]for k in m])for j in r(len(m[i][0]))]
        [m[m[i][4 if m[i][0][0]%m[i][2]else 3]][0].append(m[i][0].pop(0))for _ in r(len(m[i][0]))]
    return reduce(lambda a,b:a*b,sorted([i[5]for i in m])[-2:])

def problem12(inputfile="12.input", part=1):
    """Problem #12."""
    map, start, goal, best = [[ord(i)-ord("a") for i in j] for j in open(inputfile).read().split("\n")], ord("S") - ord("a"), ord("E") - ord("a"), 10**99
    starti, goali, width, height = [(start in i) for i in map].index(True), [(goal in i) for i in map].index(True), len(map[0]), len(map)
    startj, goalj, dist = map[starti].index(start), map[goali].index(goal), [[10**99 for j in range(width)] for i in range(height)]
    map[goali][goalj], map[starti][startj], dist[goali][goalj]=ord("z")-ord("a"), 0, 0
    for _, i, j in product(range(height * width), range(height), range(width)): dist[max(0, i-1)][j],dist[min(height-1,i+1)][j],dist[i][max(0,j-1)],dist[i][min(width-1,j+1)] = min(dist[i][j]+1, dist[max(0, i-1)][j]) if i > 0 and map[i][j] - 1 <= map[max(0, i-1)][j] else dist[max(0, i-1)][j], min(dist[i][j]+1, dist[min(height-1,i+1)][j]) if i < height-1 and map[i][j] - 1 <= map[min(height-1,i+1)][j] else dist[min(height-1,i+1)][j], min(dist[i][j]+1, dist[i][max(0,j-1)]) if j > 0 and map[i][j] - 1 <= map[i][max(0,j-1)] else dist[i][max(0,j-1)], min(dist[i][j]+1, dist[i][min(width-1,j+1)]) if j < width-1 and map[i][j] - 1 <= map[i][min(width-1,j+1)] else dist[i][min(width-1,j+1)]
    if part == 1: return dist[starti][startj]
    for i, j in product(range(height), range(width)): best = min(best, dist[i][j]) if map[i][j] == 0 else best
    return best

def problem13(inputfile="13.input", part=1):
    """Problem 13."""
    def compare(left, right, good=0, i=0):
        while good == 0 and i < len(left): good, i = 1 if len(right) <= i else compare(list([left[i]]) if type(left[i]) is int else left[i], list([right[i]]) if type(right[i]) is int else right[i]) if type(left[i]) is list or type(right[i]) is list else (left[i]>right[i])-(left[i]<right[i]), i + 1
        return good if good else -1 if len(left) < len(right) else 0
    return [sum([1 + xx for (xx,yy) in enumerate([compare(*[eval(j) for j in i.split("\n")]) for i in open(inputfile).read().split("\n\n")]) if yy!=1]),sum(map(lambda x: (x.index([2])+1)*(x.index([6])+1), [sorted([eval(i) for i in open(inputfile).read().replace("\n\n","\n").split("\n")]+[[2],[6]], key=cmp_to_key(compare))]))][part-1]

def problem14(inputfile="14.input", part=1):
    """Problem #14."""
    data = open(inputfile).read().split("\n")
    grid = defaultdict(lambda: defaultdict(lambda: 0))
    lowest = 0
    for ii in data:
        line = [[int(x) for x in j.split(",")] for j in ii.split("->")]
        i = 0
        while i + 1 < len(line):
            a, b = line[i][0], line[i][1]
            lowest = max(lowest, b)
            grid[a][b] = 1
            if line[i+1][0] > a:
                while grid[line[i+1][0]][line[i+1][1]] == 0:
                    grid[a][b] = 1
                    a += 1
            else:
                if line[i+1][0] < a:
                    while grid[line[i+1][0]][line[i+1][1]] == 0:
                        grid[a][b] = 1
                        a -= 1
                else:
                    if line[i+1][1] > b:
                        while grid[line[i+1][0]][line[i+1][1]] == 0:
                            grid[a][b] = 1
                            b += 1
                    else:
                        while grid[line[i+1][0]][line[i+1][1]] == 0:
                            grid[a][b] = 1
                            b -= 1
            i += 1
        lowest = max(lowest, b)
    aa = min(grid.keys())
    bb = max(grid.keys())
    done = False
    count = 0
    sand = [500,0]
    while not done:
        if sand[1] - 1 > lowest:
            done = True
        if not done and grid[sand[0]][sand[1]+1] == 0:
            sand = [sand[0],sand[1]+1]
        else:
            if not done and grid[sand[0]-1][sand[1]+1] == 0:
                sand = [sand[0]-1,sand[1]+1]
            else:
                if not done and grid[sand[0]+1][sand[1]+1] == 0:
                    sand = [sand[0]+1,sand[1]+1]
                else:
                    if sand[1] - 1> lowest:
                        done = True
                    else:
                        grid[sand[0]][sand[1]] = 2
                        sand = [500, 0]
                        count += 1
    count2 = 0
    for i in grid:
        for j in grid[i]:
            if grid[i][j] == 2:
                count2 += 1
    if part == 1:
        return count
    for z in range(aa - lowest - lowest, bb + lowest + lowest):
        grid[z][lowest + 2] = 1
    sand = [500,0]
    while grid[500][0] == 0:
        done = False
        if not done and grid[sand[0]][sand[1]+1] == 0:
            sand = [sand[0],sand[1]+1]
        else:
            if not done and grid[sand[0]-1][sand[1]+1] == 0:
                sand = [sand[0]-1,sand[1]+1]
            else:
                if not done and grid[sand[0]+1][sand[1]+1] == 0:
                    sand = [sand[0]+1,sand[1]+1]
                else:
                        grid[sand[0]][sand[1]] = 2
                        sand = [500, 0]
                        count += 1
    return count

def problem15(inputfile="15.input", part=1):
    """Problem #15."""
    data = open(inputfile).read().split("\n")
    goal = 0
    if inputfile=="15.test":
        goal = 10
    else:
        goal = 2000000
    if part == 2:
        gmax = goal * 2
        x = z3.Int('x')
        y = z3.Int('y')
        s = z3.Solver()
        s.add(x >= 0)
        s.add(x <= gmax)
        s.add(y >= 0)
        s.add(y <= gmax)
    grid = defaultdict(lambda: defaultdict(lambda: "."))
    for i in data:
        words = i.split(" ")
        a,b,c,d = words[2], words[3], words[8], words[9]
        a = int(a.replace(",","").replace(":","").split("=")[1])
        b = int(b.replace(",","").replace(":","").split("=")[1])
        c = int(c.replace(",","").replace(":","").split("=")[1])
        d = int(d.replace(",","").replace(":","").split("=")[1])
        dist = (abs(c-a)+abs(d-b))
        if part == 1:
            for jj in range(a-dist, a+dist+1):
                ii = goal
                if (abs(ii-b)+abs(jj-a)) <= dist:
                    if grid != "B":
                        grid[ii][jj] = "#"
            grid[d][c] = "B"
            grid[b][a] = "S"
        if part == 2:
            s.add(z3.Abs(x-a)+z3.Abs(y-b)>dist)
    if part==1:
        return len([k for k in grid[ii].values() if k=="#"])
    s.check()
    m = s.model()
    return m[x].as_long()*4000000+m[y].as_long()

def problem16(inputfile="16.input", part=1):
    """Problem #16."""
    data, valves, g = open(inputfile).read().split("\n"), {}, nx.Graph()
    for i in data:
        valves[i.split(" ")[1]] = {"flow": int(i.split(" ")[4].split("=")[1].replace(";","")), "valves": [j.replace(" ","") for j in i[i.index("valve") + 6:].split(", ")]}
        for j in valves[i.split(" ")[1]]["valves"]: g.add_edge(i.split(" ")[1], j)
    useful, best, queues, bestd, i = [i for i in valves if valves[i]["flow"] > 0], 0, [["AA"]], defaultdict(int), 0
    while i < len(queues):
        current, time, total, j, opened = queues[i], [30,26][part-1], 0, 0, set()
        while time >= 0:
            total += sum(valves[k]["flow"] for k in opened)
            if j < len(current):
                if current[j] == "Open": opened.add(current[j-1])
                j += 1
            time -= 1
        bestd[tuple(sorted(opened))], latest = max(bestd[tuple(sorted(opened))], total), -1
        while current[latest] == "Open": latest -= 1
        opened, k = set(), 0
        while k < len(current):
            if current[k] == "Open" and current[k-1] != "Open": opened.add(current[k-1])
            k += 1
        for j in [k for k in useful if k not in opened]:
            temp = (current + nx.shortest_path(g, current[latest], j)[1:] + ["Open"])[:32-4*part]
            if (temp != current) and temp[-1] == "Open": queues.append(temp)
        i += 1
    if part == 1: return max(bestd.values())
    for i, j in product(bestd, bestd):
        if set(i)&set(j)==set():
            best = max(best, bestd[i] + bestd[j])
    return best

def problem17(inputfile="17.input", part=1):
    """Problem #17."""
    jets = [i for i in open(inputfile).read()]
   # print(jets)
    grid = [[0,0,0,0,0,0,0] for j in range(10000000)]
    rocks = [i[::-1] for i in [[[1,1,1,1]],[[0,1,0],[1,1,1],[0,1,0]],[[0,0,1],[0,0,1],[1,1,1]],[[1],[1],[1],[1]],[[1,1],[1,1]]]]
    oh = []
    k = 0
    jj = 0
    highest = 0
    while k < 2022 * part:
        pos = [3 + highest, 2]
        highest = max(0, highest)
        rock = rocks[k%5]
        rw, rh = len(rock[0]), len(rock)
        goodtomoveup = True
        while goodtomoveup == True:
            if (jets[jj] == "<" and pos[1] > 0) or (jets[jj] == ">" and pos[1]+rw < 7):
                goodtomoveside = True
                try:
                  for i in range(rh):
                    for j in range(rw):
                        if grid[pos[0]+i][pos[1] + (1 if jets[jj] == ">" else -1) +j] == 1 and rock[i][j] == 1:
                            goodtomoveside = False
                except:
                    goodtomoveside = False
                pos[1] = max(pos[1], 0)
                pos[1] = min(pos[1], 7 - rw)
                if goodtomoveside == True:
                    pos[1] = pos[1] + (-1 if jets[jj] == "<" else 1)
                pos[1] = min(pos[1], 7 - rw)
                pos[1] = max(pos[1], 0)
            jj = (jj + 1) % len(jets)
            for i in range(rh):
                for j in range(rw):
                    try:
                        if grid[pos[0]-1+i][pos[1]+j] == 1 and rock[i][j] == 1:
                            goodtomoveup = False
                    except:
                        goodtomoveup = False
            if pos[0] <= 0:
                pos[0] = 0
                goodtomoveup = False
            if goodtomoveup == True:
                pos[0] -= 1
        for i in range(rh):
            for j in range(rw):
                if rock[i][j]:
                    grid[pos[0] + i][pos[1]+j] = 1
        old = highest
        while grid[highest].count(1) > 0:
            highest += 1
        oh.append(highest - old)
        k += 1
    if part == 1:
        return highest
    done = False
    z = 15
    while not done:
        done = True    
        for zz in range(1, z+1):
            if oh[len(oh)-zz] != oh[len(oh)-zz-z]:
                done = False
        z += 1
    z -= 1
    a = len(oh) - 1
    while a % z != 1000000000000 % z:
        a -= 1
    x1, x2 = a - z, a
    y1, y2 = sum(oh[:x1]), sum(oh[:x2])
    m = fractions.Fraction(y2-y1,x2-x1)
    b = 0
    while m * x1 + fractions.Fraction(b,x2-x1) > y1:
        b -= 1
    while m * x1 + fractions.Fraction(b,x2-x1) < y1:
        b += 1
    print(b, x2-x1)
    return m*1000000000000+fractions.Fraction(b, x2-x1)

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
    ["Problem_09", problem09, 9, 13, 1, 5513, 2427],
    ["Problem_10", problem10, 10, 13140, None, 14820, None],
    ["Problem_11", problem11, 11, 10605, 2713310158, 56595, 15693274740],
    ["Problem_12", problem12, 12, 31, 29, 456, 454],
    ["Problem_13", problem13, 13, 13, 140, 5675, 20383],
    ["Problem_14", problem14, 14, 24, 93, 1199, 23925],
    ["Problem_15", problem15, 15, 26, 56000011, 4725496, 12051287042458],
    ["Problem_16", problem16, 16, 1651, 1707, 1820, 2602],
    ["Problem_17", problem17, 17, 3068, 1514285714288, 3206, 1602881844347],
][-1:]

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