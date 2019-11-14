class Permutation:
    def __init__(self, a):
        if isinstance(a,list):
            res = []
            for i in range(len(a)):
                res.append(a[i]-1)
            self.a = res
        else:
            self.a = a
        self.n = len(self.a)
    
    def copy(self):
        return Permutation([i+1 for i in self.a])
    
    def __mul__(self, other):
        assert isinstance(other, Permutation)
        assert len(other.a) == len(self.a)
        return Permutation([self[other[i]] for i in range(1,self.n+1)])
    
    def  __pow__(self, other):
        if other == 1:
            return self.copy()
        elif other > 0:
            t = (self**(other//2))
            if other % 2:
                return self*(t*t)
            else:
                return t*t
        elif other == -1:
            res = [0 for i in range(self.n)]
            for i in range(self.n):
                res[self.a[i]] = i+1
            return Permutation(res)
    
    def __getitem__(self, index):
        return self.a[index-1]+1
    
    def __repr__(self):
        return " ".join(str(i+1) for i in range(self.n)) + "\n" + " ".join(str(self.a[i]+1) for i in range(self.n))
    
    def evnss(self):
        res = 0
        for j in range(self.n):
            for i in range(j-1):
                if self.a[i]>self.a[j]:
                    res += 1
        return res%2

p1 = Permutation([3, 4, 8, 7, 2, 1, 5, 6])
p2 = Permutation([6, 4, 3, 7, 1, 8, 5, 2])
p3 = Permutation([5, 6, 7, 2, 3, 1, 4, 8])
p4 = (p1**15)*(p2**(-1))
p4 = p4**185
x = (p4**(-1))*p3

print("2 задание:\nx = ")
print(x)
print()

print("3 задание:")
p = Permutation(list(range(6,186))+list(range(1,6)))
pe = p.evnss()
print(pe)
if pe == 0:
    print("чётная")
else:
    print("нечётная")
print()

from LinAlgFuncs import *
from mySybolicCalcs import *

x = Variable("x")

m = tensor_from_iterable([
    [0,x,0,6,0,0],
    [4,0,9,7,0,1],
    [x,9,x,x,0,7],
    [0,9,3,0,8,2],
    [0,x,x,4,7,8],
    [0,5,4,4,8,7]
])

print("задание 4:\ndet(m) =", det_by_minors(m))