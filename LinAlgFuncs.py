from collections.abc import Iterable
from fractions import Fraction
from random import randint


def random_tensor(shape, minval=-5, maxval=5):
    if len(shape) == 1:
        return Tensor([randint(minval, maxval) for i in range(shape[0])])
    else:
        return Tensor([random_tensor(shape[1:]) for i in range(shape[0])])


def mul(a):
    m = 1
    for i in a:
        m *= i
    return m


def zeros(*shape):
    if len(shape) == 1:
        return Tensor([0 for i in range(shape[0])])
    else:
        return Tensor([zeros(*shape[1:]) for i in range(shape[0])])


def dot_v(a, b):
    if a.shape() == b.shape() and len(a.shape()) == 1:
        return sum([a[i]*b[i] for i in range(len(a))])


def I(n):
    res = zeros(n, n)
    for i in range(n):
        res[i, i] = 1
    return res


def dot_m(a, b):
    assert a.shape()[1] == b.shape()[0], "Matrices must be same size"
    assert len(a.shape()) == 2, "Must be 2D Matrix"
    bt = b.transpose2()
    width = a.shape()[0]
    height = b.shape()[1]
    res = zeros(width, height)
    for i in range(width):
        for j in range(height):
            res[i, j] = dot_v(a[i], bt[j])
    return res


def dot(a, b):
    assert isinstance(a, Tensor) and isinstance(b, Tensor), "Terms must be tensors"
    c = len(a.shape()) - len(b.shape())
    if c == 0:
        if len(a.shape()) == 1:
            return dot_v(a, b)
        else:
            return dot_m(a, b)
    elif c > 0:
        b = Tensor([b])
        if a.shape()[1] == b.shape()[1]:
            b = b.transpose2()
        return dot_m(a, b)
    else:
        a = Tensor([a])
        if a.shape()[0] == b.shape()[0]:
            a = a.transpose2()
        return dot_m(a, b)


class Tensor(list):
    def __getitem__(self, key):
        if type(key) == int:
            return super().__getitem__(key)
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self)
            step = key.step if key.step is not None else 1
            return Tensor([self[i] for i in range(start, stop, step)])
        elif len(key) == 1:
            return self.__getitem__(key[0])
        else:
            return self.__getitem__(key[0]).__getitem__(key[1:])

    def __setitem__(self, key, val):
        if type(key) == int:
            super().__setitem__(key, val)
            return self
        elif isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop if key.stop is not None else len(self)
            step = key.step if key.step is not None else 1
            for i in range(start, stop, step):
                self[i] = val
            return self
        elif len(key) == 1:
            return self.__setitem__(key[0], val)
        else:
            self.__setitem__(key[0], self.
                             __getitem__(key[0]).__setitem__(key[1:], val))
            return self

    def __repr__(self, maxLenItem = None, tab=None):
        maxlitem = maxLenItem if maxLenItem is not None else max([len(str(i)) for i in self.flatten_to_list()])
        tab = tab if tab is not None else 0
        if len(self.shape()) == 1:
            return " "*tab + "["+", ".join([" "*(maxlitem - len(str(i))) + str(i) for i in self]) + "]"
        else:
            c = max(len(self.shape()) - 1, 1)
            res = list()
            first = True
            for i in self:
                if first:
                    res.append(i.__repr__(maxlitem, tab+1))
                    first = False
                else:
                    res.append(i.__repr__(maxlitem, tab+1))
            s = ("," + "\n"*c).join(res)
            s = s.lstrip()
            return " "*tab + "[" + s + "]"

    def __add__(self, a):
        my_shape = self.shape()
        a_shape = a.shape()

        assert a_shape == my_shape, "Second tensor must be same size"

        if len(a_shape) == 1:
            return Tensor([self[i] + a[i] for i in range(a_shape[0])])
        else:
            return Tensor([self[i].__add__(a[i]) for i in range(a_shape[0])])

    def __sub__(self, other):
        return self+-1*other

    def __rsub__(self, other):
        return other+-1*self

    def __mul__(self, m):
        shape = self.shape()
        if len(shape) == 1:
            return Tensor([i * m for i in self])
        else:
            return Tensor([i.__mul__(m) for i in self])

    __rmul__ = __mul__

    def shape(self):
        if type(self[0]) != type(self):
            return [len(self)]
        else:
            return [len(self)] + self[0].shape()

    def copy(self):
        shape = self.shape()
        if len(shape) == 1:
            return Tensor([i for i in self])
        else:
            return Tensor([i.copy() for i in self])

    def transpose2(self):
        shape = self.shape()
        w = shape[0]
        h = shape[1]

        res = zeros(h, w)

        for i in range(h):
            for j in range(w):
                res[i, j] = self[j, i]
        return res

    def reshape(self, new_shape):
        old_shape = self.shape()
        if mul(old_shape) == mul(new_shape):
            raise NotImplementedError()

    def tr2(self):
        res = 0
        dim = min(self.shape())
        for i in range(dim):
            res += self[i, i]
        return res

    def flatten_to_list(self):
        if len(self.shape()) == 1:
            return [i for i in self]
        else:
            res = []	
            for i in self:
                res += i.flatten_to_list()
            return res


def to_triangular(a, use_fractional=False):
    success = True
    permutations = 0
    a = a.copy()
    n = a.shape()[0]

    for i in range(n - 1):
        a0 = a[i, i]

        if a0 == 0:
            success = False
            for j in range(i + 1, n):
                if a[j, i] != 0:
                    success = True
                    permutations += 1
                    a[i], a[j] = a[j], a[i]
                    break
        if success:
            a0 = a[i, i]
        else:
            break

        for j in range(i + 1, n):
            a1 = a[j, i]
            if use_fractional:
                a[j] = a[j] + a[i] * Fraction(-a1, a0)
            else:
                a[j] = a[j] + a[i] * (-a1 / a0)
    return a, permutations, success


def det_triangular(a, s=0):
    n = a.shape()[0]
    assert n == a.shape()[1], "Need square Matrix"
    res = 1
    for i in range(n):
        res *= a[i, i]
    return res * ((-1)**s)


def det(a):
    a, s, success = to_triangular(a)
    if success:
        return det_triangular(a, s)
    else:
        return 0


def cut_ij(mat, i, j):
    n, m = mat.shape()
    res = zeros(n-1, m-1)
    for x in range(n-1):
        for y in range(m-1):
            xm = x
            ym = y
            if x >= i:
                xm += 1
            if y >= j:
                ym += 1
            res[x][y] = mat[xm][ym]
    return res


def det_by_minors(a, j=0):
    n = a.shape()[0]
    assert n == a.shape()[1], "Need square Matrix"
    if n == 1:
        return a[0, 0]
    elif n == 2:
        return a[0, 0]*a[1, 1] - a[1, 0]*a[0, 1]
    elif n == 3:
        return a[0, 0]*a[1, 1]*a[2, 2]+a[0, 1]*a[1, 2]*a[2, 0]+a[0, 2]*a[1, 0]*a[2, 1]-a[0, 2]*a[1, 1]*a[2, 0]\
               - a[0, 1]*a[1, 0]*a[2, 2]-a[0, 0]*a[1, 2]*a[2, 1]
    else:
        res = 0
        for i in range(n):
            res += a[i, j] * ((-1)**(i+j)) * det_by_minors(cut_ij(a, i, j))
        return res


def inverse_Gauss(a, use_fractional=False):
    n = a.shape()[0]
    assert n == a.shape()[1], "Need square Matrix"
    det_a = det(a)
    assert det_a != 0, "Can't invert Matrix with zero determinant"
    e = I(n)
    return solve_Gauss(a, e, use_fractional)


def solve_Gauss(a, b, use_fractional=False):
    n = a.shape()[0]
    assert n == a.shape()[1], "Need square Matrix"
    e = b.copy()
    a = a.copy()
    ##################################
    for i in range(n - 1):
        a0 = a[i, i]
        if a0 == 0:
            for j in range(i + 1, n):
                if a[j, i] != 0:
                    a[i], a[j] = a[j], a[i]
                    e[i], e[j] = e[j], e[i]
                    break
            a0 = a[i, i]

        for j in range(i + 1, n):
            a1 = a[j, i]
            if use_fractional:
                e[j] = e[j] + e[i] * Fraction(-a1, a0)
                a[j] = a[j] + a[i] * Fraction(-a1, a0)
            else:
                e[j] = e[j] + e[i] * (-a1 / a0)
                a[j] = a[j] + a[i] * (-a1 / a0)
    ##################################
    for i in range(n - 1, 0, -1):
        a0 = a[i, i]
        for j in range(i - 1, -1, -1):
            a1 = a[j, i]
            if use_fractional:
                e[j] = e[j] + e[i] * Fraction(-a1, a0)
                a[j] = a[j] + a[i] * Fraction(-a1, a0)
            else:
                e[j] = e[j] + e[i] * (-a1 / a0)
                a[j] = a[j] + a[i] * (-a1 / a0)
    ##################################
    for i in range(n):
        if use_fractional:
            e[i] *= Fraction(1, a[i, i])
            a[i] *= Fraction(1, a[i, i])
        else:
            e[i] *= 1 / a[i, i]
            a[i] *= 1 / a[i, i]
    return e


def tensor_from_iterable(source):
    if isinstance(source[0], Iterable):
        return Tensor([tensor_from_iterable(i) for i in source])
    else:
        return Tensor(source)


def rref(m, use_fractional=True, transpositions_allowed=True):
    """
    Reduced row echelon form
    :param m:
    :param use_fractional: if in matrix there are only int numbers, it may provide better precision
    :param transpositions_allowed: don't use transpositions
    :return: return matrix in Reduced row echelon form
    """
    if not transpositions_allowed:
        raise NotImplementedError()
    m = m.copy()
    dim1 = m.shape()[0]
    dim2 = m.shape()[1]
    for i in range(dim1):
        first = 0
        ind = 0

        for j in range(dim2):
            if m[i, j] != 0:
                first = m[i, j]
                ind = j
                break

        for j in range(dim2):
            if m[i, j] != 0:
                if use_fractional:
                    m[i, j] = Fraction(m[i, j], first)
                else:
                    m[i, j] /= first
        if not first:
            continue

        for j in range(dim1):
            if j != i:
                m[j] = m[j] - m[i]*m[j, ind]
    return m


def rank(m):
    """
    Calculates rank of matrix m by converting it into rref and counting nonzero rows
    :param m: 2D Tensor
    :return: int - rank of matrix
    """
    assert len(m.shape()) == 2, "Must be an 2D Matrix"
    res = 0
    for r in rref(m):
        if max(r) != 0 or min(r) != 0:
            res += 1
    return res


def solve_hsle(matrix, use_fractional=True, transpositions_allowed=True):
    """
    Solves homogeneous system of linear equations Ax = 0
    :param transpositions_allowed: don't use transpositions while solving
    :param use_fractional: using Fractional is more precise, but can't be used with variables
    :param matrix: matrix A
    :return: list of vector_solutions
    """
    assert len(matrix.shape()) == 2, "Need 2D Matrix"
    free_vars = []
    not_free = []
    res = []
    len_x = matrix.shape()[1]
    matrix_rref = rref(matrix, use_fractional, transpositions_allowed)
    for i, row in enumerate(matrix_rref.transpose2()):
        for j, el in enumerate(row):
            if el == 1:
                not_free.append((j, i))
            elif el != 0:
                free_vars.append(i)
                break

    free_vars = set(free_vars)
    for i in free_vars:
        x = zeros(len_x)
        x[i] = 1
        d = dot(matrix_rref, x)

        for k, j in not_free:
            x[j] = -1*d[k][0]

        res.append(x.copy())
    return res
