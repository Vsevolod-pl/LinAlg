from numbers import Number


class Variable:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if isinstance(other, Variable):
            return other.name == self.name
        return False


class ExpressionSum:
    def __init__(self, variables, const=0):
        self.const = const
        self.variables = variables

    def copy(self):
        return ExpressionSum(self.variables.copy(), self.const)

    def __add__(self, other):
        res = self.copy()
        if isinstance(other, Number):
            res.const += other
            return res
        elif isinstance(other, ExpressionSum):
            res.const += other.const
            for i in other.variables:
                res = res + i
            return res
        else:
            for i, v in enumerate(res.variables):
                if not isinstance(other+v, ExpressionSum):
                    res.variables[i] = other+v
                    return res
            res.variables.append(other)
            return res

    __radd__ = __add__

    def __sub__(self, other):
        return self+-1*other

    def __repr__(self):
        res = "("
        first = True
        for v in self.variables:
            if repr(v) != "":
                first = False
                res += repr(v)
            if not first:
                res += " + "

            res += repr(v)

        if not first or self.const != 0:
            res += repr(self.const)
        return res


class ExpressionMul:
    def __init__(self, variables, const=1):
        self.const = const
        self.variables = variables

    def copy(self):
        return ExpressionMul(self.variables.copy(), self.const)

    def __mul__(self, other):
        res = self.copy()
        if isinstance(other, Number):
            res.const *= other
            return res
        elif isinstance(other, ExpressionMul):
            res.const *= other.const
            for i in other.variables:
                res = res * i
            return res
        else:
            for i, v in enumerate(res.variables):
                if not isinstance(other*v, ExpressionMul):
                    res.variables[i] = other*v
                    return res
            res.variables.append(other)
            return res

    __rmul__ = __mul__

    def __sub__(self, other):
        return self+-1*other

    def __repr__(self):
        res = "("
        first = True
        for v in self.variables:
            if repr(v) != "":
                first = False
                res += repr(v)
            if not first:
                res += " + "

            res += repr(v)

        if not first or self.const != 0:
            res += repr(self.const)
