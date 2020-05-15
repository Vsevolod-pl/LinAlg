from numbers import Number


class Variable:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        #print(other)
        if isinstance(other, Variable):
            return other.name == self.name
        return False

    def reduce(self):
        return self

    def copy(self):
        return Variable(self.name)

    def __add__(self, other):
        if isinstance(other, Variable):
            if other.name == self.name:
                return ExpressionMul([self, ], 2)
        elif other == 0:
            return self
        elif isinstance(other, ExpressionMul):
            if len(other.variables) == 1:
                if other.variables[0] == self:
                    r = other.copy()
                    r.const += 1
                    return r
        return ExpressionSum([self, ]) + other

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, ExpressionPower):
            if other.var == self:
                r = other.copy()
                r.power += 1
                return r
        elif isinstance(other, Variable) and other == self:
            return ExpressionPower(self, 2)
        elif other == 1:
            return self
        elif other == 0:
            return 0
        return ExpressionMul([self, ])*other

    __rmul__ = __mul__

    def __sub__(self, other):
        return self.__add__(-1*other)

    def can_add(self, other):
        if isinstance(other, Variable):
            if other.name == self.name:
                return True
        elif other == 0:
            return True
        elif isinstance(other, ExpressionMul):
            if len(other.variables) == 1:
                if other.variables[0] == self:
                    return True
        return False

    def can_mul(self, other):
        if isinstance(other, ExpressionPower):
            if other.var == self:
                return True
        elif other == 1:
            return True
        elif other == 0:
            return True
        return False

    def __repr__(self):
        return self.name


class ExpressionSum:
    def __init__(self, variables, const=0):
        self.const = const
        self.variables = variables

    def copy(self):
        return ExpressionSum([var.copy() for var in self.variables], self.const)

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
                if v.can_add(other) or other.can_add(v):
                    res.variables[i] = other+v
                    return res
            res.variables.append(other)
            return res

    __radd__ = __add__

    def can_add(self, other):
        return True

    def __sub__(self, other):
        return self.copy()+-1*other

    def __repr__(self):
        res = "("
        first = True
        for v in self.variables:
            if not first:
                res += " + "
            elif repr(v) != "":
                first = False
            res += repr(v)

        if first or self.const != 0:
            if self.const >= 0 and not first:
                res += " + "
            res += repr(self.const)
        return res+")"

    def reduce(self):
        res = self.const
        for el in self.variables:
            res = res + el.reduce()
        if isinstance(res, ExpressionSum):
            if len(res.variables) == 0:
                return res.const
            if len(res.variables) == 1 and res.const == 0:
                return res.variables[0]
        return res

    def __mul__(self, other):
        res = 0
        for var in self.variables:
            res = res + var * other
        res = res + self.const*other
        return res

    __rmul__ = __mul__

    def can_mul(self, other):
        return True


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
                if other.can_mul(v) or v.can_mul(other):
                    res.variables[i] = other*v
                    return res
            res.variables.append(other)
            return res

    __rmul__ = __mul__

    def can_mul(self, other):
        return True

    def __sub__(self, other):
        return self+-1*other

    def __repr__(self):
        return repr(self.const) + "*".join([repr(var) for var in self.variables])

    def reduce(self):
        res = self.const
        for el in self.variables:
            res = res*el.reduce()
        if isinstance(res, ExpressionMul):
            if res.const == 1 and len(res.variables) == 1:
                return res.variables[0].copy()
            elif res.const == 0:
                return 0
        return res

    def can_add(self, other):
        return False

    def __add__(self, other):
        return ExpressionSum([self, ]) + other

    __radd__ = __add__


class ExpressionPower:
    def __init__(self, var, power=1):
        self.var = var
        self.power = power

    def __eq__(self, other):
        if isinstance(other, ExpressionPower):
            return other.var == self.var and other.power == self.power
        elif isinstance(other, Number):
            return self.power == 0 and other == 1
        else:
            return False

    def copy(self):
        return ExpressionPower(self.var.copy(), self.power)

    def __mul__(self, other):
        if isinstance(other, ExpressionPower):
            if other.var == self.var:
                r = self.copy()
                r.power += other.power
                return r
        elif other == 1:
            return self.copy()
        elif other == 0:
            return 0
        return ExpressionMul([self.copy()])*other

    __rmul__ = __mul__

    def __rtruediv__(self, other):
        r = self.copy()
        r.power *= -1
        return other*r

    def __truediv__(self, other):
        if isinstance(other, ExpressionPower):
            r = other.copy()
            r.power *= -1
            return self * r
        return self*ExpressionPower(other, -1)

    def can_mul(self, other):
        if isinstance(other, ExpressionPower):
            if other.var == self.var:
                return True
        elif other == 1:
            return True
        elif other == 0:
            return True
        return False

    def __add__(self, other):
        if isinstance(other, ExpressionPower):
            if other == self:
                return ExpressionMul([self.copy(), ], 2)
        elif other == 0:
            return self.copy()
        elif isinstance(other, ExpressionMul):
            if len(other.variables) == 1:
                if other.variables[0] == self:
                    r = other.copy()
                    r.const += 1
                    return r
        return ExpressionSum([self.copy(), ]) + other

    __radd__ = __add__

    def __sub__(self, other):
        return self + -1*other

    def can_add(self, other):
        if isinstance(other, ExpressionPower):
            if other == self:
                return True
        elif other == 0:
            return True
        elif isinstance(other, ExpressionMul):
            if len(other.variables) == 1:
                if other.variables[0] == self:
                    return True
        return False

    def __repr__(self):
        return self.var.__repr__()+"^"+repr(self.power)

    def reduce(self):
        return ExpressionPower(self.var.reduce(), self.power)
