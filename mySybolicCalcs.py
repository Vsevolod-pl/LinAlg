from numbers import Number


class Variable:
    def __init__(self, name, mult=1, p=1):
        self.name = name
        assert isinstance(mult, Number)
        self.mult = mult
        assert isinstance(p, Number)
        self.pow = p

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.name == other.name and self.mult == other.mult and self.pow == other.pow
        else:
            return str(other) == str(self)

    def __add__(self, other):
        if isinstance(other, Variable):
            if other.name == self.name and other.pow == self.pow:
                res = Variable(self.name, self.mult + other.mult, other.pow)
                if res == 0:
                    return 0
                return res
            else:
                return ExpressionSum([self, other], 0)
        else:
            return ExpressionSum([self], 0) + other

    __radd__ = __add__

    def __sub__(self, s):
        return self + (-1 * s)

    def __rsub__(self, other):
        return other+(-1*self)

    def __mul__(self, other):
        if isinstance(other, Number):
            if other == 0:
                return 0
            return Variable(self.name, self.mult * other, self.pow)
        elif isinstance(other, Variable):
            if other.name == self.name:
                return Variable(self.name, self.mult * other.mult, self.pow + other.pow)
            else:
                return ExpressionMul([self,other])
                raise NotImplementedError
        elif isinstance(other, ExpressionSum):
            return other.__mul__(self)
        else:
            raise NotImplementedError

    __rmul__ = __mul__

    def __pow__(self, other):
        if isinstance(other, Number):
            if self.pow*other == 0:
                return 1
            return Variable(self.name, self.mult ** other, self.pow*other)
        else:
            raise NotImplementedError

    def __repr__(self):
        if self.mult == 0:
            return "0"
        elif self.pow == 0:
            return "1"
        else:
            res = self.name
            if self.mult != 1:
                res = str(self.mult) + res
            if self.pow != 1:
                res += "^" + str(self.pow)
            return res


class ExpressionSum:
    def __init__(self, variables, const):
        self.vars = variables
        self.const = const
        self.clear()

    def clear(self):
        for var in self.vars:
            if isinstance(var, Number):
                self.const += var
        self.vars = [i for i in self.vars if not isinstance(i, Number)]

    def __add__(self, other):
        res = self.copy()
        if isinstance(other, Number):
            res.const += other
            return res
        elif isinstance(other, Variable):
            found = False
            for i in range(len(res.vars)):
                if res.vars[i] * other.mult == other * res.vars[i].mult:
                    res.vars[i] += other
                    found = True
                    break
            if not found:
                res.vars.append(other)
            res.clear()
            return res
        elif isinstance(other, ExpressionSum):
            for var in other.vars:
                res = res + var
            res += other.const
            return res
        else:
            raise NotImplementedError

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-1 * other)

    def __mul__(self, other):
        res = self.copy()
        if isinstance(other, Number):
            if other == 0:
                return 0
            for i in range(len(res.vars)):
                res.vars[i] *= other
            res.const *= other
            return res
        elif isinstance(other, Variable):
            for i in range(len(res.vars)):
                res.vars[i] *= other
            res += res.const*other
            res.const = 0
            res.clear()
            return res
        elif isinstance(other, ExpressionSum):
            res = 0
            for var in other.vars:
                res += self.copy()*var
            return res
        else:
            print(self, other)
            raise NotImplementedError

    __rmul__ = __mul__

    def __repr__(self):
        res = ""
        for var in self.vars:
            if var.mult > 0:
                res += " + " + repr(var)
            elif var.mult < 0:
                res += " - " + repr(-1 * var)
        if self.const > 0:
            res += " + " + str(self.const)
        elif self.const < 0:
            res += " - " + str(-self.const)
        elif len(res) == 0:
            res = "0"

        if len(res) > 3 and res[1] == "+":
            res = res[3:]
        elif len(res) > 3 and res[1] == "-":
            res = "-" + res[3:]

        return "(" + res + ")"

    def copy(self):
        return ExpressionSum(self.vars, self.const)


class ExpressionMul:
    def __init__(self, variables, const=1, cast_to_normal=True):
        self.vars = variables
        self.const = const
        if cast_to_normal:
            self.clear()

    def clear(self):
        name_to_varind = dict()
        new_vars = []
        for val1 in self.vars:
            if isinstance(val1, Variable):
                val = Variable(val1.name, 1, val1.pow)
                self.const *= val1.mult
                if val.name in name_to_varind:
                    new_vars[name_to_varind[val.name]] *= val
                else:
                    name_to_varind[val.name] = len(new_vars)
                    new_vars.append(val)
        self.vars = new_vars

    def __mul__(self, other):
        res = self.copy()
        if isinstance(other, Number):
            res.const *= other
        elif isinstance(other, Variable):
            res.const *= other.mult
            a1 = Variable(other.name, 1, other.pow)
            for i in range(len(res.vars)):
                pass
            return res

    def copy(self):
        return ExpressionMul(self.vars, self.const, False)

    def __repr__(self):
        if self.const == 0:
            return "0"
        else:
            res = ""
            if self.const != 1:
                res = str(self.const)
            for i in self.vars:
                res += repr(i)
            return res
