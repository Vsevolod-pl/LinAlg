from . import LinAlgFuncs as la
from . import mySybolicCalcs as sc


def symbolic_matrix(var_name, *shape):
    if len(shape) == 1:
        return la.Tensor([sc.Variable(var_name+str(i)) for i in range(shape[0])])
    return la.Tensor([symbolic_matrix(var_name+str(i), *shape[1:]) for i in range(shape[0])])
