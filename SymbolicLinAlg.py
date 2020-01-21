import sys
sys.path.insert(0, ".")

import LinAlgFuncs as la
import mySybolicCalcs as sc


def symbolic_tensor(var_name, *shape):
    if len(shape) == 1:
        return la.Tensor([sc.Variable(var_name+str(i)) for i in range(shape[0])])
    return la.Tensor([symbolic_matrix(var_name+str(i), *shape[1:]) for i in range(shape[0])])
