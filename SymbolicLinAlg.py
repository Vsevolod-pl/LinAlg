from . import LinAlgFuncs, mySymbolicCalcs

def symb_matrix(var_name, *shape):
	if len(shape) == 1:
        return Tensor([Variable(var_name+str(i)) for i in range(shape[0])])
    else:
        return Tensor([zeros(var_name+str(i),*shape[1:]) for i in range(shape[0])])