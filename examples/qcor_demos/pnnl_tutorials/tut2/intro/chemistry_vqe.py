from qcor import *

@qjit 
def ansatz(q: qreg, theta: float):
    X(q[0])
    Ry(q[1], theta)
    CX(q[1], q[0])


# Define the hamiltonian
H = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) - 6.125 * Z(1)

# Define the optimization objective
def obj(x : List[float]):
    e_at_x = ansatz.observe(H, qalloc(2), x[0])
    print('E({}) = {}'.format(x[0], e_at_x))
    return e_at_x
    
# create the cobyla optimizer
optimizer = createOptimizer('nlopt', {'maxeval':25})

# Run VQE...
results = optimizer.optimize(obj, 1)
print(results)
