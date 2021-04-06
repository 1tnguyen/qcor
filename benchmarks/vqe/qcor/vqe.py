from qcor import *
@qjit
def ansatz(q : qreg, t0: float):
  X(q[0])
  Ry(q[1], t0)
  CNOT(q[1],q[0])    

# Define the Hamiltonian
H = -2.1433 * X(0) * X(1) \
    - 2.1433 * Y(0) * Y(1) \
    + .21829 * Z(0) \
    - 6.125 * Z(1) \
    + 5.907

# Create the ObjectiveFunction
obj = createObjectiveFunction(ansatz, H, 1, {'verbose': True})

# Create the nlopt optimizer 
# Use COBYLA method by default.
optimizer = createOptimizer('nlopt')

# Run VQE.
results = optimizer.optimize(obj)

print(results)