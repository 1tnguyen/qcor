from qcor import *
import time
@qjit
def trotter_circ(q : qreg, exp_args: List[PauliOperator], n_steps: int):
  for i in range(n_steps):
    for i, exp_arg in enumerate(exp_args):
      exp_i_theta(q, 1.0, exp_args[i])

def heisenberg_ham(n_qubits):
  Jz = 1.0
  h = 1.0
  H = 0.0 * X(0)
  for i in range(n_qubits):
    H = H - h * X(i)
  for i in range(n_qubits - 1):
    H = H - Jz * (Z(i) * Z(i + 1))
  return H

n_qubits = [10, 20, 50, 100]
nbSteps = 100

for nbQubits in n_qubits:
  ham_op = heisenberg_ham(nbQubits)
  op_terms = ham_op.getNonIdentitySubTerms()
  q = qalloc(nbQubits)
  start = time.time()
  comp = trotter_circ.extract_composite(q, op_terms, nbSteps)
  end = time.time()
  print("n instructions =", comp.nInstructions(), "; Kernel eval time:", end - start, " [secs]")