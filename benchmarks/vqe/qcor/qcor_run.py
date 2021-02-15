from qcor import *

# Create the H20 hamiltonian with Pyscf
molecule_string = 'O 0 0 0; H 0 -2.757 2.587; H 0 2.757  2.587'
H = createOperator('pyscf', {'basis': 'sto-3g', 'geometry': molecule_string})
# print(H.toString())
# TODO: using XACC UCCSD ansatz
@qjit
def ansatz(q : qreg, params : List[float]):
    uccsd(q)

q = qalloc(4)
ansatz.print_kernel(q, [])