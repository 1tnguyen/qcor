from qcor import *
import math, time
import time

# Create the H20 hamiltonian with Pyscf
molecule_string = 'O 0 0 0; H 0 -2.757 2.587; H 0 2.757  2.587'
H = createOperator('pyscf', {'basis': 'sto-3g', 'geometry': molecule_string})

nQubits = H.nBits()
nElectrons = 10

# print(H.toString())
@qjit
def ansatz(q : qreg, n_electron: int,params : List[float]):
    uccsd(q, q.size(), n_electron, params)

q = qalloc(nQubits)
nOccupied = math.ceil(nElectrons / 2.0)
nVirtual = nQubits / 2 - nOccupied
nOrbitals = nOccupied + nVirtual
nSingle = nOccupied * nVirtual
nDouble = nSingle * (nSingle + 1) / 2
nParameters = int(nSingle + nDouble)
print("Number of parameters =", nParameters)
params = []
for i in range(nParameters):
    params.append(1.0)

start = time.time()
ansatz.print_kernel(q, nElectrons, params)
end = time.time()
print("Kernel eval time:", end - start, " [secs]")
