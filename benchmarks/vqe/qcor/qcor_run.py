from qcor import *
import math, time
import time

# Molecules to test
h2o_mole = 'O 0 0 0; H 0 -2.757 2.587; H 0 2.757  2.587'
n2_mol = 'N 0.0, 0.0, 0.56499; N 0.0, 0.0, -0.56499'
hcn_mole = 'C 0.0, 0.0, -0.511747; N 0.0, 0.0, 0.664461; H 0.0, 0.0, -1.580746'
test_cases = { "H20": h2o_mole, "N2": n2_mol, "HCN": hcn_mole }
# Data for validation
nElectrons_data = { "H20": 10, "N2": 14, "HCN": 14 }
nQubits_data = { "H20": 14, "N2": 20, "HCN": 22 }

@qjit
def ansatz(q : qreg, n_electron: int, params : List[float]):
    uccsd(q, q.size(), n_electron, params)

for test_case in test_cases:
    print("Run ", test_case)
    molecule_string = test_cases[test_case]
    H = createOperator('pyscf', {'basis': 'sto-3g', 'geometry': molecule_string})
    nQubits = H.nBits()
    assert nQubits == nQubits_data[test_case]
    nElectrons = nElectrons_data[test_case]
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
