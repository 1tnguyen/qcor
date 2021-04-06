import time 
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

# driver = PySCFDriver(atom='O 0. 0. 0.;H 0. -0.757 0.587;H 0. 0.757 0.587', unit=UnitsType.ANGSTROM, basis='sto3g')
# driver = PySCFDriver(atom='N 0 0 0; N 0 0 1.1', unit=UnitsType.ANGSTROM, basis='sto3g')
driver = PySCFDriver(atom='C -0.65830719 0.61123287 -0.00800148;C 0.73685281 0.61123287 -0.00800148;H 1.43439081 1.81898387 -0.00800148;H -1.35568919 1.81920887 -0.00868348;H -1.20806619 -0.34108413 -0.00755148;H 1.28636081 -0.34128013 -0.00668648', unit=UnitsType.ANGSTROM, basis='sto3g')

molecule = driver.run()
fer_op = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
map_type = 'PARITY'
qubit_op = fer_op.mapping(map_type)
print(qubit_op)


nbQubits = qubit_op.num_qubits
print(nbQubits)
ham_op = qubit_op
nbSteps = 1
q = QuantumRegister(nbQubits, 'q')
start = time.time()
circuit = ham_op.evolve().decompose()
end = time.time()
print("Kernel eval time:", end - start, " [secs]")
ops_count = circuit.count_ops()
print(ops_count)