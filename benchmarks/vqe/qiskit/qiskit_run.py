from qiskit.aqua import aqua_globals
from qiskit import Aer
from qiskit.aqua.operators import X, Z, I
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.chemistry.core import QubitMappingType
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.applications import MolecularGroundStateEnergy


basis_string = 'sto-3g'
# Molecules to test
h20_mole = 'O 0 0 0; H 0 -2.757 2.587; H 0 2.757  2.587'
n2_mol = '0.0, 0.0, 0.56499; N 0.0, 0.0, -0.56499'
hcn_mole = 'C 0.0, 0.0, -0.511747; N 0.0, 0.0, 0.664461; H 0.0, 0.0, -1.580746'
driver = PySCFDriver(atom=hcn_mole, basis = 'sto-3g')

def cb_create_solver(num_particles, num_orbitals,
                        qubit_mapping, two_qubit_reduction, z2_symmetries):
    print('num_particles =', num_particles, "; num_orbitals =", num_orbitals)
    initial_state = HartreeFock(num_orbitals, num_particles, qubit_mapping,
                                two_qubit_reduction, z2_symmetries.sq_list)
    var_form = UCCSD(num_orbitals=num_orbitals,
                        num_particles=num_particles,
                        initial_state=initial_state,
                        qubit_mapping=qubit_mapping,
                        two_qubit_reduction=two_qubit_reduction,
                        z2_symmetries=z2_symmetries)
    
    # Print circuit at each iteration.
    def iter_cb_func(eval_count, parameters, mean, std):
      print("Run ", eval_count, ": E = ", mean)
      iter_circ = var_form.construct_circuit(parameters).decompose()
      #print(iter_circ.qasm())
      op_count = iter_circ.count_ops()
      for op_name in op_count:
        print(op_name, ":", op_count[op_name])
    
    vqe = VQE(var_form=var_form, optimizer=SLSQP(maxiter=500), include_custom=True, callback=iter_cb_func)
    vqe.quantum_instance = Aer.get_backend('qasm_simulator')
    return vqe

mgse = MolecularGroundStateEnergy(driver, qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                                  two_qubit_reduction=False, freeze_core=False,
                                  z2symmetry_reduction=None)
result = mgse.compute_energy(cb_create_solver)
import pprint
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(result)