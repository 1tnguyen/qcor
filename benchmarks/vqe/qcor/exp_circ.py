from qcor import *
import time 

@qjit
def trotter_circ(q : qreg, exp_args: List[PauliOperator], n_steps: int):
  for i in range(n_steps):
    for exp_arg in exp_args:
      exp_i_theta(q, 1.0, exp_arg)



#H2 = createOperator('pyscf', {'basis': 'sto-3g', 'geometry': 'H  0.000000   0.0      0.0\nH   0.0        0.0  .7474'})

# H_2_O = createOperator('pyscf', {'basis': 'sto-3g', 'geometry': 
# '''O    0.   0.       0
# h    0.   -0.757   0.587
# h    0.   0.757    0.587'''})

#N2 = createOperator('pyscf', {'basis': 'sto-3g', 'geometry': 'N 0 0 0 \n N 0 0 1.1'})


C2_H_4 = createOperator('pyscf', {'basis': 'sto-3g', 'geometry': 
'''C -0.65830719  0.61123287 -0.00800148
   C 0.73685281  0.61123287 -0.00800148
   H  1.43439081  1.81898387 -0.00800148
   H -1.35568919  1.81920887 -0.00868348
   H -1.20806619 -0.34108413 -0.00755148
   H  1.28636081 -0.34128013 -0.00668648
'''})


H = C2_H_4
# print(H.toString())
# print(H.nBits())
# print(H.asPauli.getNonIdentitySubTerms())

q = qalloc(H.nBits())
nbSteps = 1

print('n_bits =', H.nBits())
print('n_terms =', len(H.asPauli.getNonIdentitySubTerms()))
start = time.time()
n_instructions = trotter_circ.n_instructions(q, H.asPauli.getNonIdentitySubTerms(), nbSteps)
end = time.time()
print('Elapsed time =', end - start, "[secs]")
print(n_instructions)