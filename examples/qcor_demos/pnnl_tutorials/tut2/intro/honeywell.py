# Honeywell job submission summary:
# - Set up credentials: qcor -set-credentials honeywell 

from qcor import *

@qjit
def ghz(q : qreg):
    H(q[0])
    for i in range(q.size()-1):
        X.ctrl([q[i]], q[i+1])

    Measure(q)

set_qpu('honeywell:HQS-LT-S1-SIM')
set_shots(1024)
q = qalloc(6)

ghz(q)
q.print()
