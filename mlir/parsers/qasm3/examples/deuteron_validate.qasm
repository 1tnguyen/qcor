OPENQASM 3;
include "stdgates.inc";

// Compile with: qcor -validate -qrt nisq deuteron_validate.qasm
def deuteron_xx(float[64]:theta) qubit[2]:q {
    x q[0];
    ry(theta) q[1];
    cx q[1], q[0];
    // Change basis to XX
    h q;
}

qubit q[2];
deuteron_xx(1.234) q;