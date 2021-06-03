namespace Quantum.staq {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Convert;
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Math;

    operation U(theta : Double, phi : Double, lambda : Double, q : Qubit) : Unit {
        Rz(lambda, q);
        Ry(theta, q);
        Rz(phi, q);
    }

    operation u3(theta : Double, phi : Double, lambda : Double, q : Qubit) : Unit {
        U(theta, phi, lambda, q);
    }

    operation u2(phi : Double, lambda : Double, q : Qubit) : Unit {
        U(PI()/2.0, phi, lambda, q);
    }

    operation u0(gamma : Double, q : Qubit) : Unit {
        U(0.0, 0.0, 0.0, q);
    }

    operation cy(a : Qubit, b : Qubit) : Unit {
        (Adjoint S)(b);
        CNOT(a, b);
        S(b);
    }

    operation swap(a : Qubit, b : Qubit) : Unit {
        CNOT(a, b);
        CNOT(b, a);
        CNOT(a, b);
    }

    operation cu3(theta : Double, phi : Double, lambda : Double, c : Qubit, t : Qubit) : Unit {
        Rz((lambda-phi)/2.0, t);
        CNOT(c, t);
        u3(-(theta/2.0), 0.0, -((phi+lambda)/2.0), t);
        CNOT(c, t);
        u3(theta/2.0, phi, 0.0, t);
    }

    @EntryPoint()
    operation Circuit() : Unit {
        using (qubits = Qubit[36]) {
            CCNOT(qubits[0], qubits[1], qubits[2]);
            CCNOT(qubits[3], qubits[4], qubits[5]);
            CCNOT(qubits[6], qubits[7], qubits[9]);
            CCNOT(qubits[10], qubits[11], qubits[12]);
            CCNOT(qubits[13], qubits[14], qubits[16]);
            CCNOT(qubits[17], qubits[18], qubits[20]);
            CCNOT(qubits[21], qubits[22], qubits[24]);
            CCNOT(qubits[25], qubits[26], qubits[27]);
            CCNOT(qubits[28], qubits[29], qubits[31]);
            CCNOT(qubits[32], qubits[33], qubits[34]);
            CNOT(qubits[3], qubits[4]);
            CNOT(qubits[6], qubits[7]);
            CNOT(qubits[10], qubits[11]);
            CNOT(qubits[13], qubits[14]);
            CNOT(qubits[17], qubits[18]);
            CNOT(qubits[21], qubits[22]);
            CNOT(qubits[25], qubits[26]);
            CNOT(qubits[28], qubits[29]);
            CNOT(qubits[32], qubits[33]);
            CCNOT(qubits[7], qubits[11], qubits[8]);
            CCNOT(qubits[14], qubits[18], qubits[15]);
            CCNOT(qubits[22], qubits[26], qubits[23]);
            CCNOT(qubits[29], qubits[33], qubits[30]);
            CCNOT(qubits[2], qubits[4], qubits[5]);
            CCNOT(qubits[9], qubits[11], qubits[12]);
            CCNOT(qubits[16], qubits[18], qubits[20]);
            CCNOT(qubits[24], qubits[26], qubits[27]);
            CCNOT(qubits[31], qubits[33], qubits[34]);
            CCNOT(qubits[15], qubits[23], qubits[19]);
            CCNOT(qubits[5], qubits[8], qubits[12]);
            CCNOT(qubits[20], qubits[23], qubits[27]);
            CCNOT(qubits[12], qubits[19], qubits[27]);
            CCNOT(qubits[12], qubits[15], qubits[20]);
            CCNOT(qubits[5], qubits[7], qubits[9]);
            CCNOT(qubits[12], qubits[14], qubits[16]);
            CCNOT(qubits[20], qubits[22], qubits[24]);
            CCNOT(qubits[27], qubits[29], qubits[31]);
            CCNOT(qubits[27], qubits[30], qubits[34]);
            CCNOT(qubits[15], qubits[23], qubits[19]);
            CCNOT(qubits[7], qubits[11], qubits[8]);
            CCNOT(qubits[14], qubits[18], qubits[15]);
            CCNOT(qubits[22], qubits[26], qubits[23]);
            CCNOT(qubits[29], qubits[33], qubits[30]);
            CNOT(qubits[1], qubits[35]);
            CNOT(qubits[4], qubits[2]);
            CNOT(qubits[7], qubits[5]);
            CNOT(qubits[11], qubits[9]);
            CNOT(qubits[14], qubits[12]);
            CNOT(qubits[18], qubits[16]);
            CNOT(qubits[22], qubits[20]);
            CNOT(qubits[26], qubits[24]);
            CNOT(qubits[29], qubits[27]);
            CNOT(qubits[33], qubits[30]);
            CNOT(qubits[0], qubits[35]);
            CNOT(qubits[3], qubits[4]);
            CNOT(qubits[6], qubits[7]);
            CNOT(qubits[10], qubits[11]);
            CNOT(qubits[13], qubits[14]);
            CNOT(qubits[17], qubits[18]);
            CNOT(qubits[21], qubits[22]);
            CNOT(qubits[25], qubits[26]);
            CNOT(qubits[28], qubits[29]);
            CNOT(qubits[32], qubits[33]);

            ResetAll(qubits);
        }
    }
}
