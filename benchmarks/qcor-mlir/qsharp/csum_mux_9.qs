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

    operation Circuit() : Unit {
        using (qubits = Qubit[30]) {
            CCNOT(qubits[0], qubits[3], qubits[4]);
            X(qubits[1]);
            CCNOT(qubits[1], qubits[2], qubits[5]);
            X(qubits[1]);
            X(qubits[0]);
            CCNOT(qubits[0], qubits[2], qubits[4]);
            X(qubits[0]);
            CCNOT(qubits[1], qubits[3], qubits[5]);
            CCNOT(qubits[6], qubits[9], qubits[10]);
            X(qubits[7]);
            CCNOT(qubits[7], qubits[8], qubits[11]);
            X(qubits[7]);
            X(qubits[6]);
            CCNOT(qubits[6], qubits[8], qubits[10]);
            X(qubits[6]);
            CCNOT(qubits[7], qubits[9], qubits[11]);
            CCNOT(qubits[14], qubits[17], qubits[18]);
            X(qubits[15]);
            CCNOT(qubits[15], qubits[16], qubits[19]);
            X(qubits[15]);
            X(qubits[14]);
            CCNOT(qubits[14], qubits[16], qubits[18]);
            X(qubits[14]);
            CCNOT(qubits[15], qubits[17], qubits[19]);
            CCNOT(qubits[20], qubits[23], qubits[24]);
            X(qubits[21]);
            CCNOT(qubits[21], qubits[22], qubits[25]);
            X(qubits[21]);
            X(qubits[20]);
            CCNOT(qubits[20], qubits[22], qubits[24]);
            X(qubits[20]);
            CCNOT(qubits[21], qubits[23], qubits[25]);
            X(qubits[4]);
            CCNOT(qubits[4], qubits[10], qubits[12]);
            X(qubits[4]);
            CCNOT(qubits[5], qubits[11], qubits[13]);
            X(qubits[5]);
            CCNOT(qubits[5], qubits[10], qubits[13]);
            X(qubits[5]);
            CCNOT(qubits[4], qubits[11], qubits[12]);
            X(qubits[18]);
            CCNOT(qubits[18], qubits[24], qubits[26]);
            X(qubits[18]);
            CCNOT(qubits[19], qubits[25], qubits[27]);
            X(qubits[19]);
            CCNOT(qubits[19], qubits[24], qubits[27]);
            X(qubits[19]);
            CCNOT(qubits[18], qubits[25], qubits[26]);
            X(qubits[12]);
            CCNOT(qubits[12], qubits[26], qubits[28]);
            X(qubits[12]);
            CCNOT(qubits[13], qubits[27], qubits[29]);
            X(qubits[13]);
            CCNOT(qubits[13], qubits[26], qubits[29]);
            X(qubits[13]);
            CCNOT(qubits[12], qubits[27], qubits[28]);

            ResetAll(qubits);
        }
    }
}
