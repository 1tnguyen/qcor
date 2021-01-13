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
        using (qubits = Qubit[24]) {
            CNOT(qubits[3], qubits[2]);
            CNOT(qubits[8], qubits[7]);
            CNOT(qubits[14], qubits[13]);
            CNOT(qubits[21], qubits[20]);
            CNOT(qubits[3], qubits[4]);
            CNOT(qubits[8], qubits[9]);
            CNOT(qubits[14], qubits[15]);
            CNOT(qubits[21], qubits[22]);
            CCNOT(qubits[0], qubits[1], qubits[3]);
            CCNOT(qubits[5], qubits[6], qubits[8]);
            CCNOT(qubits[11], qubits[12], qubits[14]);
            CCNOT(qubits[18], qubits[19], qubits[21]);
            CCNOT(qubits[2], qubits[3], qubits[4]);
            CCNOT(qubits[7], qubits[8], qubits[9]);
            CCNOT(qubits[7], qubits[8], qubits[10]);
            CCNOT(qubits[13], qubits[14], qubits[15]);
            CCNOT(qubits[13], qubits[14], qubits[16]);
            CCNOT(qubits[20], qubits[21], qubits[22]);
            CCNOT(qubits[20], qubits[21], qubits[23]);
            CNOT(qubits[6], qubits[5]);
            CNOT(qubits[12], qubits[11]);
            CNOT(qubits[19], qubits[18]);
            CNOT(qubits[5], qubits[8]);
            CNOT(qubits[11], qubits[14]);
            CNOT(qubits[18], qubits[21]);
            CCNOT(qubits[7], qubits[8], qubits[10]);
            CCNOT(qubits[13], qubits[14], qubits[16]);
            CCNOT(qubits[20], qubits[21], qubits[23]);
            CCNOT(qubits[16], qubits[23], qubits[17]);
            CCNOT(qubits[15], qubits[23], qubits[22]);
            CCNOT(qubits[4], qubits[10], qubits[9]);
            CCNOT(qubits[9], qubits[17], qubits[22]);
            CCNOT(qubits[9], qubits[16], qubits[15]);
            CCNOT(qubits[16], qubits[23], qubits[17]);
            CCNOT(qubits[7], qubits[8], qubits[10]);
            CCNOT(qubits[13], qubits[14], qubits[16]);
            CCNOT(qubits[20], qubits[21], qubits[23]);
            CNOT(qubits[5], qubits[8]);
            CNOT(qubits[11], qubits[14]);
            CNOT(qubits[18], qubits[21]);
            CNOT(qubits[6], qubits[5]);
            CNOT(qubits[12], qubits[11]);
            CNOT(qubits[19], qubits[18]);
            CCNOT(qubits[7], qubits[8], qubits[10]);
            CCNOT(qubits[13], qubits[14], qubits[16]);
            CCNOT(qubits[20], qubits[21], qubits[23]);
            CCNOT(qubits[0], qubits[1], qubits[3]);
            CCNOT(qubits[5], qubits[6], qubits[8]);
            CCNOT(qubits[11], qubits[12], qubits[14]);
            CCNOT(qubits[18], qubits[19], qubits[21]);
            CNOT(qubits[3], qubits[2]);
            CNOT(qubits[8], qubits[7]);
            CNOT(qubits[14], qubits[13]);
            CNOT(qubits[21], qubits[20]);
            CNOT(qubits[6], qubits[5]);
            CNOT(qubits[12], qubits[11]);
            CNOT(qubits[19], qubits[18]);
            CNOT(qubits[6], qubits[8]);
            CNOT(qubits[12], qubits[14]);
            CNOT(qubits[19], qubits[21]);
            CNOT(qubits[4], qubits[6]);
            CNOT(qubits[9], qubits[12]);
            CNOT(qubits[15], qubits[19]);
            CCNOT(qubits[0], qubits[1], qubits[3]);
            CCNOT(qubits[5], qubits[6], qubits[8]);
            CCNOT(qubits[11], qubits[12], qubits[14]);
            CCNOT(qubits[18], qubits[19], qubits[21]);
            CNOT(qubits[3], qubits[2]);
            CNOT(qubits[8], qubits[7]);
            CNOT(qubits[14], qubits[13]);
            CNOT(qubits[21], qubits[20]);
            CCNOT(qubits[0], qubits[1], qubits[3]);
            CCNOT(qubits[5], qubits[6], qubits[8]);
            CCNOT(qubits[11], qubits[12], qubits[14]);
            CCNOT(qubits[18], qubits[19], qubits[21]);
            CNOT(qubits[6], qubits[5]);
            CNOT(qubits[12], qubits[11]);
            CNOT(qubits[19], qubits[18]);
            CNOT(qubits[4], qubits[6]);
            CNOT(qubits[9], qubits[12]);
            CNOT(qubits[15], qubits[19]);
            CNOT(qubits[6], qubits[8]);
            CNOT(qubits[12], qubits[14]);
            CNOT(qubits[19], qubits[21]);
            CNOT(qubits[1], qubits[0]);
            CNOT(qubits[6], qubits[5]);
            CNOT(qubits[12], qubits[11]);
            CNOT(qubits[19], qubits[18]);
            X(qubits[0]);
            X(qubits[2]);
            X(qubits[5]);
            X(qubits[7]);
            X(qubits[11]);
            X(qubits[13]);
            CNOT(qubits[3], qubits[2]);
            CNOT(qubits[8], qubits[7]);
            CNOT(qubits[14], qubits[13]);
            CCNOT(qubits[0], qubits[1], qubits[3]);
            CCNOT(qubits[5], qubits[6], qubits[8]);
            CCNOT(qubits[11], qubits[12], qubits[14]);
            CNOT(qubits[6], qubits[5]);
            CNOT(qubits[12], qubits[11]);
            CCNOT(qubits[7], qubits[8], qubits[10]);
            CCNOT(qubits[13], qubits[14], qubits[16]);
            CNOT(qubits[5], qubits[8]);
            CNOT(qubits[11], qubits[14]);
            CCNOT(qubits[7], qubits[8], qubits[10]);
            CCNOT(qubits[13], qubits[14], qubits[16]);
            CCNOT(qubits[9], qubits[16], qubits[15]);
            CCNOT(qubits[4], qubits[10], qubits[9]);
            CCNOT(qubits[7], qubits[8], qubits[10]);
            CCNOT(qubits[13], qubits[14], qubits[16]);
            CNOT(qubits[5], qubits[8]);
            CNOT(qubits[11], qubits[14]);
            CCNOT(qubits[7], qubits[8], qubits[10]);
            CCNOT(qubits[13], qubits[14], qubits[16]);
            CNOT(qubits[6], qubits[5]);
            CNOT(qubits[12], qubits[11]);
            CCNOT(qubits[7], qubits[8], qubits[9]);
            CCNOT(qubits[13], qubits[14], qubits[15]);
            CCNOT(qubits[2], qubits[3], qubits[4]);
            CCNOT(qubits[0], qubits[1], qubits[3]);
            CCNOT(qubits[5], qubits[6], qubits[8]);
            CCNOT(qubits[11], qubits[12], qubits[14]);
            CNOT(qubits[3], qubits[4]);
            CNOT(qubits[8], qubits[9]);
            CNOT(qubits[14], qubits[15]);
            CNOT(qubits[3], qubits[2]);
            CNOT(qubits[8], qubits[7]);
            CNOT(qubits[14], qubits[13]);
            X(qubits[0]);
            X(qubits[2]);
            X(qubits[5]);
            X(qubits[7]);
            X(qubits[11]);
            X(qubits[13]);

            ResetAll(qubits);
        }
    }
}
