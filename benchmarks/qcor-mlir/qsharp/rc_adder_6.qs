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
        using (qubits = Qubit[14]) {
            CNOT(qubits[4], qubits[3]);
            CNOT(qubits[6], qubits[5]);
            CNOT(qubits[8], qubits[7]);
            CNOT(qubits[10], qubits[9]);
            CNOT(qubits[12], qubits[11]);
            CNOT(qubits[4], qubits[2]);
            CCNOT(qubits[0], qubits[1], qubits[2]);
            CNOT(qubits[6], qubits[4]);
            CCNOT(qubits[2], qubits[3], qubits[4]);
            CNOT(qubits[8], qubits[6]);
            CCNOT(qubits[4], qubits[5], qubits[6]);
            CNOT(qubits[10], qubits[8]);
            CCNOT(qubits[6], qubits[7], qubits[8]);
            CNOT(qubits[12], qubits[10]);
            CCNOT(qubits[8], qubits[9], qubits[10]);
            CNOT(qubits[12], qubits[13]);
            X(qubits[3]);
            X(qubits[5]);
            X(qubits[7]);
            X(qubits[9]);
            CCNOT(qubits[10], qubits[11], qubits[13]);
            CNOT(qubits[2], qubits[3]);
            CNOT(qubits[4], qubits[5]);
            CNOT(qubits[6], qubits[7]);
            CNOT(qubits[8], qubits[9]);
            CNOT(qubits[10], qubits[11]);
            CCNOT(qubits[8], qubits[9], qubits[10]);
            CCNOT(qubits[6], qubits[7], qubits[8]);
            X(qubits[9]);
            CNOT(qubits[12], qubits[10]);
            CCNOT(qubits[4], qubits[5], qubits[6]);
            X(qubits[7]);
            CNOT(qubits[10], qubits[8]);
            CCNOT(qubits[2], qubits[3], qubits[4]);
            X(qubits[5]);
            CNOT(qubits[8], qubits[6]);
            CCNOT(qubits[0], qubits[1], qubits[2]);
            X(qubits[3]);
            CNOT(qubits[6], qubits[4]);
            CNOT(qubits[4], qubits[2]);
            CNOT(qubits[1], qubits[0]);
            CNOT(qubits[4], qubits[3]);
            CNOT(qubits[6], qubits[5]);
            CNOT(qubits[8], qubits[7]);
            CNOT(qubits[10], qubits[9]);
            CNOT(qubits[12], qubits[11]);

            ResetAll(qubits);
        }
    }
}
