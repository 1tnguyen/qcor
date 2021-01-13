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
        using (qubits = Qubit[9]) {
            X(qubits[5]);
            H(qubits[0]);
            H(qubits[1]);
            H(qubits[2]);
            H(qubits[3]);
            H(qubits[4]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[3], qubits[7], qubits[8]);
            H(qubits[5]);
            CCNOT(qubits[4], qubits[8], qubits[5]);
            CCNOT(qubits[3], qubits[7], qubits[8]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            H(qubits[0]);
            H(qubits[1]);
            H(qubits[2]);
            H(qubits[3]);
            H(qubits[4]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            X(qubits[2]);
            X(qubits[3]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[3], qubits[7], qubits[4]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            X(qubits[2]);
            X(qubits[3]);
            H(qubits[0]);
            H(qubits[1]);
            H(qubits[2]);
            H(qubits[3]);
            H(qubits[4]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[3], qubits[7], qubits[8]);
            CCNOT(qubits[4], qubits[8], qubits[5]);
            CCNOT(qubits[3], qubits[7], qubits[8]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            H(qubits[0]);
            H(qubits[1]);
            H(qubits[2]);
            H(qubits[3]);
            H(qubits[4]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            X(qubits[2]);
            X(qubits[3]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[3], qubits[7], qubits[4]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            X(qubits[2]);
            X(qubits[3]);
            H(qubits[0]);
            H(qubits[1]);
            H(qubits[2]);
            H(qubits[3]);
            H(qubits[4]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[3], qubits[7], qubits[8]);
            CCNOT(qubits[4], qubits[8], qubits[5]);
            CCNOT(qubits[3], qubits[7], qubits[8]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            H(qubits[0]);
            H(qubits[1]);
            H(qubits[2]);
            H(qubits[3]);
            H(qubits[4]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            X(qubits[2]);
            X(qubits[3]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[3], qubits[7], qubits[4]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            X(qubits[2]);
            X(qubits[3]);
            H(qubits[0]);
            H(qubits[1]);
            H(qubits[2]);
            H(qubits[3]);
            H(qubits[4]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[3], qubits[7], qubits[8]);
            CCNOT(qubits[4], qubits[8], qubits[5]);
            CCNOT(qubits[3], qubits[7], qubits[8]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            H(qubits[0]);
            H(qubits[1]);
            H(qubits[2]);
            H(qubits[3]);
            H(qubits[4]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            X(qubits[2]);
            X(qubits[3]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[3], qubits[7], qubits[4]);
            CCNOT(qubits[2], qubits[6], qubits[7]);
            CCNOT(qubits[0], qubits[1], qubits[6]);
            X(qubits[4]);
            X(qubits[0]);
            X(qubits[1]);
            X(qubits[2]);
            X(qubits[3]);
            H(qubits[0]);
            H(qubits[1]);
            H(qubits[2]);
            H(qubits[3]);
            H(qubits[4]);

            ResetAll(qubits);
        }
    }
}
