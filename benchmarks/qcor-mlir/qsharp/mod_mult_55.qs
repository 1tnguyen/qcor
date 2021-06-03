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
        using (qubits = Qubit[9]) {
            X(qubits[2]);
            CCNOT(qubits[0], qubits[2], qubits[7]);
            X(qubits[2]);
            X(qubits[0]);
            CCNOT(qubits[1], qubits[7], qubits[6]);
            CCNOT(qubits[0], qubits[2], qubits[8]);
            CNOT(qubits[6], qubits[5]);
            CNOT(qubits[6], qubits[3]);
            CNOT(qubits[8], qubits[7]);
            X(qubits[1]);
            X(qubits[7]);
            CCNOT(qubits[1], qubits[8], qubits[6]);
            CCNOT(qubits[1], qubits[7], qubits[3]);
            X(qubits[1]);
            CNOT(qubits[6], qubits[4]);
            CNOT(qubits[5], qubits[8]);
            CCNOT(qubits[1], qubits[7], qubits[5]);
            CCNOT(qubits[0], qubits[2], qubits[8]);
            X(qubits[0]);
            X(qubits[7]);
            CNOT(qubits[5], qubits[8]);

            ResetAll(qubits);
        }
    }
}
