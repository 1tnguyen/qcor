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
        using (qubits = Qubit[19]) {
            CCNOT(qubits[0], qubits[1], qubits[10]);
            CCNOT(qubits[2], qubits[10], qubits[11]);
            CCNOT(qubits[3], qubits[11], qubits[12]);
            CCNOT(qubits[4], qubits[12], qubits[13]);
            CCNOT(qubits[5], qubits[13], qubits[14]);
            CCNOT(qubits[6], qubits[14], qubits[15]);
            CCNOT(qubits[7], qubits[15], qubits[16]);
            CCNOT(qubits[8], qubits[16], qubits[17]);
            CCNOT(qubits[9], qubits[17], qubits[18]);
            CCNOT(qubits[8], qubits[16], qubits[17]);
            CCNOT(qubits[7], qubits[15], qubits[16]);
            CCNOT(qubits[6], qubits[14], qubits[15]);
            CCNOT(qubits[5], qubits[13], qubits[14]);
            CCNOT(qubits[4], qubits[12], qubits[13]);
            CCNOT(qubits[3], qubits[11], qubits[12]);
            CCNOT(qubits[2], qubits[10], qubits[11]);
            CCNOT(qubits[0], qubits[1], qubits[10]);

            ResetAll(qubits);
        }
    }
}
