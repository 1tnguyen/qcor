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
        using (qubits = Qubit[5]) {
            H(qubits[0]);
            CNOT(qubits[0], qubits[1]);
            (Adjoint T)(qubits[1]);
            CNOT(qubits[0], qubits[1]);
            T(qubits[0]);
            T(qubits[1]);
            H(qubits[4]);
            CNOT(qubits[2], qubits[0]);
            (Adjoint S)(qubits[2]);
            CNOT(qubits[0], qubits[4]);
            CNOT(qubits[4], qubits[2]);
            T(qubits[2]);
            (Adjoint T)(qubits[4]);
            CNOT(qubits[0], qubits[2]);
            CNOT(qubits[0], qubits[4]);
            T(qubits[2]);
            (Adjoint T)(qubits[4]);
            CNOT(qubits[2], qubits[4]);
            H(qubits[2]);
            T(qubits[2]);
            H(qubits[2]);
            CNOT(qubits[2], qubits[4]);
            (Adjoint T)(qubits[2]);
            T(qubits[4]);
            CNOT(qubits[0], qubits[4]);
            CNOT(qubits[0], qubits[2]);
            T(qubits[4]);
            (Adjoint T)(qubits[2]);
            CNOT(qubits[4], qubits[2]);
            CNOT(qubits[0], qubits[4]);
            S(qubits[2]);
            CNOT(qubits[2], qubits[0]);
            H(qubits[4]);
            CNOT(qubits[0], qubits[4]);
            CCNOT(qubits[3], qubits[4], qubits[0]);
            CNOT(qubits[0], qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            (Adjoint S)(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            H(qubits[4]);
            T(qubits[4]);
            S(qubits[4]);
            CNOT(qubits[0], qubits[4]);
            CCNOT(qubits[3], qubits[4], qubits[0]);
            CNOT(qubits[0], qubits[4]);
            H(qubits[1]);
            CNOT(qubits[1], qubits[2]);
            (Adjoint T)(qubits[2]);
            CNOT(qubits[1], qubits[2]);
            T(qubits[1]);
            T(qubits[2]);
            H(qubits[4]);
            CNOT(qubits[3], qubits[1]);
            (Adjoint S)(qubits[3]);
            CNOT(qubits[1], qubits[4]);
            CNOT(qubits[4], qubits[3]);
            T(qubits[3]);
            (Adjoint T)(qubits[4]);
            CNOT(qubits[1], qubits[3]);
            CNOT(qubits[1], qubits[4]);
            T(qubits[3]);
            (Adjoint T)(qubits[4]);
            CNOT(qubits[3], qubits[4]);
            H(qubits[3]);
            T(qubits[3]);
            H(qubits[3]);
            CNOT(qubits[3], qubits[4]);
            (Adjoint T)(qubits[3]);
            T(qubits[4]);
            CNOT(qubits[1], qubits[4]);
            CNOT(qubits[1], qubits[3]);
            T(qubits[4]);
            (Adjoint T)(qubits[3]);
            CNOT(qubits[4], qubits[3]);
            CNOT(qubits[1], qubits[4]);
            S(qubits[3]);
            CNOT(qubits[3], qubits[1]);
            H(qubits[4]);
            H(qubits[2]);
            CNOT(qubits[2], qubits[3]);
            (Adjoint T)(qubits[3]);
            CNOT(qubits[2], qubits[3]);
            T(qubits[2]);
            T(qubits[3]);

            ResetAll(qubits);
        }
    }
}
