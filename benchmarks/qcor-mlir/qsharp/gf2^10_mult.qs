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
        using (qubits = Qubit[30]) {
            CCNOT(qubits[9], qubits[11], qubits[20]);
            CCNOT(qubits[8], qubits[12], qubits[20]);
            CCNOT(qubits[7], qubits[13], qubits[20]);
            CCNOT(qubits[6], qubits[14], qubits[20]);
            CCNOT(qubits[5], qubits[15], qubits[20]);
            CCNOT(qubits[4], qubits[16], qubits[20]);
            CCNOT(qubits[3], qubits[17], qubits[20]);
            CCNOT(qubits[2], qubits[18], qubits[20]);
            CCNOT(qubits[1], qubits[19], qubits[20]);
            CCNOT(qubits[9], qubits[12], qubits[21]);
            CCNOT(qubits[8], qubits[13], qubits[21]);
            CCNOT(qubits[7], qubits[14], qubits[21]);
            CCNOT(qubits[6], qubits[15], qubits[21]);
            CCNOT(qubits[5], qubits[16], qubits[21]);
            CCNOT(qubits[4], qubits[17], qubits[21]);
            CCNOT(qubits[3], qubits[18], qubits[21]);
            CCNOT(qubits[2], qubits[19], qubits[21]);
            CCNOT(qubits[9], qubits[13], qubits[22]);
            CCNOT(qubits[8], qubits[14], qubits[22]);
            CCNOT(qubits[7], qubits[15], qubits[22]);
            CCNOT(qubits[6], qubits[16], qubits[22]);
            CCNOT(qubits[5], qubits[17], qubits[22]);
            CCNOT(qubits[4], qubits[18], qubits[22]);
            CCNOT(qubits[3], qubits[19], qubits[22]);
            CCNOT(qubits[9], qubits[14], qubits[23]);
            CCNOT(qubits[8], qubits[15], qubits[23]);
            CCNOT(qubits[7], qubits[16], qubits[23]);
            CCNOT(qubits[6], qubits[17], qubits[23]);
            CCNOT(qubits[5], qubits[18], qubits[23]);
            CCNOT(qubits[4], qubits[19], qubits[23]);
            CCNOT(qubits[9], qubits[15], qubits[24]);
            CCNOT(qubits[8], qubits[16], qubits[24]);
            CCNOT(qubits[7], qubits[17], qubits[24]);
            CCNOT(qubits[6], qubits[18], qubits[24]);
            CCNOT(qubits[5], qubits[19], qubits[24]);
            CCNOT(qubits[9], qubits[16], qubits[25]);
            CCNOT(qubits[8], qubits[17], qubits[25]);
            CCNOT(qubits[7], qubits[18], qubits[25]);
            CCNOT(qubits[6], qubits[19], qubits[25]);
            CCNOT(qubits[9], qubits[17], qubits[26]);
            CCNOT(qubits[8], qubits[18], qubits[26]);
            CCNOT(qubits[7], qubits[19], qubits[26]);
            CCNOT(qubits[9], qubits[18], qubits[27]);
            CCNOT(qubits[8], qubits[19], qubits[27]);
            CCNOT(qubits[9], qubits[19], qubits[28]);
            CNOT(qubits[28], qubits[21]);
            CNOT(qubits[27], qubits[20]);
            CNOT(qubits[26], qubits[29]);
            CNOT(qubits[25], qubits[28]);
            CNOT(qubits[24], qubits[27]);
            CNOT(qubits[23], qubits[26]);
            CNOT(qubits[22], qubits[25]);
            CNOT(qubits[21], qubits[24]);
            CNOT(qubits[20], qubits[23]);
            CCNOT(qubits[9], qubits[10], qubits[29]);
            CCNOT(qubits[8], qubits[11], qubits[29]);
            CCNOT(qubits[7], qubits[12], qubits[29]);
            CCNOT(qubits[6], qubits[13], qubits[29]);
            CCNOT(qubits[5], qubits[14], qubits[29]);
            CCNOT(qubits[4], qubits[15], qubits[29]);
            CCNOT(qubits[3], qubits[16], qubits[29]);
            CCNOT(qubits[2], qubits[17], qubits[29]);
            CCNOT(qubits[1], qubits[18], qubits[29]);
            CCNOT(qubits[0], qubits[19], qubits[29]);
            CCNOT(qubits[8], qubits[10], qubits[28]);
            CCNOT(qubits[7], qubits[11], qubits[28]);
            CCNOT(qubits[6], qubits[12], qubits[28]);
            CCNOT(qubits[5], qubits[13], qubits[28]);
            CCNOT(qubits[4], qubits[14], qubits[28]);
            CCNOT(qubits[3], qubits[15], qubits[28]);
            CCNOT(qubits[2], qubits[16], qubits[28]);
            CCNOT(qubits[1], qubits[17], qubits[28]);
            CCNOT(qubits[0], qubits[18], qubits[28]);
            CCNOT(qubits[7], qubits[10], qubits[27]);
            CCNOT(qubits[6], qubits[11], qubits[27]);
            CCNOT(qubits[5], qubits[12], qubits[27]);
            CCNOT(qubits[4], qubits[13], qubits[27]);
            CCNOT(qubits[3], qubits[14], qubits[27]);
            CCNOT(qubits[2], qubits[15], qubits[27]);
            CCNOT(qubits[1], qubits[16], qubits[27]);
            CCNOT(qubits[0], qubits[17], qubits[27]);
            CCNOT(qubits[6], qubits[10], qubits[26]);
            CCNOT(qubits[5], qubits[11], qubits[26]);
            CCNOT(qubits[4], qubits[12], qubits[26]);
            CCNOT(qubits[3], qubits[13], qubits[26]);
            CCNOT(qubits[2], qubits[14], qubits[26]);
            CCNOT(qubits[1], qubits[15], qubits[26]);
            CCNOT(qubits[0], qubits[16], qubits[26]);
            CCNOT(qubits[5], qubits[10], qubits[25]);
            CCNOT(qubits[4], qubits[11], qubits[25]);
            CCNOT(qubits[3], qubits[12], qubits[25]);
            CCNOT(qubits[2], qubits[13], qubits[25]);
            CCNOT(qubits[1], qubits[14], qubits[25]);
            CCNOT(qubits[0], qubits[15], qubits[25]);
            CCNOT(qubits[4], qubits[10], qubits[24]);
            CCNOT(qubits[3], qubits[11], qubits[24]);
            CCNOT(qubits[2], qubits[12], qubits[24]);
            CCNOT(qubits[1], qubits[13], qubits[24]);
            CCNOT(qubits[0], qubits[14], qubits[24]);
            CCNOT(qubits[3], qubits[10], qubits[23]);
            CCNOT(qubits[2], qubits[11], qubits[23]);
            CCNOT(qubits[1], qubits[12], qubits[23]);
            CCNOT(qubits[0], qubits[13], qubits[23]);
            CCNOT(qubits[2], qubits[10], qubits[22]);
            CCNOT(qubits[1], qubits[11], qubits[22]);
            CCNOT(qubits[0], qubits[12], qubits[22]);
            CCNOT(qubits[1], qubits[10], qubits[21]);
            CCNOT(qubits[0], qubits[11], qubits[21]);
            CCNOT(qubits[0], qubits[10], qubits[20]);

            ResetAll(qubits);
        }
    }
}
