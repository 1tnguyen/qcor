namespace XACC 
{
open Microsoft.Quantum.Intrinsic;
operation TestBell(count : Int) : Int {
    // Simple bell test
    mutable numOnes = 0;
    mutable agree = 0;
    use q = Qubit[2];
    for test in 1..count {
        H(q[0]);
        CNOT(q[0],q[1]);
        let res0 = M(q[0]);
        let res1 = M(q[1]);
        if res0 == res0 {
            set agree += 1;
        }

        // Count the number of ones we saw:
        if res0 == One {
            set numOnes += 1;
            // Reset
            X(q[0]);
        }
        if res1 == One {
            // Reset
            X(q[0]);
        }
    }
    return numOnes;
}
}