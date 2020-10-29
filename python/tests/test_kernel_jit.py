import unittest
from qcor import *
# Import Python math with alias
import math as myMathMod

# Some global variables for testing
MY_PI = 3.1416
class TestSimpleKernelJIT(unittest.TestCase):
    def test_simple_bell(self):

        set_qpu('qpp', {'shots':8192})

        @qjit
        def bell(q : qreg):
            H(q[0])
            CX(q[0], q[1])
            for i in range(q.size()):
                Measure(q[i])

        # Allocate 2 qubits
        q = qalloc(2)

        # Run the bell experiment
        bell(q)

        # Print the results
        q.print()
        self.assertAlmostEqual(q.exp_val_z(), 1.0, places=1)
        counts = q.counts()
        self.assertEqual(len(counts), 2)
        self.assertTrue('00' in counts)
        self.assertTrue('11' in counts)

    def test_assignment(self):

        set_qpu('qpp', {'shots':8192})

        @qjit
        def varAssignKernel(q : qreg):
            # Simple value assignment
            angle = 3.14/4.0
            Rx(q[0], angle)
            CX(q[0], q[1])
            for i in range(q.size()):
                Measure(q[i])

        # Allocate 2 qubits
        q = qalloc(2)

        # Run the bell experiment
        varAssignKernel(q)

        # Print the results
        q.print()
        counts = q.counts()
        self.assertEqual(len(counts), 2)
        self.assertTrue('00' in counts)
        self.assertTrue('11' in counts)
        # Angle less than Pi/2 => 00 more than 11
        self.assertTrue(counts['00'] > counts['11'])
    
    def test_globalVar(self):

        set_qpu('qpp', {'shots':8192})

        @qjit
        def kernelUseGlobals(q : qreg):
            Rx(q[0], MY_PI)
            CX(q[0], q[1])
            for i in range(q.size()):
                Measure(q[i])

        # Allocate 2 qubits
        q = qalloc(2)

        # Run the bell experiment
        kernelUseGlobals(q)

        # Print the results
        q.print()
        counts = q.counts()
        # Pi pulse -> X gate
        self.assertTrue(counts['11'] > 8000)
    
    def test_ModuleConstants(self):

        set_qpu('qpp', {'shots':8192})

        @qjit
        def kernelUseConstants(q : qreg):
            # Use math.pi constant
            Rx(q[0], myMathMod.pi/2)
            CX(q[0], q[1])
            for i in range(q.size()):
                Measure(q[i])

        # Allocate 2 qubits
        q = qalloc(2)

        # Run the bell experiment
        kernelUseConstants(q)

        # Print the results
        q.print()
        counts = q.counts()
        self.assertEqual(len(counts), 2)
        self.assertTrue('00' in counts)
        self.assertTrue('11' in counts)
    
    def test_exp_i_theta(self):
        @qjit
        def kernelExpVar(q : qreg, theta: float):
            exponent_op = X(0) * Y(1) - Y(0) * X(1)
            exp_i_theta(q, theta, exponent_op)
        
        # Allocate 2 qubits
        q = qalloc(2)
        theta = 1.234
        # Validate exp_i_theta expansion
        comp = kernelExpVar.extract_composite(q, theta)
        self.assertEqual(comp.nInstructions(), 14)

    # Test of edge case where the first statement is a for loop
    def test_for_loop(self):
        @qjit
        def testFor(q : qreg):
            for i in range(q.size()):
                H(q[i])
        
        q = qalloc(5)
        comp = testFor.extract_composite(q)
        self.assertEqual(comp.nInstructions(), 5)   

if __name__ == '__main__':
  unittest.main()