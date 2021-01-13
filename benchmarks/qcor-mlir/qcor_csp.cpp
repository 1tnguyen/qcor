// Dummy file to compile OpenQASM using qcor in syntax handler mode.
#include "qcor.hpp"

#ifdef TEST_SOURCE_FILE
__qpu__ void testKernel(qreg quVar) {
  using qcor::openqasm;
#include TEST_SOURCE_FILE
}

int main() {
  return 0;
}
#endif