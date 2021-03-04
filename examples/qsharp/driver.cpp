#include <iostream> 
#include <vector>
#include "qcor.hpp"

// Include the external QSharp function.
qcor_include_qsharp(XACC__TestBell__body, int64_t, int64_t)


// Compile with:
// Include both the qsharp source and this driver file 
// in the command line.
// $ qcor source.qs driver.cpp
// Run with:
// $ ./a.out
int main() {
  std::cout << "HOWDY \n";
  auto oneCounts = XACC__TestBell__body(1024);
  std::cout << "Result = " << oneCounts << "\n";
  return 0.0;
}