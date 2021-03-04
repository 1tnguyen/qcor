#include <iostream> 
#include <vector>
#include "qcor.hpp"

// Include the external QSharp function.
qcor_include_qsharp(XACC__Deuteron__body, double, double, int64_t);

// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// $ qcor vqe_ansatz.qs vqe_driver.cpp
// Run with:
// $ ./a.out
int main() {
  // Manually intialize QRT...
  ::quantum::set_qrt("ftqc");
  ::quantum::initialize("qpp", "test");

  const std::vector<double> expectedResults{
      0.0,       -0.324699, -0.614213, -0.837166, -0.9694,
      -0.996584, -0.915773, -0.735724, -0.475947, -0.164595,
      0.164595,  0.475947,  0.735724,  0.915773,  0.996584,
      0.9694,    0.837166,  0.614213,  0.324699,  0.0};

  const auto angles = qcor::linspace(-M_PI, M_PI, 20);
  for (size_t i = 0; i < angles.size(); ++i) {

    const double angle = angles[i];
    const double exp_val_xx = XACC__Deuteron__body(angle, 1024);
    std::cout << "<X0X1>(" << angle << ") = " << exp_val_xx
              << " vs. expected = " << expectedResults[i] << "\n";
  }
  return 0;
}