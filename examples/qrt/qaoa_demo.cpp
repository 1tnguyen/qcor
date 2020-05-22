#include "qcor.hpp"
#include <random>

// Use QAOA algorithm to solve a QUBO problem
// QUBO function:
// y = -5x1 - 3x2 - 8x3 - 6x4 + 4x1x2 + 8x1x3 + 2x2x3 + 10x3x4
// Adapted from https://docs.entropicalabs.io/qaoa/notebooks/6_solvingqubowithqaoa
// Instructions:
// Compile: qcor -o qaoa -qpu qpp qaoa_demo.cpp 
// Run: ./qaoa
int main(int argc, char **argv) {
  // Create corresponding QUBO Hamiltonian (cost Observable)
  auto observable = qcor::createObservable(
      "-5.0 - 0.5 Z0 - 1.0 Z2 + 0.5 Z3 + 1.0 Z0 Z1 + 2.0 Z0 Z2 + 0.5 Z1 Z2 + 2.5 Z2 Z3");
  
  // Allocate 4 qubits (number of variables)
  auto q = qalloc(4);
  
  // QAOA 'p' steps:
  const int nbSteps = 6;
  // We have 4 params for the mixer Hamiltonian terms (one for each qubit)
  // and 7 terms for the above cost Hamiltonian.
  // i.e. 11 params/step.
  const int nbParams = nbSteps*11;
  // We need to seed the 'nlopt' optimizer with random numbers
  // to prevent it from being stuck.
  std::vector<double> initialParams;
  std::random_device rd;  
  std::mt19937 gen(rd()); 
  std::uniform_real_distribution<> dis(-2.0, 2.0);
  // Init random parameters
  for (int i = 0; i < nbParams; ++i) {
    initialParams.emplace_back(dis(gen));
  } 
  
  // Create the Optimizer
  auto optimizer = qcor::createOptimizer("nlopt", xacc::HeterogeneousMap{ 
    std::make_pair("initial-parameters", initialParams),
      // Scale the number of iters by the number of params 
      // to guarantee convergence.
      std::make_pair("nlopt-maxeval", nbParams*100)
  });

  auto algorithm = qcor::createAlgorithm("QAOA", {
                        std::make_pair("optimizer", optimizer),
                        std::make_pair("observable", observable),
                        std::make_pair("accelerator", get_qpu()),
                        // number of time steps (p) param
                        std::make_pair("steps", nbSteps)
                      });

  // Call taskInitiate, kick off QAOA optimization run 
  // on the backend (Async call).
  auto handle = qcor::taskInitiate(algorithm, q);

  // Go do other work...

  // Query results when ready.
  auto results = qcor::sync(handle);

  // Print the optimal value.
  printf("<Min QUBO> = %f\n", results.opt_val);
}
