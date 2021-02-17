#include <chrono>

// Trotter evolution
__qpu__ void trotter_evolve(qreg q,
                            std::vector<std::shared_ptr<Observable>> &exp_args,
                            int n_steps) {
  for (int i = 0; i < n_steps; ++i) {
    for (auto &exp_arg : exp_args) {
      exp_i_theta(q, 1.0, exp_arg);
    }
  }
}

int main() {
  // Helper to create general Heisenberg Hamiltonian
  const auto heisenberg_ham = [](int nbQubits) {
    const double Jz = 1.0;
    const double h = 1.0;
    auto H = -h * X(0);
    for (int i = 1; i < nbQubits; ++i) {
      H = H - h * X(i);
    }
    for (int i = 0; i < nbQubits - 1; ++i) {
      H = H - Jz * (Z(i) * Z(i + 1));
    }
    return H;
  };

  const std::vector<int> n_qubits{5, 10, 20, 50, 100};
  const int nbSteps = 100;

  for (const auto &nbQubits : n_qubits) {
    auto ham_op = heisenberg_ham(nbQubits);
    auto op_terms = ham_op.getNonIdentitySubTerms();
    auto q = qalloc(nbQubits);
    auto start = std::chrono::high_resolution_clock::now();
    auto nInsts = trotter_evolve::n_instructions(q, op_terms, nbSteps);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    const double elapsed_secs = (double)duration.count() / 1e6;
    print("n_qubits =", nbQubits, "; n instructions =", nInsts,
          "; Kernel eval time:", elapsed_secs, " [secs]");
  }
}
