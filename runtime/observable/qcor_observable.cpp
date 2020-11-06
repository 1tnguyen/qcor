#include "qcor_observable.hpp"

#include "ObservableTransform.hpp"
#include "xacc.hpp"
#include "xacc_quantum_gate_api.hpp"
#include "xacc_service.hpp"

namespace qcor {

PauliOperator operator+(double coeff, PauliOperator &op) {
  return PauliOperator(coeff) + op;
}
PauliOperator operator+(PauliOperator &op, double coeff) {
  return PauliOperator(coeff) + op;
}

PauliOperator operator-(double coeff, PauliOperator &op) {
  return -1.0 * coeff + op;
}

PauliOperator operator-(PauliOperator &op, double coeff) {
  return -1.0 * coeff + op;
}

PauliOperator X(int idx) { return PauliOperator({{idx, "X"}}); }

PauliOperator Y(int idx) { return PauliOperator({{idx, "Y"}}); }

PauliOperator Z(int idx) { return PauliOperator({{idx, "Z"}}); }

PauliOperator allZs(const int nQubits) {
  auto ret = Z(0);
  for (int i = 1; i < nQubits; i++) {
    ret *= Z(i);
  }
  return ret;
}

PauliOperator SP(int idx) {
  std::complex<double> imag(0.0, 1.0);
  return X(idx) + imag * Y(idx);
}

PauliOperator SM(int idx) {
  std::complex<double> imag(0.0, 1.0);
  return X(idx) - imag * Y(idx);
}
std::shared_ptr<xacc::Observable> createObservable(const std::string &repr) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::quantum::getObservable("pauli", repr);
}

std::shared_ptr<Observable> createObservable(const std::string &name,
                                             const std::string &repr) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::quantum::getObservable(name, repr);
}
std::shared_ptr<Observable> createObservable(const std::string &name,
                                             HeterogeneousMap &&options) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::quantum::getObservable(name, options);
}
std::shared_ptr<Observable> createObservable(const std::string &name,
                                             HeterogeneousMap &options) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  return xacc::quantum::getObservable(name, options);
}

std::shared_ptr<Observable> createOperator(const std::string &repr) {
  return createObservable(repr);
}
std::shared_ptr<Observable> createOperator(const std::string &name,
                                           const std::string &repr) {
  return createObservable(name, repr);
}
std::shared_ptr<Observable> createOperator(const std::string &name,
                                           HeterogeneousMap &&options) {
  return createObservable(name, options);
}
std::shared_ptr<Observable> createOperator(const std::string &name,
                                           HeterogeneousMap &options) {
  return createObservable(name, options);
}

std::shared_ptr<Observable> operatorTransform(const std::string &type,
                                              qcor::Observable &op) {
  // return xacc::getService<xacc::ObservableTransform>(type)->transform(
  //     xacc::as_shared_ptr(*&op));
}
std::shared_ptr<Observable> operatorTransform(const std::string &type,
                                              std::shared_ptr<Observable> op) {
  return xacc::getService<xacc::ObservableTransform>(type)->transform(op);
}

namespace __internal__ {
std::vector<std::shared_ptr<xacc::CompositeInstruction>> observe(
    std::shared_ptr<xacc::Observable> obs,
    std::shared_ptr<CompositeInstruction> program) {
  return obs->observe(program);
}
}  // namespace __internal__

double observe(std::shared_ptr<CompositeInstruction> program,
               std::shared_ptr<xacc::Observable> obs,
               xacc::internal_compiler::qreg &q) {
  return [program, obs, &q]() {
    // Observe the program
    auto programs = __internal__::observe(obs, program);

    xacc::internal_compiler::execute(q.results(), programs);

    // We want to contract q children buffer
    // exp-val-zs with obs term coeffs
    return q.weighted_sum(obs.get());
  }();
}

double observe(std::shared_ptr<CompositeInstruction> program, Observable &obs,
               xacc::internal_compiler::qreg &q) {
  return [program, &obs, &q]() {
    // Observe the program
    auto programs = obs.observe(program);

    xacc::internal_compiler::execute(q.results(), programs);

    // We want to contract q children buffer
    // exp-val-zs with obs term coeffs
    return q.weighted_sum(&obs);
  }();
}
}  // namespace qcor