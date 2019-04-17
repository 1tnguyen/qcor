#include "exp.hpp"
#include "IRProvider.hpp"
#include "PauliOperator.hpp"
#include "XACC.hpp"
#include "xacc_service.hpp"
#include <regex>

using namespace xacc;
using namespace xacc::quantum;

namespace qcor {
namespace instructions {
bool Exp::validateOptions() {
  if (!options.count("pauli")) {
    return false;
  }
  return true;
}

std::shared_ptr<Function>
Exp::generate(std::shared_ptr<AcceleratorBuffer> buffer,
              std::vector<InstructionParameter> parameters) {
  xacc::error("qcor::Exp::generate(buffer,params) not implemented.");
  return nullptr;
}

std::shared_ptr<xacc::Function>
Exp::generate(std::map<std::string, xacc::InstructionParameter> &parameters) {
  if (!parameters.empty()) {
    options = parameters;
  }
  return generate(std::map<std::string, InstructionParameter>{});
}

std::shared_ptr<Function>
Exp::generate(std::map<std::string, InstructionParameter> &&parameters) {

  if (!parameters.empty()) {
    options = parameters;
  }

  std::string paramLetter = "t";
  if (options.count("param-id")) {
      paramLetter = options["param-id"].toString();
  }
  auto pauliStr = options["pauli"].toString();
  PauliOperator op(pauliStr);

  double pi = 3.1415926;

  auto gateRegistry = xacc::getService<IRProvider>("gate");
  auto function = gateRegistry->createFunction("temp", {}, {});

  auto terms = op.getTerms();
  for (auto spinInst : terms) {

    // Get the individual pauli terms
    auto termsMap = std::get<2>(spinInst.second);

    std::vector<std::pair<int, std::string>> terms;
    for (auto &kv : termsMap) {
      if (kv.second != "I" && !kv.second.empty()) {
        terms.push_back({kv.first, kv.second});
      }
    }
    // The largest qubit index is on the last term
    int largestQbitIdx = terms[terms.size() - 1].first;

    for (int i = 0; i < terms.size(); i++) {

      auto qbitIdx = terms[i].first;
      auto gateName = terms[i].second;

      if (i < terms.size() - 1) {
        auto cnot = gateRegistry->createInstruction(
            "CNOT", std::vector<int>{qbitIdx, terms[i + 1].first});
        function->addInstruction(cnot);
      }

      if (gateName == "X") {
        auto hadamard =
            gateRegistry->createInstruction("H", std::vector<int>{qbitIdx});
        function->insertInstruction(0, hadamard);
      } else if (gateName == "Y") {
        auto rx =
            gateRegistry->createInstruction("Rx", std::vector<int>{qbitIdx});
        InstructionParameter p(pi / 2.0);
        rx->setParameter(0, p);
        function->insertInstruction(0, rx);
      }

      // Add the Rotation for the last term
      if (i == terms.size() - 1) {
        // FIXME DONT FORGET DIVIDE BY 2
        // std::stringstream ss;
        // ss << 2 * std::imag(std::get<0>(spinInst.second)) << " * "
        //    << std::get<1>(spinInst.second);
        auto rz =
            gateRegistry->createInstruction("Rz", std::vector<int>{qbitIdx}, {paramLetter});

        // InstructionParameter p(ss.str());
        // rz->setParameter(0, p);
        function->addInstruction(rz);
      }
    }

    int counter = function->nInstructions();
    // Add the instruction on the backend of the circuit
    for (int i = terms.size() - 1; i >= 0; i--) {

      auto qbitIdx = terms[i].first;
      auto gateName = terms[i].second;

      if (i < terms.size() - 1) {
        auto cnot = gateRegistry->createInstruction(
            "CNOT", std::vector<int>{qbitIdx, terms[i + 1].first});
        function->insertInstruction(counter, cnot);
        counter++;
      }

      if (gateName == "X") {
        auto hadamard =
            gateRegistry->createInstruction("H", std::vector<int>{qbitIdx});
        function->addInstruction(hadamard);
      } else if (gateName == "Y") {
        auto rx =
            gateRegistry->createInstruction("Rx", std::vector<int>{qbitIdx});
        InstructionParameter p(-1.0 * (pi / 2.0));
        rx->setParameter(0, p);
        function->addInstruction(rx);
      }
    }
  }

  return function;
} // namespace instructions

} // namespace instructions
} // namespace qcor