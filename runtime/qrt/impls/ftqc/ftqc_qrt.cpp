
#include <Eigen/Dense>
#include <Utils.hpp>

#include "PauliOperator.hpp"
#include "cppmicroservices/BundleActivator.h"
#include "cppmicroservices/BundleContext.h"
#include "cppmicroservices/ServiceProperties.h"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"
using namespace cppmicroservices;

namespace {
class FtqcQubitAllocator : public AllocEventListener, public QubitAllocator {
public:
  static inline const std::string ANC_BUFFER_NAME = "ftqc_temp_buffer";
  virtual void onAllocate(qubit *in_qubit) override {
    // std::cout << "Allocate: " << (void *)in_qubit << "\n";
  }

  // On deallocate: don't try to deref the qubit since it may have been gone.
  virtual void onDealloc(qubit *in_qubit) override {
    // std::cout << "Deallocate: " << (void *)in_qubit << "\n";
    // If this qubit was allocated from this pool:
    if (xacc::container::contains(m_allocatedQubits, in_qubit)) {
      const auto qIndex = std::find(m_allocatedQubits.begin(),
                                    m_allocatedQubits.end(), in_qubit) -
                          m_allocatedQubits.begin();
      // Strategy: create a storage copy of the returned qubit:
      // i.e. with the same index w.r.t. this global anc. buffer
      // but store it in the pool vector -> will stay alive
      // until giving out at the next allocate()
      qubit archive_qubit(ANC_BUFFER_NAME, qIndex, m_buffer.get());
      m_allocatedQubits[qIndex] = &archive_qubit;
      m_qubitPool.emplace_back(archive_qubit);
    }
  }

  virtual qubit allocate() override {
    if (!m_qubitPool.empty()) {
      auto recycled_qubit = m_qubitPool.back();
      m_qubitPool.pop_back();
      return recycled_qubit;
    }
    if (!m_buffer) {
      // This must be the first call.
      assert(m_allocatedQubits.empty());
      m_buffer = xacc::qalloc(1);
      m_buffer->setName(ANC_BUFFER_NAME);
    }

    // Need to allocate new qubit:
    // Each new qubit will have an incrementing index.
    const auto newIdx = m_allocatedQubits.size();
    qubit new_qubit(ANC_BUFFER_NAME, newIdx, m_buffer.get());
    // Just track that we allocated this qubit
    m_allocatedQubits.emplace_back(&new_qubit);
    m_buffer->setSize(m_allocatedQubits.size());
    return new_qubit;
  }

  static FtqcQubitAllocator *getInstance() {
    if (!g_instance) {
      g_instance = new FtqcQubitAllocator();
    }
    return g_instance;
  }
  static FtqcQubitAllocator *g_instance;

  std::shared_ptr<xacc::AcceleratorBuffer> get_buffer() { return m_buffer; }

private:
  std::vector<qubit> m_qubitPool;
  // Track the list of qubit pointers for those
  // that was allocated by this Allocator.
  std::vector<qubit *> m_allocatedQubits;
  std::shared_ptr<xacc::AcceleratorBuffer> m_buffer;
};

FtqcQubitAllocator *FtqcQubitAllocator::g_instance = nullptr;
} // namespace

namespace qcor {
class FTQC : public quantum::QuantumRuntime {
 public:
  virtual void initialize(const std::string kernel_name) override {
    provider = xacc::getIRProvider("quantum");
    qpu = xacc::internal_compiler::qpu;
    qubitIdToGlobalIdx.clear();
    setGlobalQubitManager(FtqcQubitAllocator::getInstance());
  }

  void __begin_mark_segment_as_compute() override { mark_as_compute = true; }
  void __end_mark_segment_as_compute() override { mark_as_compute = false; }
  bool isComputeSection() override { return mark_as_compute; }
  const std::string name() const override { return "ftqc"; }
  const std::string description() const override { return ""; }

  virtual void h(const qubit &qidx) override { applyGate("H", {qidx}); }
  virtual void x(const qubit &qidx) override { applyGate("X", {qidx}); }
  virtual void y(const qubit &qidx) override { applyGate("Y", {qidx}); }
  virtual void z(const qubit &qidx) override { applyGate("Z", {qidx}); }
  virtual void t(const qubit &qidx) override { applyGate("T", {qidx}); }
  virtual void tdg(const qubit &qidx) override {
    applyGate("Tdg", {qidx});
  }
  virtual void s(const qubit &qidx) override { applyGate("S", {qidx}); }
  virtual void sdg(const qubit &qidx) override {
    applyGate("Sdg", {qidx});
  }

  // Common single-qubit, parameterized instructions
  virtual void rx(const qubit &qidx, const double theta) override {
    applyGate("Rx", {qidx}, {theta});
  }
  virtual void ry(const qubit &qidx, const double theta) override {
    applyGate("Ry", {qidx}, {theta});
  }
  virtual void rz(const qubit &qidx, const double theta) override {
    applyGate("Rz", {qidx}, {theta});
  }
  // U1(theta) gate
  virtual void u1(const qubit &qidx, const double theta) override {
    applyGate("U1", {qidx}, {theta});
  }
  virtual void u3(const qubit &qidx, const double theta, const double phi,
                  const double lambda) override {
    applyGate("U", {qidx}, {theta, phi, lambda});
  }

  virtual void reset(const qubit &qidx) override {
    applyGate("Reset", {qidx});
  }

  // Measure-Z
  virtual bool mz(const qubit &qidx) override {
    applyGate("Measure", {qidx});
    // Return the measure result stored in the q reg.
    return (*qReg)[qidx.second];
  }

  // Common two-qubit gates.
  virtual void cnot(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("CNOT", {src_idx, tgt_idx});
  }
  virtual void cy(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("CY", {src_idx, tgt_idx});
  }
  virtual void cz(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("CZ", {src_idx, tgt_idx});
  }
  virtual void ch(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("CH", {src_idx, tgt_idx});
  }
  virtual void swap(const qubit &src_idx, const qubit &tgt_idx) override {
    applyGate("Swap", {src_idx, tgt_idx});
  }

  // Common parameterized 2 qubit gates.
  virtual void cphase(const qubit &src_idx, const qubit &tgt_idx,
                      const double theta) override {
    applyGate("CPhase", {src_idx, tgt_idx}, {theta});
  }
  virtual void crz(const qubit &src_idx, const qubit &tgt_idx,
                   const double theta) override {
    applyGate("CRZ", {src_idx, tgt_idx}, {theta});
  }

  // exponential of i * theta * H, where H is an Observable pointer
  virtual void exp(qreg q, const double theta,
                   xacc::Observable &H) override { /* TODO */
  }
  virtual void exp(qreg q, const double theta,
                   xacc::Observable *H) override { /* TODO */
  }
  virtual void exp(qreg q, const double theta,
                   std::shared_ptr<xacc::Observable> H) override { /* TODO */
  }

  // Submission API: sanity check that we don't call these API's.
  // e.g. catch high-level code gen errors.
  virtual void submit(xacc::AcceleratorBuffer *buffer) override {
    throw std::runtime_error("FTQC runtime doesn't support submit API.");
  }
  virtual void submit(xacc::AcceleratorBuffer **buffers,
                      const int nBuffers) override {
    throw std::runtime_error("FTQC runtime doesn't support submit API.");
  }

  void general_instruction(std::shared_ptr<xacc::Instruction> inst) override {
    std::vector<double> params;
    for (auto p : inst->getParameters()) {
      params.push_back(p.as<double>());
    }
    applyGate(inst->name(), inst->bits(), params);
  }

  // Some getters for the qcor runtime library.
  virtual void set_current_program(
      std::shared_ptr<xacc::CompositeInstruction> p) override {
    // Nothing to do
  }
  virtual std::shared_ptr<xacc::CompositeInstruction> get_current_program()
      override {
    return nullptr;
  }

  void set_current_buffer(xacc::AcceleratorBuffer *buffer) override {
    qReg = xacc::as_shared_ptr(buffer);
    qubitIdToGlobalIdx.clear();
    // The base qreg will always have exact address in the global register.
    for (size_t i = 0; i < qReg->size(); ++i) {
      qubitIdToGlobalIdx[std::make_pair(qReg->name(), i)] = i;
    }
  }

  QubitAllocator *get_anc_qubit_allocator() {
    return FtqcQubitAllocator::getInstance();
  }

 private:
  // Notes: all gate parameters must be resolved (to double) for FT-QRT
  // execution.
  void applyGate(const std::string &gateName, const std::vector<size_t> &bits,
                 const std::vector<double> &params = {}) {
    std::vector<xacc::InstructionParameter> instParams;
    for (const auto &val : params) {
      instParams.emplace_back(val);
    }
    auto gateInst = provider->createInstruction(gateName, bits, instParams);
    if (mark_as_compute) {
      gateInst->attachMetadata({{"__qcor__compute__segment__", true}});
    }
    qpu->apply(qReg, gateInst);
  }

  void applyGate(const std::string &gateName,
                 std::initializer_list<size_t> bits,
                 const std::vector<double> &params = {}) {
    applyGate(gateName, std::vector<size_t>(bits), params);
  }

  void applyGate(const std::string &gateName, const std::vector<qubit> &qbits,
                 const std::vector<double> &params = {}) {
    std::vector<xacc::InstructionParameter> instParams;
    for (const auto &val : params) {
      instParams.emplace_back(val);
    }
    std::vector<size_t> bits;
    for (const auto &qb : qbits) {
      // Never seen this qubit
      const auto qubitId = std::make_pair(qb.first, qb.second);
      if (qubitIdToGlobalIdx.find(qubitId) == qubitIdToGlobalIdx.end()) {
        qubitIdToGlobalIdx[qubitId] = qubitIdToGlobalIdx.size();
        std::stringstream logss;
        logss << "Map " << qb.first << "[" << qb.second << "] to global ID "
              << qubitIdToGlobalIdx[qubitId];
        xacc::info(logss.str());
        qReg->setSize(qubitIdToGlobalIdx.size());
      }
      bits.emplace_back(qubitIdToGlobalIdx[qubitId]);
    }
    auto gateInst = provider->createInstruction(gateName, bits, instParams);
    if (mark_as_compute) {
      gateInst->attachMetadata({{"__qcor__compute__segment__", true}});
    }
    qpu->apply(qReg, gateInst);
  }

 private:
  bool mark_as_compute = false;
  std::shared_ptr<xacc::IRProvider> provider;
  std::shared_ptr<xacc::Accelerator> qpu;
  // TODO: eventually, we may want to support an arbitrary number of qubit
  // registers when the FTQC backend can support it.
  std::shared_ptr<xacc::AcceleratorBuffer> qReg;
  std::map<std::pair<std::string, size_t>, size_t> qubitIdToGlobalIdx;
};
}  // namespace qcor

namespace {
class US_ABI_LOCAL FtqcQRTActivator : public BundleActivator {
 public:
  FtqcQRTActivator() {}
  void Start(BundleContext context) {
    auto xt = std::make_shared<qcor::FTQC>();
    context.RegisterService<quantum::QuantumRuntime>(xt);
  }
  void Stop(BundleContext /*context*/) {}
};
}  // namespace

CPPMICROSERVICES_EXPORT_BUNDLE_ACTIVATOR(FtqcQRTActivator)
