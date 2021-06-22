#include "qcor_optimizer.hpp"

#include "Optimizer.hpp"
#include "objective_function.hpp"
#include "qcor_pimpl_impl.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"

namespace qcor {

/// ------------- Optimizer Wrapper ---------------
Optimizer::Optimizer() = default;
Optimizer::Optimizer(std::shared_ptr<xacc::Identifiable> generic)
    : m_internal(std::dynamic_pointer_cast<xacc::Optimizer>(generic)) {}
Optimizer::~Optimizer() = default;

struct Optimizer::OptimizerImpl {
  std::shared_ptr<xacc::Optimizer> xacc_opt;

  OptimizerImpl() = default;
  OptimizerImpl(std::shared_ptr<xacc::Optimizer> opt) : xacc_opt(opt) {}

  std::pair<double, std::vector<double>> optimize(xacc::OptFunction &opt) {
    return xacc_opt->optimize(opt);
  }

  xacc::Optimizer *operator->() { return xacc_opt.get(); }
};

std::pair<double, std::vector<double>> Optimizer::optimize(
    std::function<double(const std::vector<double> &)> opt, const int dim) {
  xacc::OptFunction opt_(opt, dim);
  return m_internal->optimize(opt_);
}

std::pair<double, std::vector<double>> Optimizer::optimize(
    std::function<double(const std::vector<double> &, std::vector<double> &)>
        opt,
    const int dim) {
  xacc::OptFunction opt_(opt, dim);
  return m_internal->optimize(opt_);
}

std::pair<double, std::vector<double>> Optimizer::optimize(
    std::shared_ptr<ObjectiveFunction> obj) {
  xacc::OptFunction opt_(
      [obj](const std::vector<double> &x, std::vector<double> &dx) {
        return (*obj)(x, dx);
      },
      obj->dimensions());
  return m_internal->optimize(opt_);
}

std::pair<double, std::vector<double>> Optimizer::optimize(
    ObjectiveFunction *obj) {
  xacc::OptFunction opt_(
      [obj](const std::vector<double> &x, std::vector<double> &dx) {
        return (*obj)(x, dx);
      },
      obj->dimensions());
  return m_internal->optimize(opt_);
}

std::pair<double, std::vector<double>> Optimizer::optimize(
    ObjectiveFunction &obj) {
  xacc::OptFunction opt_([&](const std::vector<double> &x,
                             std::vector<double> &dx) { return obj(x, dx); },
                         obj.dimensions());
  return m_internal->optimize(opt_);
}
Optimizer::OptimizerImpl *Optimizer::operator->() {
  return m_internal.operator->();
}

std::shared_ptr<Optimizer> createOptimizer(const std::string &type,
                                           xacc::HeterogeneousMap &&options) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  auto xacc_opt = xacc::getOptimizer(type, options);

  return std::make_shared<Optimizer>(xacc_opt);
}

}  // namespace qcor