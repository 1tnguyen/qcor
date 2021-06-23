#pragma once
#include "mlir/Dialect/Affine/Passes.h"
#include "optimizations/IdentityPairRemovalPass.hpp"
#include "optimizations/PermuteGatePass.hpp"
#include "optimizations/RemoveUnusedQIRCallsPass.hpp"
#include "optimizations/RotationMergingPass.hpp"
#include "optimizations/SingleQubitGateMergingPass.hpp"
#include "quantum_to_llvm.hpp"
// Construct QCOR MLIR pass manager:
// Make sure we use the same set of passes and configs
// across different use cases of MLIR compilation.
namespace qcor {
void configureOptimizationPasses(mlir::PassManager &passManager) {
  auto inliner = mlir::createInlinerPass();
  passManager.addPass(std::move(inliner));

  // TODO: This seems to not work (no effect) with non-Affine loops.
  // We may want to rewrite our for-loop handler to
  // use the Affine dialect.
  // auto loop_unroller = mlir::createLoopUnrollPass();
  // // Nest a pass manager that operates on functions within the one which
  // // operates on ModuleOp.
  // OpPassManager &nestedFunctionPM = passManager.nest<mlir::FuncOp>();
  // nestedFunctionPM.addPass(std::move(loop_unroller));

  // TODO: configure the pass pipeline to handle repeated applications of
  // passes. Add passes
  constexpr int N_REPS = 5;
  for (int i = 0; i < N_REPS; ++i) {
    // Simple Identity pair removals
    passManager.addPass(std::make_unique<SingleQubitIdentityPairRemovalPass>());
    passManager.addPass(std::make_unique<CNOTIdentityPairRemovalPass>());

    // Rotation merging
    passManager.addPass(std::make_unique<RotationMergingPass>());
    // General gate sequence re-synthesize
    passManager.addPass(std::make_unique<SingleQubitGateMergingPass>());
    // Try permute gates to realize more merging opportunities
    passManager.addPass(std::make_unique<PermuteGatePass>());
  }

  // Remove dead code
  passManager.addPass(std::make_unique<RemoveUnusedQIRCallsPass>());
}
} // namespace qcor