#include "ModifierBlockInliner.hpp"
#include "Quantum/QuantumOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include <iostream>

namespace qcor {
void ModifierBlockInlinerPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

void ModifierBlockInlinerPass::runOnOperation() {
  // Power U to loop...
  std::vector<Operation *> deadOps;
  getOperation().walk([&](mlir::quantum::PowURegion op) {
    mlir::OpBuilder rewriter(op);
    mlir::Value powVal = [&]() -> mlir::Value {
      if (op.pow().getType().isIndex()) {
        return op.pow();
      }
      return rewriter.create<mlir::IndexCastOp>(
          op.getLoc(), rewriter.getIndexType(), op.pow());
    }();
    mlir::Value lbs_val = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::IntegerAttr::get(rewriter.getIndexType(), 0));
    mlir::Value step_val = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), mlir::IntegerAttr::get(rewriter.getIndexType(), 1));
    mlir::Block &powBlock = op.body().getBlocks().front();
    // Convert the pow modifier to a For loop,
    // which might be unrolled if possible (constant-value loop bound)
    rewriter.create<mlir::scf::ForOp>(
        op.getLoc(), lbs_val, powVal, step_val, llvm::None,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
            mlir::Value iv, mlir::ValueRange itrArgs) {
          mlir::OpBuilder::InsertionGuard guard(nestedBuilder);
          for (auto &subOp : powBlock.getOperations()) {
            if (mlir::dyn_cast_or_null<mlir::quantum::ModifierEndOp>(&subOp)) {
              break;
            }
            nestedBuilder.insert(subOp.clone());
          }

          nestedBuilder.create<mlir::scf::YieldOp>(nestedLoc);
        });
    op.body().getBlocks().clear();
    deadOps.emplace_back(op.getOperation());
  });
  for (auto &op : deadOps) {
    op->dropAllUses();
    op->erase();
  }
  deadOps.clear();
}
} // namespace qcor
