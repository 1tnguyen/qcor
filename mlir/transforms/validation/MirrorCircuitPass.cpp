#include "MirrorCircuitPass.hpp"
#include "Quantum/QuantumOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
void MirrorCircuitTransformPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect>();
}

void MirrorCircuitTransformPass::runOnOperation() {
  std::cout << "MirrorCircuitTransformPass:\n";
  getOperation().dump();
  
  
  getOperation().walk([&](mlir::FuncOp funcOp) {
    if (funcOp.getName().str().rfind("__internal_mlir_", 0) == 0) {
      mlir::OpBuilder builder(funcOp);

      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        auto parentModule = funcOp->getParentOfType<mlir::ModuleOp>();
        builder.setInsertionPointToStart(
            &parentModule.getRegion().getBlocks().front());

        auto func_decl = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(), "__quantum__rt__start_validated_region",
            builder.getFunctionType(llvm::None, llvm::None));
        func_decl.setVisibility(mlir::SymbolTable::Visibility::Private);

        mlir::Block &fnBody = funcOp.getBody().getBlocks().front();
        builder.setInsertionPointToStart(&fnBody);
        builder.create<mlir::CallOp>(builder.getUnknownLoc(), func_decl);
      }

      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        auto parentModule = funcOp->getParentOfType<mlir::ModuleOp>();
        builder.setInsertionPointToStart(
            &parentModule.getRegion().getBlocks().front());

        auto func_decl = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(), "__quantum__rt__end_validated_region",
            builder.getFunctionType(llvm::None, llvm::None));
        func_decl.setVisibility(mlir::SymbolTable::Visibility::Private);
        mlir::Block &fnBody = funcOp.getBody().getBlocks().back();
        builder.setInsertionPoint(fnBody.getTerminator());
        builder.create<mlir::CallOp>(builder.getUnknownLoc(), func_decl);
      }
    }
  });
  std::cout << "After MirrorCircuitTransformPass:\n";
  getOperation().dump();
}
} // namespace qcor