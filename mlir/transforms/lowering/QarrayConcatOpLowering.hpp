#pragma once

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "quantum_to_llvm.hpp"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

namespace qcor {
class QarrayConcatOpLowering : public ConversionPattern {
protected:
  // Constant string for runtime function name
  inline static const std::string qir_qubit_array_concat =
      "__quantum__rt__array_concatenate";
  // Rudimentary symbol table, seen variables
  std::map<std::string, mlir::Value> &variables;

public:
  // Constructor, store seen variables
  explicit QarrayConcatOpLowering(MLIRContext *context,
                                  std::map<std::string, mlir::Value> &vars)
      : ConversionPattern(mlir::quantum::ArrayConcatOp::getOperationName(), 1,
                          context),
        variables(vars) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();
    auto array_qbit_type =
        LLVM::LLVMPointerType::get(get_quantum_type("Array", context));
    FlatSymbolRefAttr symbol_ref;
    if (parentModule.lookupSymbol<LLVM::LLVMFuncOp>(qir_qubit_array_concat)) {
      symbol_ref = SymbolRefAttr::get(qir_qubit_array_concat, context);
    } else {
      // prototype is (%Array*, %Array*) -> %Array*
      auto qconcat_ftype = LLVM::LLVMFunctionType::get(
          array_qbit_type,
          llvm::ArrayRef<Type>{array_qbit_type, array_qbit_type}, false);

      // Insert the function declaration
      PatternRewriter::InsertionGuard insertGuard(rewriter);
      rewriter.setInsertionPointToStart(parentModule.getBody());
      rewriter.create<LLVM::LLVMFuncOp>(parentModule->getLoc(),
                                        qir_qubit_array_concat, qconcat_ftype);
      symbol_ref = mlir::SymbolRefAttr::get(qir_qubit_array_concat, context);
    }

    // Make the call
    auto array_concat_call = rewriter.create<mlir::CallOp>(
        loc, symbol_ref, array_qbit_type, operands);
    // Remove the old QuantumDialect QarrayConcatOp
    rewriter.replaceOp(op, array_concat_call.getResult(0));

    return success();
  }
};
} // namespace qcor