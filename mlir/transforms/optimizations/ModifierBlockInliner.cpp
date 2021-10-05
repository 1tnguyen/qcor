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

void ModifierBlockInlinerPass::handlePowU() {
  // Power U to loop...
  std::vector<Operation *> deadOps;

  getOperation().walk([&](mlir::quantum::PowURegion op) {
    // Must be a single-block op
    assert(op.body().getBlocks().size() == 1);
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
void ModifierBlockInlinerPass::handleCtrlU() {
  std::function<bool(mlir::Operation &)> is_quantum_op =
      [&is_quantum_op](mlir::Operation &opToCheck) -> bool {
    if (mlir::dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(
            &opToCheck)) {
      return true;
    }
    if (opToCheck.getNumRegions() > 0) {
      for (auto &subRegion : opToCheck.getRegions()) {
        for (auto &subBlock : subRegion.getBlocks()) {
          for (auto &subOp : subBlock.getOperations()) {
            // Recurse
            if (is_quantum_op(subOp)) {
              return true;
            }
          }
        }
      }
    }

    return false;
  };

  // Control-U
  std::vector<Operation *> deadOps;
  getOperation().walk([&](mlir::quantum::CtrlURegion op) {
    // Must be a single-block op
    assert(op.body().getBlocks().size() == 1);
    mlir::OpBuilder rewriter(op);
    mlir::Block &ctrlBlock = op.body().getBlocks().front();
    for (auto &subOp : ctrlBlock.getOperations()) {
      // We're at the end
      if (mlir::dyn_cast_or_null<mlir::quantum::ModifierEndOp>(&subOp)) {
        break;
      }

      // this is not a quantum op:
      if (!is_quantum_op(subOp)) {
        rewriter.insert(subOp.clone());
        continue;
      }
      if (mlir::dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(&subOp)) {
        mlir::quantum::ValueSemanticsInstOp qvsOp =
            mlir::dyn_cast_or_null<mlir::quantum::ValueSemanticsInstOp>(&subOp);
        const auto inst_name = qvsOp.name();
        mlir::Value control_bit = op.ctrl_qubit();
        if (inst_name == "x") {
          std::vector<mlir::Type> ret_types{qvsOp.getOperand(0).getType(),
                                            qvsOp.getOperand(0).getType()};
          std::vector<mlir::Value> qubit_operands{control_bit,
                                                  qvsOp.getOperand(0)};
          auto new_inst = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
              op.getLoc(), llvm::makeArrayRef(ret_types), "cx",
              llvm::makeArrayRef(qubit_operands), llvm::None);
          mlir::Value new_ctrl_qubit_ssa = new_inst.result().front();
          control_bit.replaceAllUsesExcept(
              new_ctrl_qubit_ssa, mlir::SmallPtrSet<Operation *, 1>{new_inst});
        } else if (inst_name == "y") {
          // cy a,b { sdg b; cx a,b; s b; }
          mlir::Type qubit_type = qvsOp.getOperand(0).getType();
          mlir::Value a = op.ctrl_qubit();
          mlir::Value b = qvsOp.getOperand(0);
          auto sdg = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
              op.getLoc(), llvm::makeArrayRef({qubit_type}), "sdg",
              llvm::makeArrayRef({b}), llvm::None);
          // !IMPORTANT! track use-def chain as well
          b = sdg.result().front();
          auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
              op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
              llvm::makeArrayRef({a, b}), llvm::None);
          a = cx.result().front();
          b = cx.result().back();
          auto s = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
              op.getLoc(), llvm::makeArrayRef({qubit_type}), "s",
              llvm::makeArrayRef({b}), llvm::None);
          b = s.result().front();
          op.ctrl_qubit().replaceAllUsesExcept(
              a, mlir::SmallPtrSet<Operation *, 3>{sdg, cx, s});
          qvsOp.getResults().front().replaceAllUsesWith(b);
        } else if (inst_name == "z") {
          // cz a,b { h b; cx a,b; h b; }
          mlir::Type qubit_type = qvsOp.getOperand(0).getType();
          mlir::Value a = op.ctrl_qubit();
          mlir::Value b = qvsOp.getOperand(0);
          std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
          {
            // h b
            auto h = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "h",
                llvm::makeArrayRef({b}), llvm::None);
            b = h.getResults().front();
            newOps.emplace_back(h);
          }
          {
            // cx a,b
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({a, b}), llvm::None);
            a = cx.getResults().front();
            b = cx.getResults().back();
            newOps.emplace_back(cx);
          }
          {
            // h b
            auto h = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "h",
                llvm::makeArrayRef({b}), llvm::None);
            b = h.getResults().front();
            newOps.emplace_back(h);
          }

          mlir::SmallPtrSet<Operation *, 3> newOpPtrs;
          for (auto &x : newOps) {
            newOpPtrs.insert(x);
          }
          op.ctrl_qubit().replaceAllUsesExcept(a, newOpPtrs);
          qvsOp.getResults().front().replaceAllUsesWith(b);
        } else if (inst_name == "h") {
          // gate ch a,b {
          //   h b; sdg b;
          //   cx a,b;
          //   h b; t b;
          //   cx a,b;
          //   t b; h b; s b; x b; s a;
          //   }
          mlir::Type qubit_type = qvsOp.getOperand(0).getType();
          mlir::Value a = op.ctrl_qubit();
          mlir::Value b = qvsOp.getOperand(0);
          std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
          {
            // h b
            auto h = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "h",
                llvm::makeArrayRef({b}), llvm::None);
            b = h.getResults().front();
            newOps.emplace_back(h);
          }
          {
            // sdg b
            auto sdg = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "sdg",
                llvm::makeArrayRef({b}), llvm::None);
            b = sdg.result().front();
            newOps.emplace_back(sdg);
          }
          {
            // cx a,b
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({a, b}), llvm::None);
            a = cx.getResults().front();
            b = cx.getResults().back();
            newOps.emplace_back(cx);
          }
          {
            // h b
            auto h = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "h",
                llvm::makeArrayRef({b}), llvm::None);
            b = h.getResults().front();
            newOps.emplace_back(h);
          }
          {
            // t b
            auto t = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "t",
                llvm::makeArrayRef({b}), llvm::None);
            b = t.getResults().front();
            newOps.emplace_back(t);
          }
          {
            // cx a,b
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({a, b}), llvm::None);
            a = cx.getResults().front();
            b = cx.getResults().back();
            newOps.emplace_back(cx);
          }
          {
            // t b
            auto t = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "t",
                llvm::makeArrayRef({b}), llvm::None);
            b = t.getResults().front();
            newOps.emplace_back(t);
          }
          {
            // h b
            auto h = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "h",
                llvm::makeArrayRef({b}), llvm::None);
            b = h.getResults().front();
            newOps.emplace_back(h);
          }
          {
            // s b
            auto s = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "s",
                llvm::makeArrayRef({b}), llvm::None);
            b = s.getResults().front();
            newOps.emplace_back(s);
          }
          {
            // x b
            auto x = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "x",
                llvm::makeArrayRef({b}), llvm::None);
            b = x.getResults().front();
            newOps.emplace_back(x);
          }
          {
            // s a
            auto s = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "s",
                llvm::makeArrayRef({a}), llvm::None);
            a = s.getResults().front();
            newOps.emplace_back(s);
          }
          assert(newOps.size() == 11);
          mlir::SmallPtrSet<Operation *, 11> newOpPtrs;
          for (auto &x : newOps) {
            newOpPtrs.insert(x);
          }
          op.ctrl_qubit().replaceAllUsesExcept(a, newOpPtrs);
          qvsOp.getResults().front().replaceAllUsesWith(b);
        } else if (inst_name == "t") {
          mlir::Type qubit_type = qvsOp.getOperand(0).getType();
          mlir::Value a = op.ctrl_qubit();
          mlir::Value b = qvsOp.getOperand(0);
          // Ctrl-T = CPhase(pi/4)
          mlir::Value pi_over_4 = rewriter.create<mlir::ConstantOp>(
              op.getLoc(),
              mlir::FloatAttr::get(rewriter.getF64Type(), M_PI / 4));
          auto cp = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
              op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}),
              "cphase", llvm::makeArrayRef({a, b}),
              llvm::makeArrayRef({pi_over_4}));
          a = cp.getResults().front();
          b = cp.getResults().back();
          mlir::SmallPtrSet<Operation *, 1> newOpPtrs{cp};
          op.ctrl_qubit().replaceAllUsesExcept(a, newOpPtrs);
          qvsOp.getResults().front().replaceAllUsesWith(b);
        } else if (inst_name == "tdg") {
          mlir::Type qubit_type = qvsOp.getOperand(0).getType();
          mlir::Value a = op.ctrl_qubit();
          mlir::Value b = qvsOp.getOperand(0);
          // Ctrl-Tdg = CPhase(-pi/4)
          mlir::Value minus_pi_over_4 = rewriter.create<mlir::ConstantOp>(
              op.getLoc(),
              mlir::FloatAttr::get(rewriter.getF64Type(), -M_PI / 4));
          auto cp = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
              op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}),
              "cphase", llvm::makeArrayRef({a, b}),
              llvm::makeArrayRef({minus_pi_over_4}));
          a = cp.getResults().front();
          b = cp.getResults().back();
          mlir::SmallPtrSet<Operation *, 1> newOpPtrs{cp};
          op.ctrl_qubit().replaceAllUsesExcept(a, newOpPtrs);
          qvsOp.getResults().front().replaceAllUsesWith(b);
        } else if (inst_name == "s") {
          mlir::Type qubit_type = qvsOp.getOperand(0).getType();
          mlir::Value a = op.ctrl_qubit();
          mlir::Value b = qvsOp.getOperand(0);
          std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
          // Ctrl-S = CPhase(pi/2)
          mlir::Value pi_over_2 = rewriter.create<mlir::ConstantOp>(
              op.getLoc(),
              mlir::FloatAttr::get(rewriter.getF64Type(), M_PI / 2));
          auto cp = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
              op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}),
              "cphase", llvm::makeArrayRef({a, b}),
              llvm::makeArrayRef({pi_over_2}));
          a = cp.getResults().front();
          b = cp.getResults().back();
          mlir::SmallPtrSet<Operation *, 1> newOpPtrs{cp};
          op.ctrl_qubit().replaceAllUsesExcept(a, newOpPtrs);
          qvsOp.getResults().front().replaceAllUsesWith(b);
        } else if (inst_name == "sdg") {
          mlir::Type qubit_type = qvsOp.getOperand(0).getType();
          mlir::Value a = op.ctrl_qubit();
          mlir::Value b = qvsOp.getOperand(0);
          std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
          // Ctrl-Sdg = CPhase(-pi/2)
          mlir::Value minus_pi_over_2 = rewriter.create<mlir::ConstantOp>(
              op.getLoc(),
              mlir::FloatAttr::get(rewriter.getF64Type(), -M_PI / 2));
          auto cp = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
              op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}),
              "cphase", llvm::makeArrayRef({a, b}),
              llvm::makeArrayRef({minus_pi_over_2}));
          a = cp.getResults().front();
          b = cp.getResults().back();
          mlir::SmallPtrSet<Operation *, 1> newOpPtrs{cp};
          op.ctrl_qubit().replaceAllUsesExcept(a, newOpPtrs);
          qvsOp.getResults().front().replaceAllUsesWith(b);
        } else if (inst_name == "rx") {
          mlir::Type qubit_type = qvsOp.getOperand(0).getType();
          mlir::Value a = op.ctrl_qubit();
          mlir::Value b = qvsOp.getOperand(0);
          std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;

          mlir::Value lambda = qvsOp.getOperand(1);
          mlir::Value float_two = rewriter.create<mlir::ConstantOp>(
              op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), 2.0));
          mlir::Value float_minus_two = rewriter.create<mlir::ConstantOp>(
              op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), -2.0));
          mlir::Value lambda_over_2 =
              rewriter.create<mlir::DivFOp>(op.getLoc(), lambda, float_two);
          mlir::Value minus_lambda_over_2 = rewriter.create<mlir::DivFOp>(
              op.getLoc(), lambda, float_minus_two);

          {
            // rx(lambda/2) b
            auto rx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "rx",
                llvm::makeArrayRef({b}), llvm::makeArrayRef({lambda_over_2}));
            b = rx.getResults().front();
            newOps.emplace_back(rx);
          }
          {
            // cx a,b
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({a, b}), llvm::None);
            a = cx.getResults().front();
            b = cx.getResults().back();
            newOps.emplace_back(cx);
          }
          {
            // rx(-lambda/2) b
            auto rx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "rx",
                llvm::makeArrayRef({b}),
                llvm::makeArrayRef({minus_lambda_over_2}));
            b = rx.getResults().front();
            newOps.emplace_back(rx);
          }
          {
            // cx a,b
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({a, b}), llvm::None);
            a = cx.getResults().front();
            b = cx.getResults().back();
            newOps.emplace_back(cx);
          }

          mlir::SmallPtrSet<Operation *, 4> newOpPtrs;
          for (auto &x : newOps) {
            newOpPtrs.insert(x);
          }
          op.ctrl_qubit().replaceAllUsesExcept(a, newOpPtrs);
          qvsOp.getResults().front().replaceAllUsesWith(b);
        } else if (inst_name == "ry") {
          mlir::Type qubit_type = qvsOp.getOperand(0).getType();
          mlir::Value a = op.ctrl_qubit();
          mlir::Value b = qvsOp.getOperand(0);
          std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
          mlir::Value lambda = qvsOp.getOperand(1);
          mlir::Value float_two = rewriter.create<mlir::ConstantOp>(
              op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), 2.0));
          mlir::Value float_minus_two = rewriter.create<mlir::ConstantOp>(
              op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), -2.0));
          mlir::Value lambda_over_2 =
              rewriter.create<mlir::DivFOp>(op.getLoc(), lambda, float_two);
          mlir::Value minus_lambda_over_2 = rewriter.create<mlir::DivFOp>(
              op.getLoc(), lambda, float_minus_two);
          // gate cry(lambda) a,b
          // {
          //   ry(lambda/2) b;
          //   cx a,b;
          //   ry(-lambda/2) b;
          //   cx a,b;
          // }
          {
            // ry(lambda/2) b;
            auto ry = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "ry",
                llvm::makeArrayRef({b}), llvm::makeArrayRef({lambda_over_2}));
            b = ry.getResults().front();
            newOps.emplace_back(ry);
          }
          {
            // cx a,b;
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({a, b}), llvm::None);
            a = cx.getResults().front();
            b = cx.getResults().back();
            newOps.emplace_back(cx);
          }
          {
            // ry(-lambda/2) b;
            auto ry = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "ry",
                llvm::makeArrayRef({b}),
                llvm::makeArrayRef({minus_lambda_over_2}));
            b = ry.getResults().front();
            newOps.emplace_back(ry);
          }
          {
            // cx a,b;
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({a, b}), llvm::None);
            a = cx.getResults().front();
            b = cx.getResults().back();
            newOps.emplace_back(cx);
          }

          mlir::SmallPtrSet<Operation *, 4> newOpPtrs;
          for (auto &x : newOps) {
            newOpPtrs.insert(x);
          }
          op.ctrl_qubit().replaceAllUsesExcept(a, newOpPtrs);
          qvsOp.getResults().front().replaceAllUsesWith(b);
        } else if (inst_name == "rz") {
          mlir::Type qubit_type = qvsOp.getOperand(0).getType();
          mlir::Value a = op.ctrl_qubit();
          mlir::Value b = qvsOp.getOperand(0);
          mlir::Value lambda = qvsOp.getOperand(1);
          mlir::Value float_two = rewriter.create<mlir::ConstantOp>(
              op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), 2.0));
          mlir::Value float_minus_two = rewriter.create<mlir::ConstantOp>(
              op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), -2.0));
          mlir::Value lambda_over_2 =
              rewriter.create<mlir::DivFOp>(op.getLoc(), lambda, float_two);
          mlir::Value minus_lambda_over_2 = rewriter.create<mlir::DivFOp>(
              op.getLoc(), lambda, float_minus_two);
          std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
          // gate crz(lambda) a,b
          // {
          //   rz(lambda/2) b;
          //   cx a,b;
          //   rz(-lambda/2) b;
          //   cx a,b;
          // }
          {
            // rz(lambda/2) b
            auto rz = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "rz",
                llvm::makeArrayRef({b}), llvm::makeArrayRef({lambda_over_2}));
            b = rz.getResults().front();
            newOps.emplace_back(rz);
          }
          {
            // cx a,b
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({a, b}), llvm::None);
            a = cx.getResults().front();
            b = cx.getResults().back();
            newOps.emplace_back(cx);
          }
          {
            // rz(-lambda/2) b
            auto rz = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "rz",
                llvm::makeArrayRef({b}),
                llvm::makeArrayRef({minus_lambda_over_2}));
            b = rz.getResults().front();
            newOps.emplace_back(rz);
          }
          {
            // cx a,b
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({a, b}), llvm::None);
            a = cx.getResults().front();
            b = cx.getResults().back();
            newOps.emplace_back(cx);
          }

          mlir::SmallPtrSet<Operation *, 4> newOpPtrs;
          for (auto &x : newOps) {
            newOpPtrs.insert(x);
          }
          op.ctrl_qubit().replaceAllUsesExcept(a, newOpPtrs);
          qvsOp.getResults().front().replaceAllUsesWith(b);
        } else if (inst_name == "cx" || inst_name == "cnot") {
          // gate ccx a,b,c
          // {
          //   h c;
          //   cx b,c; tdg c;
          //   cx a,c; t c;
          //   cx b,c; tdg c;
          //   cx a,c; t b; t c; h c;
          //   cx a,b; t a; tdg b;
          //   cx a,b;
          // }
          std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
          mlir::Type qubit_type = qvsOp.getOperand(0).getType();
          mlir::Value a = op.ctrl_qubit();
          mlir::Value b = qvsOp.getOperand(0);
          mlir::Value c = qvsOp.getOperand(1);
          // h c;
          {
            auto h = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "h",
                llvm::makeArrayRef({c}), llvm::None);
            c = h.getResults().front();
            newOps.emplace_back(h);
          }
          // cx b,c
          {
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({b, c}), llvm::None);
            b = cx.getResults().front();
            c = cx.getResults().back();
            newOps.emplace_back(cx);
          }
          // tdg c;
          {
            auto tdg = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "tdg",
                llvm::makeArrayRef({c}), llvm::None);
            c = tdg.getResults().front();
            newOps.emplace_back(tdg);
          }
          // cx a,c
          {
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({a, c}), llvm::None);
            a = cx.getResults().front();
            c = cx.getResults().back();
            newOps.emplace_back(cx);
          }
          // t c
          {
            auto t = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "t",
                llvm::makeArrayRef({c}), llvm::None);
            c = t.getResults().front();
            newOps.emplace_back(t);
          }
          // cx b,c
          {
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({b, c}), llvm::None);
            b = cx.getResults().front();
            c = cx.getResults().back();
            newOps.emplace_back(cx);
          }
          // tdg c;
          {
            auto tdg = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "tdg",
                llvm::makeArrayRef({c}), llvm::None);
            c = tdg.getResults().front();
            newOps.emplace_back(tdg);
          }
          // cx a,c
          {
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({a, c}), llvm::None);
            a = cx.getResults().front();
            c = cx.getResults().back();
            newOps.emplace_back(cx);
          }
          // t b
          {
            auto t = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "t",
                llvm::makeArrayRef({b}), llvm::None);
            b = t.getResults().front();
            newOps.emplace_back(t);
          }
          // t c
          {
            auto t = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "t",
                llvm::makeArrayRef({c}), llvm::None);
            c = t.getResults().front();
            newOps.emplace_back(t);
          }
          // h c;
          {
            auto h = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "h",
                llvm::makeArrayRef({c}), llvm::None);
            c = h.getResults().front();
            newOps.emplace_back(h);
          }
          // cx a,b
          {
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({a, b}), llvm::None);
            a = cx.getResults().front();
            b = cx.getResults().back();
            newOps.emplace_back(cx);
          }
          // t a
          {
            auto t = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "t",
                llvm::makeArrayRef({a}), llvm::None);
            a = t.getResults().front();
            newOps.emplace_back(t);
          }
          // tdg b;
          {
            auto tdg = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type}), "tdg",
                llvm::makeArrayRef({b}), llvm::None);
            b = tdg.getResults().front();
            newOps.emplace_back(tdg);
          }
          // cx a,b;
          {
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({a, b}), llvm::None);
            a = cx.getResults().front();
            b = cx.getResults().back();
            newOps.emplace_back(cx);
          }
          assert(newOps.size() == 15);
          mlir::SmallPtrSet<Operation *, 15> newOpPtrs;
          for (auto &x : newOps) {
            newOpPtrs.insert(x);
          }
          op.ctrl_qubit().replaceAllUsesExcept(a, newOpPtrs);
          qvsOp.getResults().front().replaceAllUsesWith(b);
          qvsOp.getResults().back().replaceAllUsesWith(c);
        } else if (inst_name == "cphase") {
          // Ref:
          // ccu1(lambda, a, b, c) =
          // cu1(lambda/2, a, b)
          // cx(b, c)
          // cu1(-lambda/2, a, c)
          // cx(b, c)
          // cu1(lambda/2, a, c)
          mlir::Type qubit_type = qvsOp.getOperand(0).getType();
          mlir::Value a = op.ctrl_qubit();
          mlir::Value b = qvsOp.getOperand(0);
          mlir::Value c = qvsOp.getOperand(1);
          mlir::Value lambda = qvsOp.getOperand(2);
          mlir::Value float_two = rewriter.create<mlir::ConstantOp>(
              op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), 2.0));
          mlir::Value float_minus_two = rewriter.create<mlir::ConstantOp>(
              op.getLoc(), mlir::FloatAttr::get(rewriter.getF64Type(), -2.0));
          mlir::Value lambda_over_2 =
              rewriter.create<mlir::DivFOp>(op.getLoc(), lambda, float_two);
          mlir::Value minus_lambda_over_2 = rewriter.create<mlir::DivFOp>(
              op.getLoc(), lambda, float_minus_two);
          std::vector<mlir::quantum::ValueSemanticsInstOp> newOps;
          {
            // cu1(lambda/2, a, b)
            auto cp = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}),
                "cphase", llvm::makeArrayRef({a, b}),
                llvm::makeArrayRef({lambda_over_2}));
            a = cp.getResults().front();
            b = cp.getResults().back();
            newOps.emplace_back(cp);
          }
          {
            // cx(b, c)
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({b, c}), llvm::None);
            b = cx.getResults().front();
            c = cx.getResults().back();
            newOps.emplace_back(cx);
          }
          {
            // cu1(-lambda/2, a, c)
            auto cp = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}),
                "cphase", llvm::makeArrayRef({a, c}),
                llvm::makeArrayRef({minus_lambda_over_2}));
            a = cp.getResults().front();
            c = cp.getResults().back();
            newOps.emplace_back(cp);
          }
          {
            // cx(b, c)
            auto cx = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}), "cx",
                llvm::makeArrayRef({b, c}), llvm::None);
            b = cx.getResults().front();
            c = cx.getResults().back();
            newOps.emplace_back(cx);
          }
          {
            // cu1(lambda/2, a, c)
            auto cp = rewriter.create<mlir::quantum::ValueSemanticsInstOp>(
                op.getLoc(), llvm::makeArrayRef({qubit_type, qubit_type}),
                "cphase", llvm::makeArrayRef({a, c}),
                llvm::makeArrayRef({lambda_over_2}));
            a = cp.getResults().front();
            c = cp.getResults().back();
            newOps.emplace_back(cp);
          }
          assert(newOps.size() == 5);
          mlir::SmallPtrSet<Operation *, 5> newOpPtrs;
          for (auto &x : newOps) {
            newOpPtrs.insert(x);
          }
          op.ctrl_qubit().replaceAllUsesExcept(a, newOpPtrs);
          qvsOp.getResults().front().replaceAllUsesWith(b);
          qvsOp.getResults().back().replaceAllUsesWith(c);
        } else {
          // We don't expect this gate just yet, need to add.
          std::cout << "Unknown quantum gate: " << inst_name.str() << "\n";
          assert(false);
        }
      }
    }
    op.body().getBlocks().clear();
    deadOps.emplace_back(op.getOperation());
  });
  for (auto &op : deadOps) {
    op->dropAllUses();
    op->erase();
  }
  deadOps.clear();
}
void ModifierBlockInlinerPass::handleAdjU() {}

void ModifierBlockInlinerPass::runOnOperation() {
  handlePowU();
  handleCtrlU();
  handleAdjU();
}
} // namespace qcor
