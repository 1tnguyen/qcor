#pragma once

#include "ast/ast.hpp"
#include "ast/traversal.hpp"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "parser/parser.hpp"

using namespace staq::ast;

namespace qasm_parser {

class StaqToMLIR : public staq::ast::Visitor {
 protected:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;

 public:
  StaqToMLIR(mlir::MLIRContext &context);
  mlir::ModuleOp module() {return theModule;}
  void visit(VarAccess &) override;
  // Expressions
  void visit(BExpr &) override;
  void visit(UExpr &) override;
  void visit(PiExpr &) override;
  void visit(IntExpr &) override;
  void visit(RealExpr &r) override;
  void visit(VarExpr &v) override;
  void visit(ResetStmt &) override;
  void visit(IfStmt &) override;
  void visit(BarrierGate &) override;
  void visit(GateDecl &) override;
  void visit(OracleDecl &) override;
  void visit(RegisterDecl &) override;
  void visit(AncillaDecl &) override;
  void visit(Program &prog) override;
  void visit(MeasureStmt &m) override;
  void visit(UGate &u) override;
  void visit(CNOTGate &cx) override;
  void visit(DeclaredGate &g) override;

  void addReturn();
};
}  // namespace qasm_parser