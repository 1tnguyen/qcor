
#include "expression_handler.hpp"
#include "qasm3_visitor.hpp"

namespace qcor {
antlrcpp::Any qasm3_visitor::visitCompute_action_stmt(
    qasm3Parser::Compute_action_stmtContext *context) {
  auto location = get_location(builder, file_name, context);

  builder.create<mlir::quantum::ComputeMarkerOp>(location);
  visit(context->compute_block);
  builder.create<mlir::quantum::ComputeUnMarkerOp>(location);
  visit(context->action_block);

  builder.create<mlir::quantum::ComputeMarkerOp>(location);

  auto adjUOp = builder.create<mlir::quantum::AdjURegion>(location);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&adjUOp.body().front());
    visit(context->compute_block);
    builder.create<mlir::quantum::ModifierEndOp>(location);
  }

  builder.create<mlir::quantum::ComputeUnMarkerOp>(location);

  return 0;
}

}  // namespace qcor