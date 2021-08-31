#include "expression_handler.hpp"
#include "exprtk.hpp"
#include "qasm3_visitor.hpp"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"

using symbol_table_t = exprtk::symbol_table<double>;
using expression_t = exprtk::expression<double>;
using parser_t = exprtk::parser<double>;

namespace {
/// Creates a single affine "for" loop, iterating from lbs to ubs with
/// the given step.
/// to construct the body of the loop and is passed the induction variable.
mlir::AffineForOp affineLoopBuilder(mlir::Value lbs_val, mlir::Value ubs_val, int64_t step,
                       std::function<void(mlir::Value)> bodyBuilderFn,
                       mlir::OpBuilder &builder, mlir::Location &loc) {
  if (!ubs_val.getType().isa<mlir::IndexType>()) {
    ubs_val =
        builder.create<mlir::IndexCastOp>(loc, builder.getIndexType(), ubs_val);
  }
  if (!lbs_val.getType().isa<mlir::IndexType>()) {
    lbs_val =
        builder.create<mlir::IndexCastOp>(loc, builder.getIndexType(), lbs_val);
  }
  // Note: Affine for loop only accepts **positive** step:
  // The stride, represented by step, is a positive constant integer which
  // defaults to “1” if not present.
  assert(step != 0);
  if (step > 0) {
    mlir::ValueRange lbs(lbs_val);
    mlir::ValueRange ubs(ubs_val);
    // Create the actual loop
    return builder.create<mlir::AffineForOp>(
        loc, lbs, builder.getMultiDimIdentityMap(lbs.size()), ubs,
        builder.getMultiDimIdentityMap(ubs.size()), step, llvm::None,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
            mlir::Value iv, mlir::ValueRange itrArgs) {
          mlir::OpBuilder::InsertionGuard guard(nestedBuilder);
          bodyBuilderFn(iv);
          nestedBuilder.create<mlir::AffineYieldOp>(nestedLoc);
        });
  } else {
    // Negative step:
    // a -> b step c (minus)
    // -a -> -b step c (plus) and minus the loop var
    mlir::Value minus_one = builder.create<mlir::ConstantOp>(
        loc, mlir::IntegerAttr::get(lbs_val.getType(), -1));
    lbs_val = builder.create<mlir::MulIOp>(loc, lbs_val, minus_one).result();
    ubs_val = builder.create<mlir::MulIOp>(loc, ubs_val, minus_one).result();
    mlir::ValueRange lbs(lbs_val);
    mlir::ValueRange ubs(ubs_val);
    return builder.create<mlir::AffineForOp>(
        loc, lbs, builder.getMultiDimIdentityMap(lbs.size()), ubs,
        builder.getMultiDimIdentityMap(ubs.size()), -step, llvm::None,
        [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc,
            mlir::Value iv, mlir::ValueRange itrArgs) {
          mlir::OpBuilder::InsertionGuard guard(nestedBuilder);
          mlir::Value minus_one_idx = nestedBuilder.create<mlir::ConstantOp>(
              nestedLoc, mlir::IntegerAttr::get(iv.getType(), -1));
          bodyBuilderFn(
              nestedBuilder.create<mlir::MulIOp>(nestedLoc, iv, minus_one_idx)
                  .result());
          nestedBuilder.create<mlir::AffineYieldOp>(nestedLoc);
        });
  }
}
} // namespace
namespace qcor {
antlrcpp::Any qasm3_visitor::visitLoopStatement(
    qasm3Parser::LoopStatementContext* context) {
  auto location = get_location(builder, file_name, context);

  auto loop_signature = context->loopSignature();
  auto program_block = context->programBlock();

  if (auto membership_test = loop_signature->membershipTest()) {
    // this is a for loop
    auto idx_var_name = membership_test->Identifier()->getText();

    auto set_declaration = membership_test->setDeclaration();
    if (set_declaration->LBRACE()) {
      auto exp_list = set_declaration->expressionList();
      auto n_expr = exp_list->expression().size();

      auto allocation =
          allocate_1d_memory(location, n_expr, builder.getI64Type());

      // FIXME: using Affine For loop:
      // the body builder needs to load the element at the loop index.
      // allocate i64 memref of size n_expr
      // for i in {1,2,3} -> affine.for i : 0 to 3 { element = load(memref, i) }
      int counter = 0;
      for (auto exp : exp_list->expression()) {
        qasm3_expression_generator exp_generator(builder, symbol_table,
                                                 file_name);
        exp_generator.visit(exp);
        auto value = exp_generator.current_value;

        mlir::Value pos = get_or_create_constant_index_value(
            counter, location, 64, symbol_table, builder);

        builder.create<mlir::StoreOp>(
            location, value, allocation,
            llvm::makeArrayRef(std::vector<mlir::Value>{pos}));

        counter++;
      }

      symbol_table.enter_new_scope();

      auto tmp = get_or_create_constant_index_value(0, location, 64,
                                                    symbol_table, builder);
      auto tmp2 = get_or_create_constant_index_value(0, location, 64,
                                                     symbol_table, builder);
      llvm::ArrayRef<mlir::Value> zero_index(tmp2);
      // Loop var must also be an Index type
      // since we'll store the loop index values to this variable.
      auto loop_var_memref = allocate_1d_memory_and_initialize(
          location, 1, builder.getIndexType(), std::vector<mlir::Value>{tmp},
          llvm::makeArrayRef(std::vector<mlir::Value>{tmp}));

      auto b_val = get_or_create_constant_index_value(n_expr, location, 64,
                                                      symbol_table, builder);
      auto c_val = get_or_create_constant_index_value(1, location, 64,
                                                      symbol_table, builder);

      // Strategy...

      // We need to create a header block to check that loop var is still valid
      // it will branch at the end to the body or the exit

      // Then we create the body block, it should branch to the incrementor
      // block

      // Then we create the incrementor block, it should branch back to header

      // Any downstream children that will create blocks will need to know what
      // the fallback block for them is, and it should be the incrementor block
      auto savept = builder.saveInsertionPoint();
      auto currRegion = builder.getBlock()->getParent();
      auto headerBlock = builder.createBlock(currRegion, currRegion->end());
      auto bodyBlock = builder.createBlock(currRegion, currRegion->end());
      auto incBlock = builder.createBlock(currRegion, currRegion->end());
      mlir::Block* exitBlock =
          builder.createBlock(currRegion, currRegion->end());
      builder.restoreInsertionPoint(savept);

      builder.create<mlir::BranchOp>(location, headerBlock);
      builder.setInsertionPointToStart(headerBlock);

      auto load =
          builder.create<mlir::LoadOp>(location, loop_var_memref, zero_index);
      auto cmp = builder.create<mlir::CmpIOp>(
          location, mlir::CmpIPredicate::slt, load, b_val);
      builder.create<mlir::CondBranchOp>(location, cmp, bodyBlock, exitBlock);

      builder.setInsertionPointToStart(bodyBlock);
      // Load the loop variable from the memref allocation
      auto load2 =
          builder.create<mlir::LoadOp>(location, allocation, load.result());

      // Save the loaded value as the loop variable name
      symbol_table.add_symbol(idx_var_name, load2.result(), {}, true);

      current_loop_exit_block = exitBlock;

      current_loop_incrementor_block = incBlock;

      visitChildren(program_block);

      current_loop_header_block = nullptr;
      current_loop_exit_block = nullptr;

      builder.create<mlir::BranchOp>(location, incBlock);

      builder.setInsertionPointToStart(incBlock);
      auto load_inc =
          builder.create<mlir::LoadOp>(location, loop_var_memref, zero_index);
      auto add = builder.create<mlir::AddIOp>(location, load_inc, c_val);
      
      assert(tmp2.getType().isa<mlir::IndexType>());
      builder.create<mlir::StoreOp>(
          location, add, loop_var_memref,
          llvm::makeArrayRef(std::vector<mlir::Value>{tmp2}));

      builder.create<mlir::BranchOp>(location, headerBlock);

      builder.setInsertionPointToStart(exitBlock);

      symbol_table.set_last_created_block(exitBlock);

      // Exit scope and restore insertion
      symbol_table.exit_scope();

    } else if (auto range = set_declaration->rangeDefinition()) {
      // this is a range definition
      //     rangeDefinition
      // : LBRACKET expression? COLON expression? ( COLON expression )? RBRACKET
      // ;

      auto range_str =
          range->getText().substr(1, range->getText().length() - 2);
      auto range_elements = split(range_str, ':');
      auto n_expr = range->expression().size();
      int a, b, c;

      // First question what type should we use?
      mlir::Type int_type = builder.getI64Type();
      if (symbol_table.has_symbol(range->expression(0)->getText())) {
        int_type =
            symbol_table.get_symbol(range->expression(0)->getText()).getType();
      }
      if (n_expr == 3) {
        if (symbol_table.has_symbol(range->expression(1)->getText())) {
          int_type = symbol_table.get_symbol(range->expression(1)->getText())
                         .getType();
        } else if (symbol_table.has_symbol(range->expression(2)->getText())) {
          int_type = symbol_table.get_symbol(range->expression(2)->getText())
                         .getType();
        }
      } else {
        if (symbol_table.has_symbol(range->expression(1)->getText())) {
          int_type = symbol_table.get_symbol(range->expression(1)->getText())
                         .getType();
        }
      }

      if (int_type.isa<mlir::MemRefType>()) {
        int_type = int_type.cast<mlir::MemRefType>().getElementType();
      }

      c = 1;
      mlir::Value a_value, b_value,
          c_value = get_or_create_constant_integer_value(c, location, int_type,
                                                         symbol_table, builder);

      const auto const_eval_a_val =
          symbol_table.try_evaluate_constant_integer_expression(
              range->expression(0)->getText());
      if (const_eval_a_val.has_value()) {
        // std::cout << "A val = " << const_eval_a_val.value() << "\n";
        a_value = get_or_create_constant_integer_value(const_eval_a_val.value(),
                                                       location, int_type,
                                                       symbol_table, builder);
      } else {
        qasm3_expression_generator exp_generator(builder, symbol_table,
                                                 file_name);
        exp_generator.visit(range->expression(0));
        a_value = exp_generator.current_value;
        if (a_value.getType().isa<mlir::MemRefType>()) {
          a_value = builder.create<mlir::LoadOp>(location, a_value);
        }
      }

      if (n_expr == 3) {
        const auto const_eval_b_val =
          symbol_table.try_evaluate_constant_integer_expression(
              range->expression(2)->getText());
        if (const_eval_b_val.has_value()) {
          // std::cout << "B val = " << const_eval_b_val.value() << "\n";
          b_value = get_or_create_constant_integer_value(
              const_eval_b_val.value(), location, int_type, symbol_table,
              builder);
        } else {
          qasm3_expression_generator exp_generator(builder, symbol_table,
                                                   file_name);
          exp_generator.visit(range->expression(2));
          b_value = exp_generator.current_value;
          if (b_value.getType().isa<mlir::MemRefType>()) {
            b_value = builder.create<mlir::LoadOp>(location, b_value);
          }
        }
        if (symbol_table.has_symbol(range->expression(1)->getText())) {
          printErrorMessage("You must provide loop step as a constant value.",
                            context);
          // c_value = symbol_table.get_symbol(range->expression(1)->getText());
          // c_value = builder.create<mlir::LoadOp>(location, c_value);
          // if (c_value.getType() != int_type) {
          //   printErrorMessage("For loop a, b, and c types are not equal.",
          //                     context, {a_value, c_value});
          // }
        } else {
          c = symbol_table.evaluate_constant_integer_expression(
              range->expression(1)->getText());
          c_value = get_or_create_constant_integer_value(
              c, location, a_value.getType(), symbol_table, builder);
        }

      } else {
        const auto const_eval_b_val =
            symbol_table.try_evaluate_constant_integer_expression(
                range->expression(1)->getText());
        if (const_eval_b_val.has_value()) {
          // std::cout << "B val = " << const_eval_b_val.value() << "\n";
          b_value = get_or_create_constant_integer_value(
              const_eval_b_val.value(), location, int_type, symbol_table,
              builder);
        } else {
          qasm3_expression_generator exp_generator(builder, symbol_table,
                                                   file_name);
          exp_generator.visit(range->expression(1));
          b_value = exp_generator.current_value;
          if (b_value.getType().isa<mlir::MemRefType>()) {
            b_value = builder.create<mlir::LoadOp>(location, b_value);
          }
        }
      }

      const std::string program_block_str = program_block->getText();
      // std::cout << "HOWDY:\n" << program_block_str << "\n";

      // HACK: Currently, we don't handle 'break', 'continue' or nested loop
      // in the Affine for loop yet.
      if (program_block_str.find("QCOR_EXPECT_TRUE") == std::string::npos) {
        // Can use Affine for loop....
        auto forLoop = affineLoopBuilder(
            a_value, b_value, c,
            [&](mlir::Value loop_var) {
              // Create a new scope for the for loop
              symbol_table.enter_new_scope();
              auto loop_var_cast = builder.create<mlir::IndexCastOp>(
                  location, builder.getI64Type(), loop_var);
              symbol_table.add_symbol(idx_var_name, loop_var_cast, {}, true);
              visitChildren(program_block);
              symbol_table.exit_scope();
            },
            builder, location);
        if (!loop_break_vars.empty()) {
          mlir::OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToStart(
              &(forLoop.getLoopBody().getBlocks().front()));
          mlir::Value mustBreak =
              builder.create<mlir::LoadOp>(location, loop_break_vars.top());
          assert(mustBreak.getType().isa<mlir::IntegerType>() &&
                 mustBreak.getType().getIntOrFloatBitWidth() == 1);
          loop_break_vars.pop();
          // Wrap/Outline the loop body in an IfOp:
          auto scfIfOp = builder.create<mlir::scf::IfOp>(
              location, mlir::TypeRange(), mustBreak, false);
          size_t count = 0;
          std::vector<mlir::Operation *> ops_to_clone;
          for (auto op_iter = forLoop.getLoopBody().op_begin();
               op_iter != forLoop.getLoopBody().op_end(); ++op_iter) {
            mlir::Operation &op = *op_iter;
            count++;
            // The first 2 are the Load and If that we just inserted
            if (count <= 2) {
              continue;
            }
            ops_to_clone.emplace_back(&op);
          }
          // Last one is affine yield
          // should be left outside:
          assert(
              mlir::dyn_cast_or_null<mlir::AffineYieldOp>(ops_to_clone.back()));
          ops_to_clone.pop_back();
          for (auto &op : ops_to_clone) {
            scfIfOp.getThenBodyBuilder().clone(*op);
          }

          for (auto &op : ops_to_clone) {
            op->remove();
          }
        }
      } else {
        // TODO: Remove this code path once we convert control flow to Affine/SCF
        // Need to use the legacy for loop construction for now...
        // Create a new scope for the for loop
        symbol_table.enter_new_scope();

        llvm::ArrayRef<int64_t> shaperef{};
        auto mem_type = mlir::MemRefType::get(shaperef, a_value.getType());
        mlir::Value loop_var_memref =
            builder.create<mlir::AllocaOp>(location, mem_type);
        builder.create<mlir::StoreOp>(location, a_value, loop_var_memref);

        // Save the current builder point
        // auto savept = builder.saveInsertionPoint();
        auto loaded_var =
            builder.create<mlir::LoadOp>(location, loop_var_memref);

        symbol_table.add_symbol(idx_var_name, loaded_var, {}, true);

        // Strategy...

        // We need to create a header block to check that loop var is still
        // valid it will branch at the end to the body or the exit

        // Then we create the body block, it should branch to the incrementor
        // block

        // Then we create the incrementor block, it should branch back to header

        // Any downstream children that will create blocks will need to know
        // what the fallback block for them is, and it should be the incrementor
        // block
        auto savept = builder.saveInsertionPoint();
        auto currRegion = builder.getBlock()->getParent();
        auto headerBlock = builder.createBlock(currRegion, currRegion->end());
        auto bodyBlock = builder.createBlock(currRegion, currRegion->end());
        auto incBlock = builder.createBlock(currRegion, currRegion->end());
        mlir::Block *exitBlock =
            builder.createBlock(currRegion, currRegion->end());
        builder.restoreInsertionPoint(savept);

        builder.create<mlir::BranchOp>(location, headerBlock);
        builder.setInsertionPointToStart(headerBlock);

        mlir::Value load = builder.create<mlir::LoadOp>(location, loop_var_memref);

        if (load.getType().getIntOrFloatBitWidth() <
            b_value.getType().getIntOrFloatBitWidth()) {
          load =
              builder.create<mlir::ZeroExtendIOp>(location, load, b_value.getType());
        } else if (b_value.getType().getIntOrFloatBitWidth() <
                   load.getType().getIntOrFloatBitWidth()) {
          b_value =
              builder.create<mlir::ZeroExtendIOp>(location, b_value, load.getType());
        }

        auto cmp = builder.create<mlir::CmpIOp>(
            location,
            c > 0 ? mlir::CmpIPredicate::slt : mlir::CmpIPredicate::sge, load,
            b_value);
        builder.create<mlir::CondBranchOp>(location, cmp, bodyBlock, exitBlock);

        builder.setInsertionPointToStart(bodyBlock);
        // body needs to load the loop variable
        auto x = builder.create<mlir::LoadOp>(location, loop_var_memref);
        symbol_table.add_symbol(idx_var_name, x, {}, true);

        current_loop_exit_block = exitBlock;

        current_loop_incrementor_block = incBlock;

        visitChildren(program_block);

        current_loop_incrementor_block = nullptr;
        current_loop_exit_block = nullptr;

        builder.create<mlir::BranchOp>(location, incBlock);

        builder.setInsertionPointToStart(incBlock);
        auto load_inc = builder.create<mlir::LoadOp>(location, loop_var_memref);

        auto add = builder.create<mlir::AddIOp>(location, load_inc, c_value);

        builder.create<mlir::StoreOp>(location, add, loop_var_memref);

        builder.create<mlir::BranchOp>(location, headerBlock);

        builder.setInsertionPointToStart(exitBlock);

        symbol_table.set_last_created_block(exitBlock);

        symbol_table.exit_scope();
      }
    } else {
      printErrorMessage(
          "For loops must be of form 'for i in {SET}' or 'for i in [RANGE]'.");
    }

  } else {
    // FIXME: convert to mlir::scf::WhileOp (should be easy since it's
    // conditioned on an i1 value) 
    // this is a while loop
    auto while_expr = loop_signature->booleanExpression();

    // Create a new scope for the for loop
    symbol_table.enter_new_scope();

    auto currRegion = builder.getBlock()->getParent();

    auto savept = builder.saveInsertionPoint();
    auto headerBlock = builder.createBlock(currRegion, currRegion->end());
    auto bodyBlock = builder.createBlock(currRegion, currRegion->end());
    auto exitBlock = builder.createBlock(currRegion, currRegion->end());

    builder.restoreInsertionPoint(savept);
    builder.create<mlir::BranchOp>(location, headerBlock);

    builder.setInsertionPointToEnd(headerBlock);
    qasm3_expression_generator exp_generator(builder, symbol_table, file_name);
    exp_generator.visit(while_expr);
    auto expr_value = exp_generator.current_value;
    builder.create<mlir::CondBranchOp>(location, expr_value, bodyBlock,
                                       exitBlock);

    builder.setInsertionPointToStart(bodyBlock);
    current_loop_exit_block = exitBlock;
    current_loop_header_block = headerBlock;

    visitChildren(program_block);

    current_loop_header_block = nullptr;
    current_loop_exit_block = nullptr;

    builder.create<mlir::BranchOp>(location, headerBlock);
    builder.setInsertionPointToStart(exitBlock);

    symbol_table.exit_scope();

    // This is where we do some manipulation of
    // the basic blocks, lets store the current last block
    // so that finalize_mlirgen() can add return and deallocs
    // correctly

    symbol_table.set_last_created_block(exitBlock);
  }

  return 0;
}
}  // namespace qcor