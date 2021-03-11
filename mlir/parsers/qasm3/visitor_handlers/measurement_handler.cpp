
#include <numeric>

#include "qasm3_visitor.hpp"

namespace qcor {

//     quantumMeasurement
//     : 'measure' indexIdentifierList
//     ;

// quantumMeasurementAssignment
//     : quantumMeasurement ( ARROW indexIdentifierList)?
//     | indexIdentifierList EQUALS quantumMeasurement
//     ;

antlrcpp::Any qasm3_visitor::visitQuantumMeasurement(
    qasm3Parser::QuantumMeasurementContext* context) {
  printErrorMessage("visiting this node is not implemented, havent seen it yet. if you hit this, let alex know", context);
  return 0;
}

antlrcpp::Any qasm3_visitor::visitQuantumMeasurementAssignment(
    qasm3Parser::QuantumMeasurementAssignmentContext* context) {
  auto location = get_location(builder, file_name, context);
  auto str_attr = builder.getStringAttr("mz");

  if (is_return_stmt) {
    // should be something like measure q;
    auto qubit_or_qreg = context->quantumMeasurement()
                             ->indexIdentifierList()
                             ->indexIdentifier(0)
                             ->Identifier();
    auto measured_object = symbol_table.get_symbol(qubit_or_qreg->getText());
    if (measured_object.getType() == qubit_type) {
      mlir::Value instop =
          builder
              .create<mlir::quantum::InstOp>(
                  location, result_type, str_attr,
                  llvm::makeArrayRef(std::vector<mlir::Value>{measured_object}),
                  llvm::makeArrayRef(std::vector<mlir::Value>{}))
              .bit();
      symbol_table.add_symbol("return_value_" + qubit_or_qreg->getText(),
                              instop);
      return 0;
    } else {
      // array type
      // create memref of size qreg_size, measure all qubits,
      // and store the bit results to each element of the memref,
      // then return that.
      return 0;
    }
  }

  // Handle all other measure stmts
  if (context->EQUALS() || context->ARROW()) {
    // we have EXPR = measure EXPR
    auto indexIdentifierList = context->indexIdentifierList();
    auto measured_list = context->quantumMeasurement()->indexIdentifierList();

    // Get the Qubits
    std::string measured_qreg =
        measured_list->indexIdentifier(0)->Identifier()->getText();
    auto value = symbol_table.get_symbol(measured_qreg);

    // Get the Bits
    auto bit_variable_name =
        indexIdentifierList->indexIdentifier(0)->Identifier()->getText();
    auto bit_value = symbol_table.get_symbol(bit_variable_name);

    // First handle the case measure qubit, or measure qubit[i];
    if (value.getType() == array_type &&
        measured_list->indexIdentifier(0)->expressionList()) {
      // Here we are measuring a qubit from an array

      // let's get that single qubit value
      auto idx_str = measured_list->indexIdentifier(0)
                         ->expressionList()
                         ->expression(0)
                         ->getText();

      try {
        value = get_or_extract_qubit(measured_qreg, std::stoi(idx_str),
                                     location, symbol_table, builder);
      } catch (...) {
        if (symbol_table.has_symbol(idx_str)) {
          auto qubits = symbol_table.get_symbol(measured_qreg);
          auto qbit = symbol_table.get_symbol(idx_str);
          llvm::StringRef qubit_type_name("Qubit");
          mlir::Identifier dialect =
              mlir::Identifier::get("quantum", builder.getContext());
          auto qubit_type = mlir::OpaqueType::get(builder.getContext(), dialect,
                                                  qubit_type_name);

          value = builder.create<mlir::quantum::ExtractQubitOp>(
              location, qubit_type, qubits, qbit);
        } else {
          printErrorMessage(
              "Invalid measurement index on the given qubit register: " +
              measured_qreg + ", " + idx_str, context);
        }
      }
    } else if (value.getType() == array_type &&
               measured_list->indexIdentifier(0)->rangeDefinition()) {
      // This is measuring a range...

      if (auto bit_range =
              indexIdentifierList->indexIdentifier(0)->rangeDefinition()) {
        if (bit_range->expression().size() > 2) {
          printErrorMessage(
              "we only support measuring ranges with [start:stop]");
        }

        // Get bit and qubit ranges integers...
        auto ba = symbol_table.evaluate_constant_integer_expression(
            bit_range->expression(0)->getText());
        auto bb = symbol_table.evaluate_constant_integer_expression(
            bit_range->expression(1)->getText());
        auto qa = symbol_table.evaluate_constant_integer_expression(
            measured_list->indexIdentifier(0)
                ->rangeDefinition()
                ->expression(0)
                ->getText());
        auto qb = symbol_table.evaluate_constant_integer_expression(
            measured_list->indexIdentifier(0)
                ->rangeDefinition()
                ->expression(1)
                ->getText());

        auto nqubits_in_register =
            value.getDefiningOp<mlir::quantum::QallocOp>()
                .size()
                .getLimitedValue();
        if (qb >= nqubits_in_register) {
          printErrorMessage("End value for qubit range is >= n_qubits ( = " +
                                std::to_string(nqubits_in_register) +
                                ") in register.\n",
                            context);
        }

        auto n_bits_in_register =
            bit_value.getType().cast<mlir::MemRefType>().getShape()[0];
        if (bb >= n_bits_in_register) {
          printErrorMessage("End value for bit range is >= n_bits ( = " +
                                std::to_string(n_bits_in_register) +
                                ") in register.\n",
                            context);
        }

        if ((bb - ba) != (qb - qa)) {
          printErrorMessage("Range sizes must be equal.", context);
        }

        auto size = bb - ba;
        std::vector<int> qubit_indices(size), bit_indices(size);
        std::iota(std::begin(qubit_indices), std::end(qubit_indices), qa);
        std::iota(std::begin(bit_indices), std::end(bit_indices), ba);
        qubit_indices.push_back(qb);
        bit_indices.push_back(bb);

        for (int i = 0; i <= size; i++) {
          auto qbit_idx = qubit_indices[i];
          auto bit_idx = bit_indices[i];

          mlir::Value idx_val = get_or_create_constant_index_value(
              qbit_idx, location, 64, symbol_table, builder);
          mlir::Value bit_idx_val = get_or_create_constant_index_value(
              bit_idx, location, 64, symbol_table, builder);

          auto extract = builder.create<mlir::quantum::ExtractQubitOp>(
              location, qubit_type, value, idx_val);

          auto instop = builder.create<mlir::quantum::InstOp>(
              location, result_type, str_attr,
              llvm::makeArrayRef(extract.qbit()),
              llvm::makeArrayRef(std::vector<mlir::Value>{}));

          builder.create<mlir::StoreOp>(
              location, instop.bit(), bit_value,
              llvm::makeArrayRef(std::vector<mlir::Value>{bit_idx_val}));
        }

        return 0;

      } else {
        printErrorMessage(
            "If you are measuring a qubit range you must store to a bit "
            "range.", context);
      }
    }

    // Ok, now if this is a qubit type, and not an array
    if (value.getType() == qubit_type) {
      auto instop = builder.create<mlir::quantum::InstOp>(
          location, result_type, str_attr, llvm::makeArrayRef(value),
          llvm::makeArrayRef(std::vector<mlir::Value>{}));

      // Get the bit or bit[]
      int bit_idx = 0;
      if (auto index_list =
              indexIdentifierList->indexIdentifier(0)->expressionList()) {
        // Need to extract element from bit array to set it
        auto idx_str = index_list->expression(0)->getText();
        bit_idx = std::stoi(idx_str);
      }


      // Store the mz result into the bit_value
      mlir::Value pos = get_or_create_constant_index_value(
          bit_idx, location, 64, symbol_table, builder);

      builder.create<mlir::StoreOp>(
          location, instop.bit(), bit_value,
          llvm::makeArrayRef(std::vector<mlir::Value>{pos}));
    } else {
      // This is the case where we are measuring an entire qubit array
      // to a bit array
      // First check that the sizes match up
      std::uint64_t nqubits;
      if (auto qalloc_op = value.getDefiningOp<mlir::quantum::QallocOp>()) {
        nqubits = qalloc_op.size().getLimitedValue();
      } else {
        // then this is a block arg, we should have a size attribute
        auto attributes = symbol_table.get_variable_attributes(measured_qreg);
        if (!attributes.empty()) {
          try {
            nqubits = std::stoi(attributes[0]);
          } catch (...) {
            printErrorMessage(
                "Could not infer qubit[] size from block argument.", context);
          }
        } else {
          printErrorMessage(
              "Could not infer qubit[] size from block argument. No size "
              "attribute for variable in symbol table.",
              context);
        }
      }
      auto nbits = bit_value.getType().cast<mlir::MemRefType>().getShape()[0];
      if (nbits != nqubits) {
        printErrorMessage(
            "cannot measure the qubit array to the specified bit array, sizes "
            "do not match: n_qubits = " +
                std::to_string(nqubits) + ", n_bits = " + std::to_string(nbits),
            context);
      }

      for (int i = 0; i < nqubits; i++) {
        mlir::Value idx_val = get_or_create_constant_index_value(
            i, location, 64, symbol_table, builder);

        auto extract = builder.create<mlir::quantum::ExtractQubitOp>(
            location, qubit_type, value, idx_val);

        auto instop = builder.create<mlir::quantum::InstOp>(
            location, result_type, str_attr, llvm::makeArrayRef(extract.qbit()),
            llvm::makeArrayRef(std::vector<mlir::Value>{}));

        builder.create<mlir::StoreOp>(
            location, instop.bit(), bit_value,
            llvm::makeArrayRef(std::vector<mlir::Value>{idx_val}));
      }
    }

  } else {
    printErrorMessage("invalid measure syntax.",
                      context);  
  }
  return 0;
}
}  // namespace qcor