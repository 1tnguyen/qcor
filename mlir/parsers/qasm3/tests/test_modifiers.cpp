/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the BSD 3-Clause License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"
namespace {
// returns count of non-overlapping occurrences of 'sub' in 'str'
int countSubstring(const std::string &str, const std::string &sub) {
  if (sub.length() == 0)
    return 0;
  int count = 0;
  for (size_t offset = str.find(sub); offset != std::string::npos;
       offset = str.find(sub, offset + sub.length())) {
    ++count;
  }
  return count;
}
} // namespace
// Kitchen-sink *functional* testing of modifiers (pow/inv/ctrl)
TEST(qasm3VisitorTester, checkPow) {
  const std::string check_pow = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
bit c;

pow(2) @ x q;
measure q -> c;

QCOR_EXPECT_TRUE(c == 0);

reset q;

pow(5) @ x q;
c = measure q;

QCOR_EXPECT_TRUE(c == 1);

gate test r, s {
  x r;
  h s;
}

reset q;
qubit qq;

pow(2) @ test q, qq;

bit xx, yy;
xx = measure q;

QCOR_EXPECT_TRUE(xx == 0);

x qq;
yy = measure qq;
QCOR_EXPECT_TRUE(yy == 1);

qubit a;

s a;
inv @ s a;
bit b;
b = measure a;
QCOR_EXPECT_TRUE(b == 0);

reset a;

bit bb;
pow(2) @ inv @ s a;
measure a -> bb;
QCOR_EXPECT_TRUE(bb == 0);

qubit z, zz;
int count = 0;
for i in [0:100] {
  h z;
  ctrl @ x z, zz;
  bit g[2];
  measure z -> g[0];
  measure zz -> g[1];
  print(g[0], g[1]);
  if (g[0] == 0 && g[1] == 0) {
    count += 1;
  }
  reset z;
  reset zz;
}
print(count);

)#";
  auto mlir = qcor::mlir_compile(check_pow, "check_pow",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";

 
  EXPECT_FALSE(qcor::execute(check_pow, "check_pow"));
}

// Test codegen and optimization for modifiers
TEST(qasm3VisitorTester, checkPowOpt) {
  {
    const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q, qq;

// These are identity ops
pow(2) @ x q;
pow(2) @ h q;
// SS = Z
pow(4) @ s q;
// TTTT = Z
pow(8) @ t q;

// Check gate-def (should be inlined)
gate test r, s {
  x r;
  h s;
}
pow(2) @ test q, qq;
)#";
    auto llvm = qcor::mlir_compile(src, "pow_test", qcor::OutputType::LLVMIR, true);
    std::cout << "LLVM:\n" << llvm << "\n";
    // Get the main kernel
    llvm = llvm.substr(llvm.find("@__internal_mlir_pow_test"));
    const auto last = llvm.find_first_of("}");
    llvm = llvm.substr(0, last + 1);
    std::cout << "LLVM:\n" << llvm << "\n";
    // All instructions are cancelled thanks to inlining/unrolling/gate merge...
    EXPECT_TRUE(llvm.find("__quantum__qis") == std::string::npos);
  }
  {
    const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q, qq;
pow(3) @ x q;
pow(3) @ cx q, qq;
)#";
    auto llvm = qcor::mlir_compile(src, "pow_test", qcor::OutputType::LLVMIR, true);
    std::cout << "LLVM:\n" << llvm << "\n";
    llvm = llvm.substr(llvm.find("@__internal_mlir_pow_test"));
    const auto last = llvm.find_first_of("}");
    llvm = llvm.substr(0, last + 1);
    std::cout << "LLVM:\n" << llvm << "\n";
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__x"), 1);
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__cnot"), 1);
    // No runtime involved
    EXPECT_EQ(countSubstring(llvm, "__quantum__rt__start_pow_u_region"), 0);
    EXPECT_EQ(countSubstring(llvm, "__quantum__rt__end_pow_u_region"), 0);
  }
  {
    // Test code-gen no optimization.
    const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q, qq;
pow(3) @ x q;
pow(3) @ cx q, qq;
)#";
    auto llvm =
        qcor::mlir_compile(src, "pow_test", qcor::OutputType::LLVMIR, false, /* opt-level */0);
    std::cout << "LLVM:\n" << llvm << "\n";
    llvm = llvm.substr(llvm.find("@__internal_mlir_pow_test"));
    const auto last = llvm.find_first_of("}");
    llvm = llvm.substr(0, last + 1);
    std::cout << "LLVM:\n" << llvm << "\n";
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__x"), 1);
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__cnot"), 1);
    // Runtime functions are used:
    EXPECT_EQ(countSubstring(llvm, "__quantum__rt__start_pow_u_region"), 2);
    EXPECT_EQ(countSubstring(llvm, "__quantum__rt__end_pow_u_region"), 2);
  }
}

TEST(qasm3VisitorTester, checkInvOpt) {
  {
    // Note: we tested each gate on different qubits to validate the adjoint
    // mapping (prevent gate merging to operate...)
    const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q[6];
// Self adjoint
inv @ x q[0];
inv @ y q[1];
inv @ z q[2];
inv @ h q[3];
inv @ cx q[4], q[5];
)#";
    auto llvm = qcor::mlir_compile(src, "inv_test", qcor::OutputType::LLVMIR, false);
    std::cout << "LLVM:\n" << llvm << "\n";
    llvm = llvm.substr(llvm.find("@__internal_mlir_inv_test"));
    const auto last = llvm.find_first_of("}");
    llvm = llvm.substr(0, last + 1);
    std::cout << "LLVM:\n" << llvm << "\n";
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__x"), 1);
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__y"), 1);
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__z"), 1);
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__h"), 1);
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__cnot"), 1);
    // No runtime involved
    EXPECT_EQ(countSubstring(llvm, "__quantum__rt__start_adj_u_region"), 0);
    EXPECT_EQ(countSubstring(llvm, "__quantum__rt__end_adj_u_region"), 0);
  }
  {
    // Gate-Adjoint pair
    const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q[2];
inv @ t q[0];
inv @ s q[1];
)#";
    auto llvm =
        qcor::mlir_compile(src, "inv_test", qcor::OutputType::LLVMIR, false);
    std::cout << "LLVM:\n" << llvm << "\n";
    llvm = llvm.substr(llvm.find("@__internal_mlir_inv_test"));
    const auto last = llvm.find_first_of("}");
    llvm = llvm.substr(0, last + 1);
    std::cout << "LLVM:\n" << llvm << "\n";
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__tdg"), 1);
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__sdg"), 1);
    // No runtime involved
    EXPECT_EQ(countSubstring(llvm, "__quantum__rt__start_adj_u_region"), 0);
    EXPECT_EQ(countSubstring(llvm, "__quantum__rt__end_adj_u_region"), 0);
  }
  {
    // Gate-Adjoint pair
    const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q[2];
inv @ tdg q[0];
inv @ sdg q[1];
)#";
    auto llvm =
        qcor::mlir_compile(src, "inv_test", qcor::OutputType::LLVMIR, false);
    std::cout << "LLVM:\n" << llvm << "\n";
    llvm = llvm.substr(llvm.find("@__internal_mlir_inv_test"));
    const auto last = llvm.find_first_of("}");
    llvm = llvm.substr(0, last + 1);
    std::cout << "LLVM:\n" << llvm << "\n";
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__t"), 1);
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__s"), 1);
    // No runtime involved
    EXPECT_EQ(countSubstring(llvm, "__quantum__rt__start_adj_u_region"), 0);
    EXPECT_EQ(countSubstring(llvm, "__quantum__rt__end_adj_u_region"), 0);
  }
  {
    // Parametric gates
    const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q[5];

inv @ rx(1.0) q[0];
inv @ ry(-1.0) q[1];
inv @ rz(1.0) q[2];
inv @ cphase(-1.0) q[3], q[4];
)#";
    auto llvm =
        qcor::mlir_compile(src, "inv_test", qcor::OutputType::LLVMIR, false);
    std::cout << "LLVM:\n" << llvm << "\n";
    llvm = llvm.substr(llvm.find("@__internal_mlir_inv_test"));
    const auto last = llvm.find_first_of("}");
    llvm = llvm.substr(0, last + 1);
    std::cout << "LLVM:\n" << llvm << "\n";
    // Check the angle values change signs
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__rx(double -1.0"), 1);
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__ry(double 1.0"), 1);
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__rz(double -1.0"), 1);
    EXPECT_EQ(countSubstring(llvm, "__quantum__qis__cphase(double 1.0"), 1);

    // No runtime involved
    EXPECT_EQ(countSubstring(llvm, "__quantum__rt__start_adj_u_region"), 0);
    EXPECT_EQ(countSubstring(llvm, "__quantum__rt__end_adj_u_region"), 0);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}