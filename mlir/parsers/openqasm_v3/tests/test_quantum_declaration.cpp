#include "gtest/gtest.h"
#include "qcor_mlir_api.hpp"

TEST(qasm3VisitorTester, checkQuantumDeclaration) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
qubit single_q;
qubit my_qreg[10];
qreg qq_qreg[25];
)#";
  auto mlir =
      qcor::mlir_compile("qasm3", src, "test", qcor::OutputType::MLIR, true);

  std::cout << "MLIR:\n" << mlir << "\n";

  auto expected = R"#(module  {
  func @main(%arg0: i32, %arg1: !quantum.ArgvType) -> i32 {
    "quantum.init"(%arg0, %arg1) : (i32, !quantum.ArgvType) -> ()
    call @__internal_mlir_test() : () -> ()
    "quantum.finalize"() : () -> ()
    %c0_i32 = constant 0 : i32
    return %c0_i32 : i32
  }
  func @__internal_mlir_test() {
    %0 = "quantum.qalloc"() {name = "single_q", size = 1 : i64} : () -> !quantum.Array
    %c0_i64 = constant 0 : i64
    %1 = "quantum.qextract"(%0, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %2 = "quantum.qalloc"() {name = "my_qreg", size = 10 : i64} : () -> !quantum.Array
    %3 = "quantum.qalloc"() {name = "qq_qreg", size = 25 : i64} : () -> !quantum.Array
    "quantum.dealloc"(%0) : (!quantum.Array) -> ()
    "quantum.dealloc"(%2) : (!quantum.Array) -> ()
    "quantum.dealloc"(%3) : (!quantum.Array) -> ()
    return
  }
  func @test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";
  EXPECT_EQ(expected, mlir);
}

TEST(qasm3VisitorTester, checkClassicalDeclarations) {
  const std::string const_test = R"#(OPENQASM 3;
include "qelib1.inc";
const layers = 22;
const layers2 = layers / 2;
const t = layers * 3;
const d = 1.2;
const tt = d * 33.3;
const mypi = pi / 2;
)#";
  auto mlir = qcor::mlir_compile("qasm3", const_test, "const_test",
                                 qcor::OutputType::MLIR, false);

  std::cout << "Const Test MLIR:\n" << mlir << "\n";

  const std::string expected_const = R"#(module  {
  func @__internal_mlir_const_test() {
    %c22_i64 = constant 22 : i64
    %c2_i64 = constant 2 : i64
    %0 = divi_unsigned %c22_i64, %c2_i64 : i64
    %c3_i64 = constant 3 : i64
    %1 = muli %c22_i64, %c3_i64 : i64
    %cst = constant 1.200000e+00 : f64
    %cst_0 = constant 3.330000e+01 : f64
    %2 = mulf %cst, %cst_0 : f64
    %cst_1 = constant 3.1415926535897931 : f64
    %c2_i64_2 = constant 2 : i64
    %3 = "std.divf"(%cst_1, %c2_i64_2) : (f64, i64) -> f64
    return
  }
  func @const_test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_const_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";
  EXPECT_EQ(expected_const, mlir);

  const std::string var_test = R"#(OPENQASM 3;
include "qelib1.inc";
int[32] i = 10;
float[32] f;
float[64] ff = 3.14;
bit result;
bit results[2];
creg c[22];
bool b;
bool bb = 1;
bool bbb = 0;
)#";
  auto var_mlir = qcor::mlir_compile("qasm3", var_test, "var_test",
                                     qcor::OutputType::MLIR, false);

  std::cout << "Var Test MLIR:\n" << var_mlir << "\n";
  const std::string expected_var = R"#(module  {
  func @__internal_mlir_var_test() {
    %c0_i32 = constant 0 : i32
    %c10_i32 = constant 10 : i32
    %c0_i64 = constant 0 : i64
    %0 = index_cast %c0_i64 : i64 to index
    %1 = alloca() : memref<1xi32>
    store %c10_i32, %1[%0] : memref<1xi32>
    %cst = constant 0.000000e+00 : f32
    %2 = index_cast %c0_i64 : i64 to index
    %3 = alloca() : memref<1xf32>
    store %cst, %3[%2] : memref<1xf32>
    %cst_0 = constant 3.140000e+00 : f64
    %4 = index_cast %c0_i64 : i64 to index
    %5 = alloca() : memref<1xf64>
    store %cst_0, %5[%4] : memref<1xf64>
    %6 = alloc() : memref<1xi1>
    %7 = alloc() : memref<2xi1>
    %8 = alloc() : memref<22xi1>
    %false = constant false
    %9 = index_cast %c0_i64 : i64 to index
    %10 = alloca() : memref<1xi1>
    store %false, %10[%9] : memref<1xi1>
    %true = constant true
    %11 = index_cast %c0_i64 : i64 to index
    %12 = load %true[%11] : i1
    %13 = alloca() : memref<1xi1>
    store %12, %13[%11] : memref<1xi1>
    %false_1 = constant false
    %14 = index_cast %c0_i64 : i64 to index
    %15 = load %false_1[%14] : i1
    %16 = alloca() : memref<1xi1>
    store %15, %16[%14] : memref<1xi1>
    return
  }
  func @var_test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_var_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";

  EXPECT_EQ(expected_var, var_mlir);

  const std::string cast_op_test = R"#(OPENQASM 3;
include "qelib1.inc";
int[32] i = 1;
bool b = bool(i);

)#";
  auto cast_op_mlir = qcor::mlir_compile("qasm3", cast_op_test, "cast_op_test",
                                         qcor::OutputType::MLIR, false);

  std::cout << "Cast Test MLIR:\n" << cast_op_mlir << "\n";
  const std::string expected_cast = R"#(module  {
  func @__internal_mlir_cast_op_test() {
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    %c0_i64 = constant 0 : i64
    %0 = index_cast %c0_i64 : i64 to index
    %1 = alloca() : memref<1xi32>
    store %c1_i32, %1[%0] : memref<1xi32>
    %2 = memref_cast %1 : memref<1xi32> to i1
    %3 = index_cast %c0_i64 : i64 to index
    %4 = load %2[%3] : i1
    %5 = alloca() : memref<1xi1>
    store %4, %5[%3] : memref<1xi1>
    return
  }
  func @cast_op_test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_cast_op_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";
  EXPECT_EQ(expected_cast, cast_op_mlir);
}

TEST(qasm3VisitorTester, checkMeasurements) {
  const std::string measure_test = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
qubit qq[2];

bit r, s;
r = measure q;
s = measure qq[0];

bit rr[2];
rr[0] = measure qq[0];

bit xx[4];
qubit qqq[4];
xx = measure qqq;

bit y, yy[2];
measure q -> y;
measure qq -> yy;
measure qq[0] -> y;
)#";
  auto mlir = qcor::mlir_compile("qasm3", measure_test, "measure_test",
                                 qcor::OutputType::MLIR, false);

  std::cout << "measure_test MLIR:\n" << mlir << "\n";
  const std::string meas_test = R"#(module  {
  func @__internal_mlir_measure_test() {
    %0 = "quantum.qalloc"() {name = "q", size = 1 : i64} : () -> !quantum.Array
    %c0_i64 = constant 0 : i64
    %1 = "quantum.qextract"(%0, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %2 = "quantum.qalloc"() {name = "qq", size = 2 : i64} : () -> !quantum.Array
    %3 = alloc() : memref<1xi1>
    %4 = alloc() : memref<1xi1>
    %5 = "quantum.inst"(%1) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    %6 = index_cast %c0_i64 : i64 to index
    store %5, %3[%6] : memref<1xi1>
    %7 = "quantum.qextract"(%2, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %8 = "quantum.inst"(%7) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    %9 = index_cast %c0_i64 : i64 to index
    store %8, %4[%9] : memref<1xi1>
    %10 = alloc() : memref<2xi1>
    %11 = "quantum.inst"(%7) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    %12 = index_cast %c0_i64 : i64 to index
    store %11, %10[%12] : memref<2xi1>
    %13 = alloc() : memref<4xi1>
    %14 = "quantum.qalloc"() {name = "qqq", size = 4 : i64} : () -> !quantum.Array
    affine.for %arg0 = 0 to 4 {
      %21 = "quantum.qextract"(%14, %arg0) : (!quantum.Array, index) -> !quantum.Qubit
      %22 = "quantum.inst"(%21) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
      store %22, %13[%arg0] : memref<4xi1>
    }
    %15 = alloc() : memref<1xi1>
    %16 = alloc() : memref<2xi1>
    %17 = "quantum.inst"(%1) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    %18 = index_cast %c0_i64 : i64 to index
    store %17, %15[%18] : memref<1xi1>
    affine.for %arg0 = 0 to 2 {
      %21 = "quantum.qextract"(%2, %arg0) : (!quantum.Array, index) -> !quantum.Qubit
      %22 = "quantum.inst"(%21) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
      store %22, %16[%arg0] : memref<2xi1>
    }
    %19 = "quantum.inst"(%7) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    %20 = index_cast %c0_i64 : i64 to index
    store %19, %15[%20] : memref<1xi1>
    "quantum.dealloc"(%0) : (!quantum.Array) -> ()
    "quantum.dealloc"(%2) : (!quantum.Array) -> ()
    "quantum.dealloc"(%14) : (!quantum.Array) -> ()
    return
  }
  func @measure_test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_measure_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";

  EXPECT_EQ(meas_test, mlir);
}

TEST(qasm3VisitorTester, checkQuantumInsts) {
  const std::string qinst_test = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
h q;
ry(2.2) q;

qubit qq[2];
x qq[0];
CX qq[0], qq[1];
U(0.1,0.2,0.3) qq[1];
cx q, qq[1];

)#";
  auto mlir = qcor::mlir_compile("qasm3", qinst_test, "qinst_test",
                                 qcor::OutputType::MLIR, false);

  std::cout << "qinst_test MLIR:\n" << mlir << "\n";
  const std::string expected = R"#(module  {
  func @__internal_mlir_qinst_test() {
    %0 = "quantum.qalloc"() {name = "q", size = 1 : i64} : () -> !quantum.Array
    %c0_i64 = constant 0 : i64
    %1 = "quantum.qextract"(%0, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %2 = "quantum.inst"(%1) {name = "h", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    %cst = constant 2.200000e+00 : f64
    %3 = "quantum.inst"(%1, %cst) {name = "ry", operand_segment_sizes = dense<1> : vector<2xi32>} : (!quantum.Qubit, f64) -> none
    %4 = "quantum.qalloc"() {name = "qq", size = 2 : i64} : () -> !quantum.Array
    %5 = "quantum.qextract"(%4, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %6 = "quantum.inst"(%5) {name = "x", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    %c1_i64 = constant 1 : i64
    %7 = "quantum.qextract"(%4, %c1_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %8 = "quantum.inst"(%5, %7) {name = "cnot", operand_segment_sizes = dense<[2, 0]> : vector<2xi32>} : (!quantum.Qubit, !quantum.Qubit) -> none
    %cst_0 = constant 1.000000e-01 : f64
    %cst_1 = constant 2.000000e-01 : f64
    %cst_2 = constant 3.000000e-01 : f64
    %9 = "quantum.inst"(%7, %cst_0, %cst_1, %cst_2) {name = "u3", operand_segment_sizes = dense<[1, 3]> : vector<2xi32>} : (!quantum.Qubit, f64, f64, f64) -> none
    %10 = "quantum.inst"(%1, %7) {name = "cnot", operand_segment_sizes = dense<[2, 0]> : vector<2xi32>} : (!quantum.Qubit, !quantum.Qubit) -> none
    "quantum.dealloc"(%0) : (!quantum.Array) -> ()
    "quantum.dealloc"(%4) : (!quantum.Array) -> ()
    return
  }
  func @qinst_test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_qinst_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";

  EXPECT_EQ(expected, mlir);
}

TEST(qasm3VisitorTester, checkSubroutine) {
  const std::string subroutine_test = R"#(OPENQASM 3;
include "qelib1.inc";
def xmeasure qubit:q -> bit { h q; return measure q; }
qubit q, qq[2];
bit r, rr[2];

rr[0] = xmeasure q;
r = xmeasure qq[0];
)#";
  auto mlir = qcor::mlir_compile("qasm3", subroutine_test, "subroutine_test",
                                 qcor::OutputType::MLIR, false);

  std::cout << "subroutine_test MLIR:\n" << mlir << "\n";

  const std::string s = R"#(module  {
  func @__internal_mlir_subroutine_test() {
    %0 = "quantum.qalloc"() {name = "q", size = 1 : i64} : () -> !quantum.Array
    %c0_i64 = constant 0 : i64
    %1 = "quantum.qextract"(%0, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %2 = "quantum.qalloc"() {name = "qq", size = 2 : i64} : () -> !quantum.Array
    %3 = alloc() : memref<1xi1>
    %4 = alloc() : memref<2xi1>
    %5 = call @xmeasure(%1) : (!quantum.Qubit) -> i1
    store %5, %4[%c0_i64] : memref<2xi1>
    %6 = "quantum.qextract"(%2, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %7 = call @xmeasure(%6) : (!quantum.Qubit) -> i1
    store %7, %3[%c0_i64] : memref<1xi1>
    "quantum.dealloc"(%0) : (!quantum.Array) -> ()
    "quantum.dealloc"(%2) : (!quantum.Array) -> ()
    return
  }
  func @subroutine_test(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_subroutine_test() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
  func @xmeasure(%arg0: !quantum.Qubit) -> i1 {
    %0 = "quantum.inst"(%arg0) {name = "h", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    %1 = "quantum.inst"(%arg0) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    return %1 : i1
  }
}
)#";
  EXPECT_EQ(s, mlir);

  const std::string subroutine_test2 = R"#(OPENQASM 3;
include "qelib1.inc";
def xcheck qubit[4]:d, qubit:a -> bit {
  // reset a;
  for i in [0: 3] cx d[i], a;
  return measure a;
}
qubit q;
const n = 10;
def parity(bit[n]:cin) -> bit {
  bit c;
  for i in [0: n-1] {
    c ^= cin[i];
  }
  return c;
}
)#";
  auto mlir2 = qcor::mlir_compile("qasm3", subroutine_test2, "subroutine_test2",
                                  qcor::OutputType::MLIR, false);

  std::cout << "subroutine_test MLIR:\n" << mlir2 << "\n";

  const std::string expected2 = R"#(module  {
  func @__internal_mlir_subroutine_test2() {
    %0 = "quantum.qalloc"() {name = "q", size = 1 : i64} : () -> !quantum.Array
    %c0_i64 = constant 0 : i64
    %1 = "quantum.qextract"(%0, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %c10_i64 = constant 10 : i64
    "quantum.dealloc"(%0) : (!quantum.Array) -> ()
    return
  }
  func @subroutine_test2(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_subroutine_test2() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
  func @xcheck(%arg0: !quantum.Qubit, %arg1: !quantum.Qubit) -> i1 {
    affine.for %arg2 = 0 to 3 {
      %1 = "quantum.qextract"(%arg0, %arg2) : (!quantum.Qubit, index) -> !quantum.Qubit
      %2 = "quantum.inst"(%1, %arg1) {name = "cnot", operand_segment_sizes = dense<[2, 0]> : vector<2xi32>} : (!quantum.Qubit, !quantum.Qubit) -> none
    }
    %0 = "quantum.inst"(%arg1) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    return %0 : i1
  }
  func @parity(%arg0: memref<10xi1>) -> i1 {
    %0 = alloc() : memref<1xi1>
    affine.for %arg1 = 0 to 9 {
      %3 = load %arg0[%arg1] : memref<10xi1>
      %4 = index_cast %c0_i64 : i64 to index
      %5 = load %0[%4] : memref<1xi1>
      %6 = xor %5, %3 : i1
      %7 = index_cast %c0_i64 : i64 to index
      store %6, %0[%7] : memref<1xi1>
    }
    %1 = index_cast %c0_i64 : i64 to index
    %2 = load %0[%1] : memref<1xi1>
    return %2 : i1
  }
}
)#";

  EXPECT_EQ(expected2, mlir2);
}

TEST(qasm3VisitorTester, checkAddAssignment) {
  const std::string add_assignment = R"#(OPENQASM 3;
include "qelib1.inc";
int[64] i = 1;
i += 2;
)#";
  auto mlir = qcor::mlir_compile("qasm3", add_assignment, "add_assignment",
                                 qcor::OutputType::MLIR, false);
  std::cout << "add_assignment MLIR:\n" << mlir << "\n";

  const std::string s = R"#(module  {
  func @__internal_mlir_add_assignment() {
    %c0_i64 = constant 0 : i64
    %c1_i64 = constant 1 : i64
    %0 = index_cast %c0_i64 : i64 to index
    %1 = alloca() : memref<1xi64>
    store %c1_i64, %1[%0] : memref<1xi64>
    %c2_i64 = constant 2 : i64
    %2 = index_cast %c0_i64 : i64 to index
    %3 = load %1[%2] : memref<1xi64>
    %4 = addi %3, %c2_i64 : i64
    %5 = index_cast %c0_i64 : i64 to index
    store %4, %1[%5] : memref<1xi64>
    return
  }
  func @add_assignment(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_add_assignment() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";
  EXPECT_EQ(s, mlir);
  // auto mlir2 = qcor::mlir_compile("qasm3", add_assignment, "add_assignment",
  //                                qcor::OutputType::LLVMMLIR, false);
  // std::cout << "add_assignment MLIR:\n" << mlir2 << "\n";

  //   auto mlir3 = qcor::mlir_compile("qasm3", add_assignment,
  //   "add_assignment",
  //                                qcor::OutputType::LLVMIR, false);
  // std::cout << "add_assignment MLIR:\n" << mlir3 << "\n";
}

TEST(qasm3VisitorTester, checkIfStmt) {
  const std::string if_stmt = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q, s, qq[2];
const layers = 2;

bit c;
c = measure q;

if (c == 1) {
    z s;
}

if (layers == 2) {
    z s;
} else {
    x s;
}

bit cc[2];
cc = measure qq;
if ( cc[1] == 1) {
  ry(2.2) s;
}

if ( 1 == cc[0] ) {
  ry(2.2) s;
}
)#";
  auto mlir = qcor::mlir_compile("qasm3", if_stmt, "if_stmt",
                                 qcor::OutputType::MLIR, false);
  std::cout << "if_stmt MLIR:\n" << mlir << "\n";
  const std::string test = R"#(module  {
  func @__internal_mlir_if_stmt() {
    %0 = "quantum.qalloc"() {name = "q", size = 1 : i64} : () -> !quantum.Array
    %c0_i64 = constant 0 : i64
    %1 = "quantum.qextract"(%0, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %2 = "quantum.qalloc"() {name = "s", size = 1 : i64} : () -> !quantum.Array
    %3 = "quantum.qextract"(%2, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %4 = "quantum.qalloc"() {name = "qq", size = 2 : i64} : () -> !quantum.Array
    %c2_i64 = constant 2 : i64
    %5 = alloc() : memref<1xi1>
    %6 = "quantum.inst"(%1) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    %7 = index_cast %c0_i64 : i64 to index
    store %6, %5[%7] : memref<1xi1>
    %8 = index_cast %c0_i64 : i64 to index
    %9 = load %5[%8] : memref<1xi1>
    %c1_i64 = constant 1 : i64
    %10 = cmpi "eq", %9, %c1_i64 : i1
    cond_br %10, ^bb1, ^bb2
    %c2_i64_0 = constant 2 : i64
    %11 = cmpi "eq", %c2_i64, %c2_i64_0 : i64
    cond_br %11, ^bb3, ^bb4
    %12 = alloc() : memref<2xi1>
    affine.for %arg0 = 0 to 2 {
      %22 = "quantum.qextract"(%4, %arg0) : (!quantum.Array, index) -> !quantum.Qubit
      %23 = "quantum.inst"(%22) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
      store %23, %12[%arg0] : memref<2xi1>
    }
    %c1_i64_1 = constant 1 : i64
    %13 = load %12[%c1_i64_1] : memref<2xi1>
    %c1_i64_2 = constant 1 : i64
    %14 = cmpi "eq", %13, %c1_i64_2 : i1
    cond_br %14, ^bb5, ^bb6
    %c1_i64_3 = constant 1 : i64
    %c0_i64_4 = constant 0 : i64
    %15 = load %12[%c0_i64_4] : memref<2xi1>
    %16 = cmpi "eq", %c1_i64_3, %15 : i64
    cond_br %16, ^bb7, ^bb8
    "quantum.dealloc"(%0) : (!quantum.Array) -> ()
    "quantum.dealloc"(%2) : (!quantum.Array) -> ()
    "quantum.dealloc"(%4) : (!quantum.Array) -> ()
    return
  ^bb1:  // pred: ^bb0
    %17 = "quantum.inst"(%3) {name = "z", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    return
  ^bb2:  // pred: ^bb0
    return
  ^bb3:  // pred: ^bb0
    %18 = "quantum.inst"(%3) {name = "z", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    return
  ^bb4:  // pred: ^bb0
    %19 = "quantum.inst"(%3) {name = "x", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    return
  ^bb5:  // pred: ^bb0
    %cst = constant 2.200000e+00 : f64
    %20 = "quantum.inst"(%3, %cst) {name = "ry", operand_segment_sizes = dense<1> : vector<2xi32>} : (!quantum.Qubit, f64) -> none
    return
  ^bb6:  // pred: ^bb0
    return
  ^bb7:  // pred: ^bb0
    %cst_5 = constant 2.200000e+00 : f64
    %21 = "quantum.inst"(%3, %cst_5) {name = "ry", operand_segment_sizes = dense<1> : vector<2xi32>} : (!quantum.Qubit, f64) -> none
    return
  ^bb8:  // pred: ^bb0
    return
  }
  func @if_stmt(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_if_stmt() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";
  EXPECT_EQ(test, mlir);
}

TEST(qasm3VisitorTester, checkLoopStmt) {
  const std::string for_stmt = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q[4];

for i in {1,2,3} {
  x q[i];
}

h q[0];
bit c;
c = measure q[0];

for j in [0:2:4] {
  x q[j];
}


for j in [0:4] {
  x q[j];
}

for i in [0:4] {
 for j in {1,2,3} {
   cx q[i],q[j];
 }
}

qubit qq[12];
int[32] i = 1;
while ( i < 10 ){
  i += 1;
  x qq[i];
  
  if (i == 5) {
    break;
  }
}
)#";
  auto mlir = qcor::mlir_compile("qasm3", for_stmt, "for_stmt",
                                 qcor::OutputType::MLIR, false);
  std::cout << "for_stmt MLIR:\n" << mlir << "\n";

  const std::string expected = R"#(module  {
  func @__internal_mlir_for_stmt() {
    %0 = "quantum.qalloc"() {name = "q", size = 4 : i64} : () -> !quantum.Array
    %1 = alloc() : memref<3xi64>
    %c1_i64 = constant 1 : i64
    %c0_i64 = constant 0 : i64
    store %c1_i64, %1[%c0_i64] : memref<3xi64>
    %c2_i64 = constant 2 : i64
    store %c2_i64, %1[%c1_i64] : memref<3xi64>
    %c3_i64 = constant 3 : i64
    store %c3_i64, %1[%c2_i64] : memref<3xi64>
    affine.for %arg0 = 0 to 3 {
      %10 = load %1[%arg0] : memref<3xi64>
      %11 = "quantum.qextract"(%0, %10) : (!quantum.Array, i64) -> !quantum.Qubit
      %12 = "quantum.inst"(%11) {name = "x", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    }
    %2 = "quantum.qextract"(%0, %c0_i64) : (!quantum.Array, i64) -> !quantum.Qubit
    %3 = "quantum.inst"(%2) {name = "h", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    %4 = alloc() : memref<1xi1>
    %5 = "quantum.inst"(%2) {name = "mz", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> i1
    %6 = index_cast %c0_i64 : i64 to index
    store %5, %4[%6] : memref<1xi1>
    affine.for %arg0 = 0 to 4 step 2 {
      %10 = "quantum.qextract"(%0, %arg0) : (!quantum.Array, index) -> !quantum.Qubit
      %11 = "quantum.inst"(%10) {name = "x", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    }
    affine.for %arg0 = 0 to 4 {
      %10 = "quantum.qextract"(%0, %arg0) : (!quantum.Array, index) -> !quantum.Qubit
      %11 = "quantum.inst"(%10) {name = "x", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
    }
    affine.for %arg0 = 0 to 4 {
      %10 = alloc() : memref<3xi64>
      %c1_i64_0 = constant 1 : i64
      store %c1_i64_0, %10[%c0_i64] : memref<3xi64>
      %c2_i64_1 = constant 2 : i64
      store %c2_i64_1, %10[%c1_i64] : memref<3xi64>
      %c3_i64_2 = constant 3 : i64
      store %c3_i64_2, %10[%c2_i64] : memref<3xi64>
      affine.for %arg1 = 0 to 3 {
        %11 = load %10[%arg1] : memref<3xi64>
        %12 = "quantum.qextract"(%0, %arg0) : (!quantum.Array, index) -> !quantum.Qubit
        %13 = "quantum.qextract"(%0, %11) : (!quantum.Array, i64) -> !quantum.Qubit
        %14 = "quantum.inst"(%12, %13) {name = "cnot", operand_segment_sizes = dense<[2, 0]> : vector<2xi32>} : (!quantum.Qubit, !quantum.Qubit) -> none
      }
    }
    %7 = "quantum.qalloc"() {name = "qq", size = 12 : i64} : () -> !quantum.Array
    %c0_i32 = constant 0 : i32
    %c1_i32 = constant 1 : i32
    %8 = index_cast %c0_i64 : i64 to index
    %9 = alloca() : memref<1xi32>
    store %c1_i32, %9[%8] : memref<1xi32>
    affine.for %arg0 = 0 to -1 {
      %c1_i32_0 = constant 1 : i32
      %10 = index_cast %c0_i64 : i64 to index
      %11 = load %9[%10] : memref<1xi32>
      %12 = addi %11, %c1_i32_0 : i32
      %13 = index_cast %c0_i64 : i64 to index
      store %12, %9[%13] : memref<1xi32>
      %14 = index_cast %c0_i64 : i64 to index
      %15 = load %9[%14] : memref<1xi32>
      %16 = "quantum.qextract"(%7, %15) : (!quantum.Array, i32) -> !quantum.Qubit
      %17 = "quantum.inst"(%16) {name = "x", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (!quantum.Qubit) -> none
      %18 = index_cast %c0_i64 : i64 to index
      %19 = load %9[%18] : memref<1xi32>
      %c5_i64 = constant 5 : i64
      %20 = cmpi "eq", %19, %c5_i64 : i32
      cond_br %20, ^bb1, ^bb2
      %21 = index_cast %c0_i64 : i64 to index
      %22 = load %9[%21] : memref<1xi32>
      %c10_i64 = constant 10 : i64
      %23 = cmpi "slt", %22, %c10_i64 : i32
      cond_br %23, ^bb3, ^bb4
    ^bb1:  // pred: ^bb0
      affine.yield
      return
    ^bb2:  // pred: ^bb0
      return
    ^bb3:  // pred: ^bb0
      return
    ^bb4:  // pred: ^bb0
      affine.yield
    }
    "quantum.dealloc"(%0) : (!quantum.Array) -> ()
    "quantum.dealloc"(%7) : (!quantum.Array) -> ()
    return
  }
  func @for_stmt(%arg0: !quantum.qreg) {
    "quantum.set_qreg"(%arg0) : (!quantum.qreg) -> ()
    call @__internal_mlir_for_stmt() : () -> ()
    "quantum.finalize"() : () -> ()
    return
  }
}
)#";
  EXPECT_EQ(expected, mlir);

  const std::string for_stmt2 = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
bit result;

int[32] i = 0;
// Keep applying hadamards and measuring a qubit
// until 10, |1>s are measured
while (i < 10) {
    h q;
    result = measure q;
    if (result) {
        i++;
    }
}
)#";
  auto mlir2 = qcor::mlir_compile("qasm3", for_stmt2, "for_stmt2",
                                  qcor::OutputType::MLIR, false);
  std::cout << "for_stmt2 MLIR:\n" << mlir2 << "\n";
}

TEST(qasm3VisitorTester, checkPrint) {
  const std::string print_stmt = R"#(OPENQASM 3;
include "qelib1.inc";
int[32] i = 22;
float[32] j = 2.2;
print(i);
print(j);
print(i,j);
print("hello world");
print("can print with other vals", i, j);
)#";
  auto mlir = qcor::mlir_compile("qasm3", print_stmt, "print_stmt",
                                 qcor::OutputType::MLIR, false);
  std::cout << mlir << "\n";

  auto mlir2 = qcor::mlir_compile("qasm3", print_stmt, "print_stmt",
                                  qcor::OutputType::LLVMMLIR, true);
  std::cout << mlir2 << "\n";

  auto mlir3 = qcor::mlir_compile("qasm3", print_stmt, "print_stmt",
                                  qcor::OutputType::LLVMIR, true);
  std::cout << mlir3 << "\n";
}
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
