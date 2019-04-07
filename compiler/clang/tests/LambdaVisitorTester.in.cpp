#include <gtest/gtest.h>
#include <llvm-9/llvm/Support/raw_ostream.h>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendOptions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "FuzzyParsingExternalSemaSource.hpp"
#include "LambdaVisitor.hpp"
#include "QCORASTConsumer.hpp"
#include "QCORPluginAction.hpp"

#include "CNOT.hpp"
#include "CountGatesOfTypeVisitor.hpp"
#include "Hadamard.hpp"
#include "Measure.hpp"
#include "XACC.hpp"
#include "xacc_service.hpp"

#include "IRGenerator.hpp"
#include "qcor.hpp"
#include "clang/Parse/ParseAST.h"

#include <fstream>
using namespace llvm;
using namespace clang;
using namespace qcor;

class TestQCORFrontendAction : public clang::ASTFrontendAction {

public:
  TestQCORFrontendAction(Rewriter &rw) : rewriter(rw) {}

protected:
  Rewriter &rewriter;
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef /* dummy */) override {
    return llvm::make_unique<compiler::QCORASTConsumer>(Compiler, rewriter);
  }

  void ExecuteAction() override {
    CompilerInstance &CI = getCompilerInstance();
    CI.createSema(getTranslationUnitKind(), nullptr);
    compiler::FuzzyParsingExternalSemaSource fuzzyParser(CI.getASTContext());
    CI.getSema().addExternalSource(&fuzzyParser);

    rewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());

    ParseAST(CI.getSema());
    CI.getDiagnosticClient().EndSourceFile();

    std::error_code error_code;
    llvm::raw_fd_ostream outFile(".output.cpp", error_code,
                                 llvm::sys::fs::F_None);
    rewriter.getEditBuffer(CI.getSourceManager().getMainFileID())
        .write(outFile);
  }
};

const std::string bell = R"bell(
int main() {
    auto l = []() {
        H(0);
        CX(0,1);
        Measure(0);
    };
    return 0;
})bell";
const std::string param0 = R"param0(
int main() {
    auto l = [&](double t) {
        X(0);
        Ry(t, 1);
        CX(1,0);
    };
    return 0;
})param0";

const std::string hwe0 = R"hwe0(#include <vector>
int main(int argc, char** argv){
    auto l = [&](std::vector<double> x) {
        hwe(x, {{"n-qubits",2},{"layers",1}});
    };
    return 0;
})hwe0";

const std::string hwe1 = R"hwe1(#include <vector>
int main(int argc, char** argv){
    int nq = 2;
    auto l = [&](std::vector<double> x) {
        hwe(x, {{"n-qubits", nq},{"layers",1}});
    };
    return 0;
})hwe1";

const std::string hwe2 = R"hwe2(#include <vector>
int main(int argc, char** argv){
    int nq = argc;
    auto l = [&](std::vector<double> x) {
        hwe(x, {{"n-qubits", nq},{"layers",1}});
    };
    return 0;
})hwe2";
const std::string hwe3 = R"hwe3(#include <vector>
int main(int argc, char** argv){
    int nq = argc;
    std::vector<std::pair<int,int>> c{{1,0}};
    auto l = [&](std::vector<double> x) {
        hwe(x, {{"n-qubits", nq},{"layers",1},{"coupling", c} });
    };
    return 0;
})hwe3";
TEST(LambdaVisitorTester, checkSimple) {
  Rewriter rewriter1, rewriter2;
  auto action1 = new TestQCORFrontendAction(rewriter1);
  auto action2 = new TestQCORFrontendAction(rewriter2);

  xacc::setOption("qcor-compiled-filename", "lambda_visitor_tester");

  std::vector<std::string> args{"-std=c++11"};

  EXPECT_TRUE(tooling::runToolOnCodeWithArgs(action1, bell, args));

  const std::string expectedSrc = R"expectedSrc(
int main() {
    auto l = [&](){return "lambda_visitor_tester";};
    return 0;
})expectedSrc";
  std::ifstream t(".output.cpp");
  std::string src((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());
  std::remove(".output.cpp");

  EXPECT_EQ(expectedSrc, src);

  auto function = qcor::loadCompiledCircuit("lambda_visitor_tester");
  xacc::quantum::CountGatesOfTypeVisitor<xacc::quantum::Hadamard> h(function);
  xacc::quantum::CountGatesOfTypeVisitor<xacc::quantum::CNOT> cx(function);
  xacc::quantum::CountGatesOfTypeVisitor<xacc::quantum::Measure> m(function);

  EXPECT_EQ(1, h.countGates());
  EXPECT_EQ(1, cx.countGates());
  EXPECT_EQ(1, m.countGates());

  EXPECT_TRUE(tooling::runToolOnCodeWithArgs(action2, param0, args));

  std::ifstream t2(".output.cpp");
  std::string src2((std::istreambuf_iterator<char>(t2)),
                   std::istreambuf_iterator<char>());
  std::remove(".output.cpp");

  EXPECT_EQ(expectedSrc, src2);
}

TEST(LambdaVisitorTester, checkGenerator) {
    Rewriter rewriter1, rewriter2;
    auto action1 = new TestQCORFrontendAction(rewriter1);
    auto action2 = new TestQCORFrontendAction(rewriter2);

    xacc::setOption("qcor-compiled-filename", "lambda_visitor_tester");

    std::vector<std::string> args{"-std=c++11"};

    std::cout << "Source Code:\n" << hwe0 << "\n";
    // first case, I know compile time values, so ahead-of-time compilation
    EXPECT_TRUE(tooling::runToolOnCodeWithArgs(action1, hwe0, args));

    const std::string expectedSrc = R"expectedSrc(#include <vector>
int main(int argc, char** argv){
    auto l = [&](){return "lambda_visitor_tester";};
    return 0;
})expectedSrc";

    std::ifstream t(".output.cpp");
    std::string src((std::istreambuf_iterator<char>(t)),
                    std::istreambuf_iterator<char>());
    std::remove(".output.cpp");

    std::cout << "SOURCE:\n" << src << "\n";
    EXPECT_EQ(expectedSrc, src);

    EXPECT_TRUE(tooling::runToolOnCodeWithArgs(action2, hwe1, args));

    std::ifstream t1(".output.cpp");
    std::string src2((std::istreambuf_iterator<char>(t1)),
                     std::istreambuf_iterator<char>());
    std::remove(".output.cpp");

    const std::string expectedSrc2 = R"expectedSrc2(#include <vector>
int main(int argc, char** argv){
    int nq = 2;
    auto l = [&](){return "lambda_visitor_tester";};
    return 0;
})expectedSrc2";
    EXPECT_EQ(expectedSrc2, src2);
}


TEST(LambdaVisitorTester, checkRuntimeGenerator) {
    Rewriter rewriter1, rewriter2;
    auto action1 = new TestQCORFrontendAction(rewriter1);
    auto action2 = new TestQCORFrontendAction(rewriter2);

    xacc::setOption("qcor-compiled-filename", "lambda_visitor_tester");

    std::vector<std::string> args{"-std=c++11"};

    std::cout << "Source Code:\n" << hwe2 << "\n";
    // first case, I know compile time values, so ahead-of-time compilation
    EXPECT_TRUE(tooling::runToolOnCodeWithArgs(action1, hwe2, args));

    std::ifstream t1(".output.cpp");
    std::string src2((std::istreambuf_iterator<char>(t1)),
                     std::istreambuf_iterator<char>());
    std::remove(".output.cpp");

    std::cout << "HELLO:\n" << src2 <<"\n";
    const std::string exp1 = R"exp1(#include <vector>
int main(int argc, char** argv){
    int nq = argc;
    auto l = [&]() {
qcor::storeRuntimeVariable("n-qubits", nq);
return "lambda_visitor_tester";
};
    return 0;
})exp1";

    EXPECT_EQ(exp1,src2);
}


TEST(LambdaVisitorTester, checkRuntimeGeneratorWithVectorPair) {
    Rewriter rewriter1, rewriter2;
    auto action1 = new TestQCORFrontendAction(rewriter1);

    xacc::setOption("qcor-compiled-filename", "lambda_visitor_tester");

    std::vector<std::string> args{"-std=c++11"};

    std::cout << "Source Code:\n" << hwe3 << "\n";
    // first case, I know compile time values, so ahead-of-time compilation
    EXPECT_TRUE(tooling::runToolOnCodeWithArgs(action1, hwe3, args));

    std::ifstream t1(".output.cpp");
    std::string src2((std::istreambuf_iterator<char>(t1)),
                     std::istreambuf_iterator<char>());
    std::remove(".output.cpp");

    std::cout << "HELLO:\n" << src2 <<"\n";
    const std::string exp1 = R"exp1(#include <vector>
int main(int argc, char** argv){
    int nq = argc;
    auto l = [&]() {
qcor::storeRuntimeVariable("n-qubits", nq);
return "lambda_visitor_tester";
};
    return 0;
})exp1";

    // EXPECT_EQ(exp1,src2);
}
int main(int argc, char **argv) {
  qcor::Initialize(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
