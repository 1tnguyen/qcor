#include <xacc.hpp>

#include "clang/Sema/DeclSpec.h"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "token_collector.hpp"
#include "xacc_service.hpp"

TEST(PyXASMTokenCollectorTester, checkSimple) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(R"(
    H(qb[0])
    CX(qb[0],qb[1])
    for i in range(qb.size()):
        X(qb[i])
        X(qb[i])
        Measure(qb[i])
)");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  std::stringstream ss;
  auto xasm_tc = xacc::getService<qcor::TokenCollector>("pyxasm");
  xasm_tc->collect(*PP.get(), cached, {"qb"}, ss);
  std::cout << "heres the test\n";
  std::cout << ss.str() << "\n";

  EXPECT_EQ(R"#(quantum::h(qb[0]);
quantum::cnot(qb[0], qb[1]);
for (int i = 0; i < qb.size(); ++i ) {
quantum::x(qb[i]);
quantum::x(qb[i]);
quantum::mz(qb[i]);
}
)#",
            ss.str());
}

int main(int argc, char **argv) {
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
