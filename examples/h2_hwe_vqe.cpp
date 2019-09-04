#include "qcor.hpp"

int main(int argc, char **argv) {

  qcor::Initialize(argc, argv);

  auto optimizer =
      qcor::getOptimizer("nlopt", {std::make_pair("nlopt-optimizer", "cobyla"),
                                   std::make_pair("nlopt-maxeval", 2000)});

  const std::string src = R"src(0.7080240949826064
- 1.248846801817026 0^ 0
- 1.248846801817026 1^ 1
- 0.4796778151607899 2^ 2
- 0.4796778151607899 3^ 3
+ 0.33667197218932576 0^ 1^ 1 0
+ 0.0908126658307406 0^ 1^ 3 2
+ 0.09081266583074038 0^ 2^ 0 2
+ 0.331213646878486 0^ 2^ 2 0
+ 0.09081266583074038 0^ 3^ 1 2
+ 0.331213646878486 0^ 3^ 3 0
+ 0.33667197218932576 1^ 0^ 0 1
+ 0.0908126658307406 1^ 0^ 2 3
+ 0.09081266583074038 1^ 2^ 0 3
+ 0.331213646878486 1^ 2^ 2 1
+ 0.09081266583074038 1^ 3^ 1 3
+ 0.331213646878486 1^ 3^ 3 1
+ 0.331213646878486 2^ 0^ 0 2
+ 0.09081266583074052 2^ 0^ 2 0
+ 0.331213646878486 2^ 1^ 1 2
+ 0.09081266583074052 2^ 1^ 3 0
+ 0.09081266583074048 2^ 3^ 1 0
+ 0.34814578469185886 2^ 3^ 3 2
+ 0.331213646878486 3^ 0^ 0 3
+ 0.09081266583074052 3^ 0^ 2 1
+ 0.331213646878486 3^ 1^ 1 3
+ 0.09081266583074052 3^ 1^ 3 1
+ 0.09081266583074048 3^ 2^ 0 1
+ 0.34814578469185886 3^ 2^ 2 3)src";

  auto op = qcor::getObservable("fermion", src);

  int nq = op->nBits();

  std::vector<std::pair<int, int>> coupling{{0, 1}, {1, 2}, {2, 3}};

  auto future = qcor::submit([&](qcor::qpu_handler &qh) {
    qh.vqe(
        [&](qbit q, std::vector<double> x) {
          X(q[0]);
          X(q[1]);
          hwe(q, x, {{"nq", nq}, {"layers", 1}, {"coupling", coupling}});
        },
        op, optimizer, std::vector<double>{});
  });

  auto results = future.get();
  auto energy = mpark::get<double>(results->getInformation("opt-val"));
  std::cout << "Results: " << energy << "\n";
  //   results->print();
}
