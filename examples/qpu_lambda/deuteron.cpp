#include "qcor.hpp"
using namespace qcor;

int main() {
  qpu_lambda<> ansatz_X0X1(
      [](qreg q, double x) {
        qpu_lambda_body({
          X(q[0]);
          Ry(q[1], x);
          CX(q[1], q[0]);
          H(q);
          Measure(q);
        })
      },
      qpu_lambda_variables({"q", "x"}, {}));

  OptFunction obj(
      [&](const std::vector<double> &x, std::vector<double> &) {
        print("running ", x[0]);
        auto q = qalloc(2);
        ansatz_X0X1(q, x[0]);
        auto exp = q.exp_val_z();
        print(x[0], exp);
        return exp;
      },
      1);

  auto optimizer = createOptimizer(
      "nlopt",
      {{"initial-parameters", std::vector<double>{1.2}}, {"maxeval", 10}});
  auto [opt_val, opt_params] = optimizer->optimize(obj);
  print("opt_val = ", opt_val);
}