#include <ATen/ops/maximum.h>
#include <c10/core/TensorOptions.h>
#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

#include <cmath>
#include <iostream>

//////////////////////////////////////////////////////////////////////
// Test 2
using torch::Tensor;

class Option {
  public:
    virtual Tensor payoff(Tensor S) = 0; 
};

class EuroOption : public Option {
  private:
    double m_strike;

  public:
    explicit EuroOption(double strike) : m_strike(strike) {}

    Tensor payoff(Tensor S) override {
      return torch::maximum(S - m_strike, torch::zeros(S.sizes()));
    }
};

Tensor standard_normal(int N) {
  Tensor z = torch::randn({N});
  Tensor mean = torch::mean(z);
  Tensor std = torch::std(z);
  return (z - mean) / std;
}

class StochasticProcess {
  public:
    virtual Tensor next(Tensor S0, double dt, int mc_number_of_paths) = 0;
};

class GeometricBrownianMotion : public StochasticProcess {
  private:
    double m_mu;
    double m_sigma;

  public:
    GeometricBrownianMotion(double mu, double sigma) : m_mu(mu), m_sigma(sigma) {}

    Tensor next(Tensor S, double dt, int mc_number_of_paths) override {
      auto z = standard_normal(mc_number_of_paths);
      return S * torch::exp((m_mu - 0.5 * m_sigma * m_sigma)*dt + m_sigma*std::sqrt(dt)*z);
    }
};

Tensor mc_pv(
  Option& option,
  StochasticProcess& process,
  Tensor S0,
  double T,
  double r,
  int mc_number_of_paths,
  int mc_number_of_points
) {
  Tensor S = torch::zeros({mc_number_of_points, mc_number_of_paths});
  S[0] = S0;

  double dt = T / mc_number_of_points;

  for (int t =1; t < mc_number_of_points; ++t) {
    S[t] = process.next(S[t-1], dt, mc_number_of_paths);
  }

  auto discount_factor = std::exp(-r*T);

  auto S_last = S[mc_number_of_points-1];
  std::cout << "S_last.dim(): " << S_last.dim() << 
    "; S_last.size(): " << S_last.size(0) << "; S_last.sizes(): " << S_last.sizes() << "\n";
  
  Tensor payoff = option.payoff(S_last);
  Tensor C = torch::sum(discount_factor*payoff) / mc_number_of_paths;

  return C;
}

void test2() {
  std::cout << "Test 2\n";
  Tensor S0 = torch::tensor({100.0}, torch::requires_grad());
  const double T = 1.0;   // 1Y
  const double r = 0.05;  // 5%
  const double sigma = 0.1; // 10%
  // 32768 65536 131072
  int mc_number_of_paths = 16384;
  int mc_number_of_points = 365;

  auto euro_option = EuroOption(110);

  auto process = GeometricBrownianMotion(r, sigma);

  auto option_price = mc_pv(euro_option, process, S0, T, r, mc_number_of_paths, mc_number_of_points);
  std::cout << "Price: " << option_price << "\n";

  const auto delta = torch::autograd::grad({option_price}, {S0});
  std::cout << "Delta: " << delta << "\n";

  std::cout << "Done: Test 2\n";
}

//////////////////////////////////////////////////////////////////////
// Test 1
void test1() {
  std::cout << "Test 1\n";
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  std::cout << "Autograd prep\n";
  torch::Tensor a = 10*torch::ones({2, 3}, torch::requires_grad());
  //torch::Tensor b = torch::randn({2, 3});
  torch::Tensor c = (a*a + tensor).sum();
  std::cout << "a: " << a << "\n";
  std::cout << "c: " << c << "\n";

  //c.backward(); // a.grad() will now hold the gradient of c w.r.t. a.
  std::cout << "Autograd call\n";
  // \frac{dc}{da}
  const auto dc_da = torch::autograd::grad({c}, {a});
  std::cout << dc_da << "\n";

  std::cout << "Done: Test1\n";
}


int main() {
  std::cout << "******************************************\n";
  test1();
  std::cout << "******************************************\n";
  test2();
  std::cout << "******************************************\n";
  std::cout << "Done: all\n";
}

