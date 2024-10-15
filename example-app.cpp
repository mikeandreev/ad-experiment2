#include <torch/torch.h>

#define TORCH_USE_CXX11_ABI 0
#include <iostream>

int main() {
    std::cout << "Test 1\n";
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    std::cout << "Autograd prep\n";
    torch::Tensor a = 10*torch::ones({2, 3}, torch::requires_grad());
    //torch::Tensor b = torch::randn({2, 3});
    torch::Tensor c = (a*a*tensor).sum();
    std::cout << "a: " << a << "\n";
    std::cout << "c: " << c << "\n";
    std::cout << "Autograd call\n";
    //torch::autograd::grad({c}, {a});
    const auto a_grad = torch::autograd::grad({c}, {a});
    std::cout << "a_grad: " << a_grad.at(0) << "\n";
    //c.backward(); // a.grad() will now hold the gradient of c w.r.t. a.
    //std::cout << a << "\n";
    std::cout << "Done\n";
}
