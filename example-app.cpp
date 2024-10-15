#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

#include <iostream>

int main() {
    std::cout << "Test 1\n";
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    std::cout << "Autograd prep\n";
    torch::Tensor a = 10*torch::ones({2, 3}, torch::requires_grad());
    //torch::Tensor b = torch::randn({2, 3});
    torch::Tensor c = a*a*tensor;
    std::cout << "a: " << a << "\n";
    std::cout << "c: " << c << "\n";
    std::cout << "Autograd call\n";
    torch::autograd::grad({c}, {a});
    //c.backward(); // a.grad() will now hold the gradient of c w.r.t. a.
    //std::cout << a << "\n";
    std::cout << "Done\n";
}
