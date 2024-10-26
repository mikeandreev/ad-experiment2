#include <torch/torch.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

#include <cmath>
#include <iostream>


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
    std::cout << "******************************************\n";
    std::cout << "Done: all\n";
}

