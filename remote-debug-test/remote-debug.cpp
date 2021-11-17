#include <iostream>
#include <opencv2/core/core.hpp>

int main()
{
    std::cout << "This is a debug test.\n";
    int a{1};
    std::cout << "A is equal to 1.\n";
    int b{2};
    std::cout << "B is equal to 2.\n";
    int ans{a * b};
    std::cout << "A times B equals 2.\n";
}