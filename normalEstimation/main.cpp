#include <iostream>

extern void launchKernel();

int main()
{
    std::cout << "Hello world!" << std::endl;
    launchKernel();
    return 0;
}