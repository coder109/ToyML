#include "include/tensor.h"
#include "include/active_fn.h"
#include "include/backprop.h"
#include "include/loss_fn.h"
#include "include/optimize.h"
#include "include/file.h"

int main(int argc, char* argv[]) {
    Tensor* random_tensor = CreateRandom(2, (int[]){2, 3});
    
    SaveTensor(random_tensor, "test.txt");
    return 0;
}