#include "include/tensor.h"

int main(int argc, char* argv[]) {
    Tensor* tensor1 = CreateZeroTensor(2, (int[]){2, 3});
    Tensor* tensor2 = CreateOneTensor(2, (int[]){2, 3});
    Tensor* bad_te1 = CreateZeroTensor(3, (int[]){2, 3, 4});

    PrintTensor(tensor1);
    PrintTensor(tensor2);
    PrintTensor(bad_te1);

    AddTensorInPlace(tensor1, tensor2);
    SubTensorInPlace(tensor1, tensor2);

    FreeTensor(tensor1);
    FreeTensor(tensor2);
    FreeTensor(bad_te1);
    return 0;
}