#include "include/tensor.h"
#include "include/active_fn.h"
#include "include/backprop.h"

int main(int argc, char* argv[]) {
    Tensor* tensor = CreateBaseOnArray(2, (int[]){2, 3}, (double[]){1, 2, 3, 4, 5, 6});

    Tensor* sigmoid_tensor = Sigmoid(tensor);
    for(int i = 0; i < GetDataNum(sigmoid_tensor); i++) {
        sigmoid_tensor->grad[i] = 1.0;
    }
    Backprop(sigmoid_tensor);

    PrintTensor(sigmoid_tensor);
    PrintTensor(tensor);

    FreeTensor(tensor);
    FreeTensor(sigmoid_tensor);
    return 0;
}