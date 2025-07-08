#include "include/tensor.h"

int main(int argc, char* argv[]) {
    Tensor* tensor = CreateBaseOnArray(2, (int[]){2, 3}, (double[]){1, 2, 3, 4, 5, 6});

    PrintTensor(tensor);
    ChangeShape(tensor, 6, (int[]){1, 1, 1, 1, 1, 6});
    PrintTensor(tensor);

    FreeTensor(tensor);
    return 0;
}