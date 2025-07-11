#include "include/tensor.h"
#include "include/active_fn.h"
#include "include/backprop.h"
#include "include/loss_fn.h"
#include "include/optimize.h"

int main(int argc, char* argv[]) {
    Tensor* in_tensor = CreateBaseOnArray(2, (int[]){3, 3}, (double[]){1, 2, 3, 4, 5, 6, 7, 8, 9});

    Tensor* layer = CreateBaseOnArray(2, (int[]){3, 3}, (double[]){0, 0, 0, 0, 0, 0, 0, 0, 0});

    Tensor* gt_tensor = CreateBaseOnArray(2, (int[]){3, 3}, (double[]){1, 2, 3, 4, 5, 6, 7, 8, 9});


    double lr = 0.02;
    for(int iter = 0; iter < 1000; ++iter) {
        Tensor* out_tensor = MatMul(in_tensor, layer);
        Tensor* loss = MSE(gt_tensor, out_tensor);

        Backprop(loss);

        SGDOptimize(layer, lr);

        ZeroGradTensor(layer);
        
        free(loss);
        free(out_tensor);
    }

    PrintTensor(layer);
    FreeTensor(in_tensor);
    FreeTensor(layer);
    FreeTensor(gt_tensor);
    return 0;
}