#include "include/tensor.h"
#include "include/active_fn.h"
#include "include/backprop.h"
#include "include/loss_fn.h"
#include "include/optimize.h"

int main(int argc, char* argv[]) {
    Tensor* in_tensor = CreateBaseOnArray(2, (int[]){6, 1}, (double[]){1, 2, 3, 4, 5, 6});

    Tensor* layer = CreateBaseOnArray(2, (int[]){1, 1}, (double[]){1});

    Tensor* gt_tensor = CreateBaseOnArray(2, (int[]){6, 1}, (double[]){6, 12, 18, 24, 30, 36});


    double lr = 0.001;
    for(int iter = 0; iter < 200; ++iter) {
        Tensor* out_tensor = MatMul(in_tensor, layer);
        Tensor* loss = MSE(gt_tensor, out_tensor);

        Backprop(loss);

        SGDOptimize(layer, lr);

        printf("%lf\n", layer->grad[0]);

        ZeroGradTensor(layer);
        
        FreeTensor(loss);
        FreeTensor(out_tensor);
    }
    
    PrintTensor(layer);
    FreeTensor(in_tensor);
    FreeTensor(layer);
    FreeTensor(gt_tensor);
    return 0;
}