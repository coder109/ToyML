#include "include/tensor.h"
#include "include/active_fn.h"
#include "include/backprop.h"
#include "include/loss_fn.h"
#include "include/optimize.h"

int main(int argc, char* argv[]) {
    Tensor* in_tensor = CreateBaseOnArray(2, (int[]){6, 1}, (double[]){1, 2, 3, 4, 5, 6});

    Tensor* layer = CreateBaseOnArray(2, (int[]){1, 1}, (double[]){1});

    Tensor* gt_tensor = CreateBaseOnArray(2, (int[]){6, 1}, (double[]){6, 12, 18, 24, 30, 36});


    double lr = 0.05;
    AdamState* state = CreateAdamOptimizeState(layer);
    SGDMomentumState* momentum_state = CreateSGDMomentumOptimizeState(layer);

    for(int iter = 0; iter < 500; ++iter) {
        Tensor* out_tensor = MatMul(in_tensor, layer);
        Tensor* loss = MSE(gt_tensor, out_tensor);

        Backprop(loss);

        //AdamOptimize(state, lr, 0.9, 0.999);
        SGDMomentumOptimize(momentum_state, lr, 0.9);

        printf("%lf\n", layer->grad[0]);

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