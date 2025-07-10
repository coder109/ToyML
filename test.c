#include "include/tensor.h"
#include "include/active_fn.h"
#include "include/backprop.h"
#include "include/loss_fn.h"

int main(int argc, char* argv[]) {
    Tensor* in_tensor = CreateBaseOnArray(2, (int[]){6, 1}, (double[]){1, 2, 3, 4, 5, 6});

    Tensor* layer = CreateBaseOnArray(2, (int[]){1, 1}, (double[]){1});

    Tensor* gt_tensor = CreateBaseOnArray(2, (int[]){6, 1}, (double[]){6, 12, 18, 24, 30, 36});


    double lr = 0.1;
    for(int iter = 0; iter < 100; ++iter) {
        Tensor* out_tensor = MatMul(in_tensor, layer);
        Tensor* loss = MSE(gt_tensor, out_tensor);

        Backprop(loss);

        // Update Layer
        for(int i = 0; i < layer->shape[0]; i++) {
            for(int j = 0; j < layer->shape[1]; j++) {
                layer->data[GetElemIdxBasedOnRowCol(layer, i, j)] -= lr * layer->grad[GetElemIdxBasedOnRowCol(layer, i, j)];

                lr /= 2.;
            }
        }

        printf("Grad of layer: %lf\n", layer->grad[0]);

        free(loss);
        free(out_tensor);
    }

    FreeTensor(in_tensor);
    FreeTensor(layer);
    FreeTensor(gt_tensor);
    return 0;
}