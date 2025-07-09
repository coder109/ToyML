#include "include/tensor.h"
#include "include/active_fn.h"
#include "include/backprop.h"

int main(int argc, char* argv[]) {
    Tensor* raw_tensor = CreateBaseOnArray(2, (int[]){6, 1}, (double[]){1, 2, 3, 4, 5, 6});

    Tensor* gt_tensor = CreateBaseOnArray(2, (int[]){2, 3}, (double[]){6, 12, 18, 24, 30, 36});


    double lr = 0.5;
    for(int iter = 0; iter < 10; ++iter) {
        
    }

    FreeTensor(raw_tensor);
    FreeTensor(gt_tensor);
    return 0;
}