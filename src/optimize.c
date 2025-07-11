#include "optimize.h"

bool SGDOptimize(Tensor* tensor, double learning_rate) {
    int data_num = GetDataNum(tensor);
    for(int i = 0; i < data_num; i++) {
        tensor->data[i] -= learning_rate * tensor->grad[i];
    }

    return CheckNAN(tensor);
}

void ZeroGradTensor(Tensor* tensor) {
    int data_num = GetDataNum(tensor);
    for(int i = 0; i < data_num; i++) {
        tensor->grad[i] = 0;
    }
}

bool CheckNAN(Tensor* tensor) {
    int data_num = GetDataNum(tensor);
    for(int i = 0; i < data_num; i++) {
        if(isnan(tensor->data[i])) {
            printf("WARNING: NAN\n");
            return true;
        }
    }
    return false;
}