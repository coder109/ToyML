#include "loss_fn.h"

double MSE(Tensor* ground_truth, Tensor* prediction) {
    double loss = 0.0;
    int data_num = GetDataNum(ground_truth);
    for(int i = 0; i < data_num; i++) {
        loss += (ground_truth->data[i] - prediction->data[i]) * (ground_truth->data[i] - prediction->data[i]);
    }
    return loss / data_num;
}