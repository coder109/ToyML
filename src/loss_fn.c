#include "loss_fn.h"

Tensor* MSE(Tensor* ground_truth, Tensor* prediction) {
    Tensor* loss = CreateZeroTensor(prediction->n_dim, prediction->shape);
    int data_num = GetDataNum(prediction);

    for(int i = 0; i < data_num; i++) {
        loss->data[i] = (prediction->data[i] - ground_truth->data[i]) * (prediction->data[i] - ground_truth->data[i]);
    }

    loss->n_prev = 2;
    loss->prev = (Tensor**)malloc(sizeof(Tensor*) * 2);
    if(loss->prev == NULL) {
        perror("Prev malloc");
        exit(EXIT_MALLOC_FAILURE);
    }
    loss->prev[0] = prediction;
    loss->prev[1] = ground_truth;
    loss->fn_id = OP_MSE;
    return loss;
}