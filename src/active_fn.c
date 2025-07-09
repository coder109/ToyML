#include "active_fn.h"

Tensor* Sigmoid(Tensor* tensor) {
    Tensor* result = CreateZeroTensor(tensor->n_dim, tensor->shape);
    int data_num = GetDataNum(tensor);

    for(int i = 0; i < data_num; i++) {
        result->data[i] = 1. / (1. + exp(-tensor->data[i]));
    }

    result->fn_id = OP_SIGMOID;
    result->n_prev = 1;
    result->prev = (Tensor**)malloc(sizeof(Tensor*));
    if(result->prev == NULL) {
        perror("Prev malloc");
        exit(EXIT_MALLOC_FAILURE);
    }
    result->prev[0] = tensor;
    return result;
}

Tensor* Softmax(Tensor* tensor) {
    Tensor* result = CreateZeroTensor(tensor->n_dim, tensor->shape);
    int data_num = GetDataNum(tensor);

    double sum = 0.0;
    for(int i = 0; i < data_num; i++) {
        sum += exp(tensor->data[i]);
    }
    for(int i = 0; i < data_num; i++) {
        result->data[i] = exp(tensor->data[i]) / sum;
    }

    result->fn_id = OP_SOFTMAX;
    result->n_prev = 1;
    result->prev = (Tensor**)malloc(sizeof(Tensor*));
    if(result->prev == NULL) {
        perror("Prev malloc");
        exit(EXIT_MALLOC_FAILURE);
    }
    result->prev[0] = tensor;
    return result;
}