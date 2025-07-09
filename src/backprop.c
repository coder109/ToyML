#include "backprop.h"

void Backprop(Tensor* tensor) {
    // BackProp based on the fn_id.
    switch(tensor->fn_id) {
        case OP_SIGMOID:
            SigmoidBackprop(tensor);
            break;
        default:
            break;
    }
    // Recursively backprop to the previous tensors.
    for(int i = 0; i < tensor->n_prev; i++) {
        Backprop(tensor->prev[i]);
    }
}

void LossBackprop(Tensor* ground_truth, Tensor* prediction) {
    switch(prediction->fn_id) {
        case OP_MSE:
            MSEBackprop(ground_truth, prediction);
            break;
        default:
            break;
    }
    for(int i = 0; i < prediction->n_prev; i++) {
        Backprop(prediction->prev[i]);
    }
}

void SigmoidBackprop(Tensor* tensor) {
    for(int i = 0; i < GetDataNum(tensor); i++) {
        tensor->prev[0]->grad[i] += tensor->grad[i] * tensor->data[i] * (1 - tensor->data[i]);
    }
}

void MatmulBackprop(Tensor* tensor) {
    
}

void MSEBackprop(Tensor* ground_truth, Tensor* prediction) {
    for(int i = 0; i < GetDataNum(prediction); i++) {
        prediction->grad[i] += 2 * (prediction->data[i] - ground_truth->data[i]) / GetDataNum(prediction);
    }
}