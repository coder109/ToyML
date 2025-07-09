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

void SigmoidBackprop(Tensor* tensor) {
    for(int i = 0; i < GetDataNum(tensor); i++) {
        tensor->prev[0]->grad[i] += tensor->grad[i] * tensor->data[i] * (1 - tensor->data[i]);
    }
}