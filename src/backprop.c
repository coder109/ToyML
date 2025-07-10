#include "backprop.h"

void Backprop(Tensor* tensor) {
    // BackProp based on the fn_id.
    switch(tensor->fn_id) {
        case OP_SIGMOID:
            SigmoidBackprop(tensor);
            break;
        case OP_MATMUL:
            MatmulBackprop(tensor);
            break;
        case OP_MSE:
            MSEBackprop(tensor);
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

void MatmulBackprop(Tensor* tensor) {
    // Suppose Y = AB, where Y is tensor, A is tensor->prev[0], B is tensor->prev[1].
    // Update A's grad
    for(int elem_row = 0; elem_row < tensor->prev[0]->shape[0]; elem_row++) {
        for(int elem_col = 0; elem_col < tensor->prev[0]->shape[1]; elem_col++) {
            double grad_sum = 0.;
            for(int col = 0; col < tensor->shape[1]; col++) {
                double tensor_grad = tensor->grad[GetElemIdxBasedOnRowCol(tensor, elem_row, col)];
                double curr_elem = tensor->prev[1]->data[GetElemIdxBasedOnRowCol(tensor->prev[1], elem_col, col)];
                grad_sum += tensor_grad * curr_elem;
            }
            tensor->prev[0]->grad[GetElemIdxBasedOnRowCol(tensor->prev[0], elem_row, elem_col)] += grad_sum;
        }
    }

    // Update B's grad
    for(int elem_row = 0; elem_row < tensor->prev[1]->shape[0]; elem_row++) {
        for(int elem_col = 0; elem_col < tensor->prev[1]->shape[1]; elem_col++) {
            double grad_sum = 0.;
            for(int row = 0; row < tensor->shape[0]; row++) {
                double tensor_grad = tensor->grad[GetElemIdxBasedOnRowCol(tensor, row, elem_col)];
                double curr_elem = tensor->prev[0]->data[GetElemIdxBasedOnRowCol(tensor->prev[0], row, elem_row)];
                grad_sum += tensor_grad * curr_elem;
            }
            tensor->prev[1]->grad[GetElemIdxBasedOnRowCol(tensor->prev[1], elem_row, elem_col)] += grad_sum;
        }
    }

}

void MSEBackprop(Tensor* loss) {
    for(int i = 0; i < GetDataNum(loss->prev[0]); i++) {
        double diff = loss->prev[0]->data[i] - loss->prev[1]->data[i];
        loss->prev[0]->grad[i] = diff * 2 / GetDataNum(loss->prev[0]);
    }
}