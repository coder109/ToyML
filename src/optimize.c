#include "optimize.h"

SGDMomentumState* CreateSGDMomentumOptimizeState(Tensor* tensor) {
    SGDMomentumState* state = (SGDMomentumState*)malloc(sizeof(SGDMomentumState));
    if(state == NULL) {
        perror("State malloc");
        exit(EXIT_MALLOC_FAILURE);
    }
    state->tensor = tensor;
    state->previous_v = (double*)calloc(GetDataNum(tensor), sizeof(double));
    if(state->previous_v == NULL) {
        perror("Previous_v malloc");
        exit(EXIT_MALLOC_FAILURE);
    }

    return state;
}

AdamState* CreateAdamOptimizeState(Tensor* tensor) {
    AdamState* state = (AdamState*)malloc(sizeof(AdamState));
    if(state == NULL) {
        perror("State malloc");
        exit(EXIT_MALLOC_FAILURE);
    }
    state->tensor = tensor;

    state->m = (double*)calloc(GetDataNum(tensor), sizeof(double));
    if(state->m == NULL) {
        perror("M malloc");
        exit(EXIT_MALLOC_FAILURE);
    }

    state->v = (double*)calloc(GetDataNum(tensor), sizeof(double));
    if(state->v == NULL) {
        perror("V malloc");
        exit(EXIT_MALLOC_FAILURE);
    }

    state->time_step = 0;
    return state;
}

bool SGDOptimize(Tensor* tensor, double learning_rate) {
    int data_num = GetDataNum(tensor);
    for(int i = 0; i < data_num; i++) {
        tensor->data[i] -= learning_rate * tensor->grad[i];
    }

    return CheckNAN(tensor);
}

bool SGDMomentumOptimize(SGDMomentumState* state, double learning_rate, double momentum_parameter) {
    double* curr_v = (double*)calloc(GetDataNum(state->tensor), sizeof(double));
    if(curr_v == NULL) {
        perror("Curr_v malloc");
        exit(EXIT_MALLOC_FAILURE);
    }
    int data_num = GetDataNum(state->tensor);
    for(int i = 0; i < data_num; i++) {
        curr_v[i] = momentum_parameter * state->previous_v[i] - learning_rate * state->tensor->grad[i];

        state->tensor->data[i] += curr_v[i];
    }
    free(state->previous_v);
    state->previous_v = curr_v;
    return CheckNAN(state->tensor);
}


bool AdamOptimize(AdamState* state, double learning_rate, double beta1, double beta2) {
    state->time_step++;

    Tensor* new_m = CreateZeroTensor(state->tensor->n_dim, state->tensor->shape);
    Tensor* new_v = CreateZeroTensor(state->tensor->n_dim, state->tensor->shape);

    int data_num = GetDataNum(state->tensor);
    for(int i = 0; i < data_num; i++) {
        new_m->data[i] = beta1 * state->m[i] + (1 - beta1) * state->tensor->grad[i];
        new_v->data[i] = beta2 * state->v[i] + (1 - beta2) * state->tensor->grad[i] * state->tensor->grad[i];
    }

    for(int i = 0; i < data_num; i++) {
        state->m[i] = new_m->data[i];
        state->v[i] = new_v->data[i];
    }

    for(int i = 0; i < data_num; i++) {
        new_m->data[i] /= (1 - pow(beta1, state->time_step + 1));
        new_v->data[i] /= (1 - pow(beta2, state->time_step + 1));
    }

    for(int i = 0; i < data_num; i++) {
        state->tensor->data[i] -= learning_rate * new_m->data[i] / (sqrt(new_v->data[i]) + 1e-8);
    }
    return CheckNAN(state->tensor);
    
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