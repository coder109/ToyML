#ifndef OPTIMIZE_H_
#define OPTIMIZE_H_

#include "tensor.h"

#include <math.h>

typedef struct {
    Tensor* tensor;
    double* previous_v;
} SGDMomentumState;

typedef struct {
    Tensor* tensor;
    double* m;
    double* v;
    int time_step;
} AdamState;

SGDMomentumState* CreateSGDMomentumOptimizeState(Tensor* tensor);

AdamState* CreateAdamOptimizeState(Tensor* tensor);

bool SGDOptimize(Tensor* tensor, double learning_rate);

bool SGDMomentumOptimize(SGDMomentumState* tensor, double learning_rate, double momentum_parameter);

bool AdamOptimize(AdamState* state, double learning_rate, double beta1, double beta2);

void ZeroGradTensor(Tensor* tensor);

bool CheckNAN(Tensor* tensor);

#endif