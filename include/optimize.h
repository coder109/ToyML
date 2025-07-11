#ifndef OPTIMIZE_H_
#define OPTIMIZE_H_

#include "tensor.h"

#include <math.h>

bool SGDOptimize(Tensor* tensor, double learning_rate);

void ZeroGradTensor(Tensor* tensor);

bool CheckNAN(Tensor* tensor);

#endif