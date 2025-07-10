#ifndef LOSS_H_
#define LOSS_H_

#include "tensor.h"

#include <math.h>

Tensor* MSE(Tensor* ground_truth, Tensor* prediction);

#endif