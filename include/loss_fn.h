#ifndef LOSS_H_
#define LOSS_H_

#include "tensor.h"

#include <math.h>

double MSE(Tensor* ground_truth, Tensor* prediction);

#endif