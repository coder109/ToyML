#ifndef ACTIVE_FN_H_
#define ACTIVE_FN_H_

#include "tensor.h"
#include "macro_code.h"

#include <math.h>

Tensor* Sigmoid(Tensor* tensor);

Tensor* Softmax(Tensor* tensor);

#endif