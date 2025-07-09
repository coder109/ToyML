#ifndef BACKPROP_H_
#define BACKPROP_H_

#include "tensor.h"

void Backprop(Tensor* tensor);

void SigmoidBackprop(Tensor* tensor);

#endif