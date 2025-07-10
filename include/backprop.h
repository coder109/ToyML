#ifndef BACKPROP_H_
#define BACKPROP_H_

#include "tensor.h"

void Backprop(Tensor* tensor);

void LossBackprop(Tensor* ground_truth, Tensor* prediction);

void SigmoidBackprop(Tensor* tensor);

void MatmulBackprop(Tensor* tensor);

void MSEBackprop(Tensor* tensor);

#endif