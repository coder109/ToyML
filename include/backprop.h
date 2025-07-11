#ifndef BACKPROP_H_
#define BACKPROP_H_

#include "tensor.h"

int Backprop(Tensor* tensor);

void LossBackprop(Tensor* ground_truth, Tensor* prediction);

void SigmoidBackprop(Tensor* tensor);

void MatmulBackprop(Tensor* tensor);

void MSEBackprop(Tensor* tensor);

void AddBackprop(Tensor* tensor);

void SubBackprop(Tensor* tensor);

void HadamardProductBackprop(Tensor* tensor);

void ReLUBackprop(Tensor* tensor);

void SoftmaxBackprop(Tensor* tensor);

#endif