#ifndef TENSOR_H_
#define TENSOR_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdarg.h>

#include "error_code.h"

typedef struct {
    double* data;
    int* shape;
    int n_dim;
} Tensor;

// Creation and Free
Tensor* CreateZeroTensor(int n_dim, int* shape_elem);

Tensor* CreateOneTensor(int n_dim, int* shape_elem);

void FreeTensor(Tensor* tensor);

Tensor* AddTensor(Tensor* tensor1, Tensor* tensor2);

bool AddTensorInPlace(Tensor* tensor1, Tensor* tensor2);

Tensor* SubTensor(Tensor* tensor1, Tensor* tensor2);

bool SubTensorInPlace(Tensor* tensor1, Tensor* tensor2);

// Utils
int GetDataNum(Tensor* tensor);

void PrintTensor(Tensor* tensor);

bool IsSameShape(Tensor* tensor1, Tensor* tensor2);

#endif