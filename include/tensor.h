#ifndef TENSOR_H_
#define TENSOR_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdarg.h>

#include "macro_code.h"

typedef struct Tensor {
    // Data
    double* data;
    int* shape;
    int n_dim;

    // Backprop
    double* grad;
    int n_prev;
    struct Tensor** prev;
    int fn_id;
} Tensor;

// Creation and Free
Tensor* CreateZeroTensor(int n_dim, int* shape_elem);

Tensor* CreateOneTensor(int n_dim, int* shape_elem);

Tensor* CreateBaseOnArray(int n_dim, int* shape_elem, double* data);

void FreeTensor(Tensor* tensor);

// Shape Operation
bool ChangeShape(Tensor* tensor, int n_dim, int* shape_elem);

// Basic Operation on Calculation
Tensor* AddTensor(Tensor* tensor1, Tensor* tensor2);

bool AddTensorInPlace(Tensor* tensor1, Tensor* tensor2);

Tensor* SubTensor(Tensor* tensor1, Tensor* tensor2);

bool SubTensorInPlace(Tensor* tensor1, Tensor* tensor2);

// Matrix Operation
bool isMatrix(Tensor* tensor);

Tensor* MatMul(Tensor* tensor1, Tensor* tensor2);

Tensor* Transpose(Tensor* tensor);

// Utils
int GetDataNum(Tensor* tensor);

void PrintTensor(Tensor* tensor);

bool IsSameShape(Tensor* tensor1, Tensor* tensor2);

#endif