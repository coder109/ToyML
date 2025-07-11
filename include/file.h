#ifndef FILE_H
#define FILE_H

#include "tensor.h"
#include "macro_code.h"

#include <string.h>
#include <math.h>

char* Tensor2String(Tensor* tensor);

bool SaveTensor(Tensor* tensor, const char* file_name);

Tensor* LoadTensor(const char* line);

int String2Int(const char* str);

double String2Double(const char* str);

#endif