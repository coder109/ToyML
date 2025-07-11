#ifndef FILE_H
#define FILE_H

#include "tensor.h"

#include <string.h>

char* Tensor2String(Tensor* tensor);

bool SaveTensor(Tensor* tensor, const char* file_name);

#endif