#include "file.h"

char* Tensor2String(Tensor* tensor) {
    // FORMAT: n_dim shape data grad
    int data_num = GetDataNum(tensor);
    size_t tensor_size = 0;

    tensor_size += (size_t)snprintf(NULL, 0, "%d ", tensor->n_dim);
    for(int i = 0; i < tensor->n_dim; i++) {
        tensor_size += (size_t)snprintf(NULL, 0, "%d ", tensor->shape[i]);
    }
    for(int i = 0; i < data_num; i++) {
        tensor_size += (size_t)snprintf(NULL, 0, "%.17g ", tensor->data[i]);
    }
    for(int i = 0; i < data_num; i++) {
        tensor_size += (size_t)snprintf(NULL, 0, "%.17g ", tensor->grad[i]);
    }
    tensor_size += 1;

    char* tensor_str = (char*)calloc(tensor_size, sizeof(char));
    if(tensor_str == NULL) {
        perror("Tensor str malloc");
        exit(EXIT_MALLOC_FAILURE);
    }

    size_t offset = 0;
    offset += (size_t)snprintf(tensor_str + offset, tensor_size - offset, "%d ", tensor->n_dim);
    for(int i = 0; i < tensor->n_dim; i++) {
        offset += (size_t)snprintf(tensor_str + offset, tensor_size - offset, "%d ", tensor->shape[i]);
    }
    for(int i = 0; i < data_num; i++) {
        offset += (size_t)snprintf(tensor_str + offset, tensor_size - offset, "%.17g ", tensor->data[i]);
    }
    for(int i = 0; i < data_num; i++) {
        offset += (size_t)snprintf(tensor_str + offset, tensor_size - offset, "%.17g ", tensor->grad[i]);
    }

    return tensor_str;
}

bool SaveTensor(Tensor* tensor, const char* file_name) {
    char* tensor_str = Tensor2String(tensor);
    FILE* fp = fopen(file_name, "w+");
    if(fp == NULL) {
        perror("File open failed");
        return false;
    }
   
    fwrite(tensor_str, sizeof(char), strlen(tensor_str), fp);
    fclose(fp);
    free(tensor_str);
    return true;
}

Tensor* LoadTensor(const char* line) {
    /* strtok modifies the buffer; copy to mutable memory so const input is not violated */
    size_t len = strlen(line) + 1;
    char* buf = (char*)malloc(len);
    if(buf == NULL) {
        perror("LoadTensor buffer malloc");
        exit(EXIT_MALLOC_FAILURE);
    }
    strcpy(buf, line);

    char* token = strtok(buf, " ");
    if(token == NULL) {
        free(buf);
        return NULL;
    }
    int n_dim = String2Int(token);
    int* shape = (int*)malloc(sizeof(int) * n_dim);
    if(shape == NULL) {
        perror("Shape malloc");
        free(buf);
        exit(EXIT_MALLOC_FAILURE);
    }

    for(int i = 0; i < n_dim; i++) {
        token = strtok(NULL, " ");
        if(token == NULL) {
            free(shape);
            free(buf);
            return NULL;
        }
        shape[i] = String2Int(token);
    }
    Tensor* tensor = CreateZeroTensor(n_dim, shape);
    free(shape);
    int data_num = GetDataNum(tensor);
    double* data = tensor->data;
    double* grad = tensor->grad;
    for(int i = 0; i < data_num; i++) {
        token = strtok(NULL, " ");
        if(token == NULL) {
            FreeTensor(tensor);
            free(buf);
            return NULL;
        }
        data[i] = String2Double(token);
    }
    for(int i = 0; i < data_num; i++) {
        token = strtok(NULL, " ");
        if(token == NULL) {
            FreeTensor(tensor);
            free(buf);
            return NULL;
        }
        grad[i] = String2Double(token);
    }
    free(buf);
    return tensor;
}

int String2Int(const char* str) {
    int sign = 1;
    int num = 0;
    int start_idx = 0;

    if(str == NULL || str[0] == '\0') {
        return 0;
    }
    if(str[0] == '-') {
        sign = -1;
        start_idx = 1;
    }

    for(int i = start_idx; str[i] != '\0'; i++) {
        if(str[i] < '0' || str[i] > '9') {
            break;
        }
        num = num * 10 + (str[i] - '0');
    }
    return sign * num;
}

double String2Double(const char* str) {
    if(str == NULL) {
        return 0.0;
    }
    return strtod(str, NULL);
}