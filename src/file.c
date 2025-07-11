#include "file.h"

char* Tensor2String(Tensor* tensor) {
    // FORMAT: n_dim shape data grad
    int tensor_size = sizeof(int) + sizeof(char) + (sizeof(char) + sizeof(int)) * tensor->n_dim + GetDataNum(tensor) * (sizeof(double) + sizeof(char)) * 2 + 1;
    char* tensor_str = (char*)calloc(tensor_size, sizeof(char));
    if(tensor_str == NULL) {
        perror("Tensor str malloc");
        exit(EXIT_MALLOC_FAILURE);
    }

    int offset = 0;
    char divide_char = ' ';

    offset += sprintf(tensor_str + offset, "%d", tensor->n_dim); 

    offset += sprintf(tensor_str + offset, "%c", divide_char);

    for(int i = 0; i < tensor->n_dim; i++) {
        offset += sprintf(tensor_str + offset, "%d", tensor->shape[i]); 

        offset += sprintf(tensor_str + offset, "%c", divide_char);
    }
    for(int i = 0; i < GetDataNum(tensor); i++) {
        offset += sprintf(tensor_str + offset, "%lf", tensor->data[i]); 
        offset += sprintf(tensor_str + offset, "%c", divide_char);
    }
    for(int i = 0; i < GetDataNum(tensor); i++) {
        offset += sprintf(tensor_str + offset, "%lf", tensor->grad[i]); 
        offset += sprintf(tensor_str + offset, "%c", divide_char);
    }

    return tensor_str;
}

bool SaveTensor(Tensor* tensor, const char* file_name) {
    char* tensor_str = Tensor2String(tensor);
    printf("%s", tensor_str);
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