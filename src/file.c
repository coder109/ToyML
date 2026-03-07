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

    int n_dim = String2Int(strtok(buf, " "));
    int* shape = (int*)malloc(sizeof(int) * n_dim);
    if(shape == NULL) {
        perror("Shape malloc");
        free(buf);
        exit(EXIT_MALLOC_FAILURE);
    }

    for(int i = 0; i < n_dim; i++) {
        shape[i] = String2Int(strtok(NULL, " "));
    }
    Tensor* tensor = CreateZeroTensor(n_dim, shape);
    free(shape);
    int data_num = GetDataNum(tensor);
    double* data = tensor->data;
    double* grad = tensor->grad;
    for(int i = 0; i < data_num; i++) {
        data[i] = String2Double(strtok(NULL, " "));
    }
    for(int i = 0; i < data_num; i++) {
        grad[i] = String2Double(strtok(NULL, " "));
    }
    free(buf);
    return tensor;
}

int String2Int(const char* str) {
    int num = 0;
    for(int i = 0; i < strlen(str); i++) {
        num *= 10;
        num += str[i] - '0';
    }
    return num;
}

static double String2Double_NoExpo(const char* str) {
    double num = 0;
    double dot_number = 0;
    int dot_number_length = 0;
    bool dot_mode = false;
    for(int i = 0; i < strlen(str); i++) {
        if(str[i] == '.') {
            dot_mode = true;
            continue;
        }
        if(str[i] == ' ' || str[i] == '\n') {
            continue;
        }
        if(dot_mode) {
            dot_number *= 10.;
            dot_number += str[i] - '0';
            dot_number_length++;
        } else {
            num *= 10.;
            num += str[i] - '0';
        }
    }
    num += dot_number / (double)pow(10, dot_number_length);
    return num;
}

double String2Double(const char* str) {
    if(str[0] == '-') {
        return -String2Double_NoExpo(str + 1);
    } else {
        return String2Double_NoExpo(str);
    }
}