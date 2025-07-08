#include "tensor.h"

Tensor* CreateZeroTensor(int n_dim, int* shape_elem) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if(tensor == NULL) {
        perror("Tensor malloc");
        exit(EXIT_MALLOC_FAILURE);
    }

    int* shape = (int*)malloc(sizeof(int) * n_dim);
    if(shape == NULL) {
        perror("Shape malloc");
        exit(EXIT_MALLOC_FAILURE);
    }

    int data_num = 1;
    for(int i = 0; i < n_dim; i++) {
        shape[i] = shape_elem[i];
        data_num *= shape[i];
    }

    double* data = (double*)calloc(data_num, sizeof(double));
    if(data == NULL) {
        perror("Data malloc");
        exit(EXIT_MALLOC_FAILURE);
    }

    tensor->data = data;
    tensor->n_dim = n_dim;
    tensor->shape = shape;
    return tensor;
}

Tensor* CreateOneTensor(int n_dim, int* shape_elem) {
    Tensor* tensor = CreateZeroTensor(n_dim, shape_elem);

    int data_num = GetDataNum(tensor);
    for(int i = 0; i < data_num; i++) {
        tensor->data[i] = 1.;
    }
    return tensor;
}

Tensor* AddTensor(Tensor* tensor1, Tensor* tensor2) {
    if(!IsSameShape(tensor1, tensor2)) {
        perror("Tensor shape not same");
        exit(EXIT_OP_SHAPE_FAILURE);
    }
    Tensor* result = CreateZeroTensor(tensor1->n_dim, tensor1->shape);
    int data_num = GetDataNum(result);
    for(int i = 0; i < data_num; i++) {
        result->data[i] = tensor1->data[i] + tensor2->data[i];
    }
}

bool AddTensorInPlace(Tensor* tensor1, Tensor* tensor2) {
    if(!IsSameShape(tensor1, tensor2)) {
        perror("Tensor shape not same");
        exit(EXIT_OP_SHAPE_FAILURE);
    }
    int data_num = GetDataNum(tensor1);
    for(int i = 0; i < data_num; i++) {
        tensor1->data[i] += tensor2->data[i];
    }
}

Tensor* SubTensor(Tensor* tensor1, Tensor* tensor2) {
    if(!IsSameShape(tensor1, tensor2)) {
        perror("Tensor shape not same");
        exit(EXIT_OP_SHAPE_FAILURE);
    }
    Tensor* result = CreateZeroTensor(tensor1->n_dim, tensor1->shape);
    int data_num = GetDataNum(result);
    for(int i = 0; i < data_num; i++) {
        result->data[i] = tensor1->data[i] - tensor2->data[i];
    }
}

bool SubTensorInPlace(Tensor* tensor1, Tensor* tensor2) {
    if(!IsSameShape(tensor1, tensor2)) {
        perror("Tensor shape not same");
        exit(EXIT_OP_SHAPE_FAILURE);
    }
    int data_num = GetDataNum(tensor1);
    for(int i = 0; i < data_num; i++) {
        tensor1->data[i] -= tensor2->data[i];
    }
}

int GetDataNum(Tensor* tensor) {
    int data_num = 1;
    for(int i = 0; i < tensor->n_dim; i++) {
        data_num *= tensor->shape[i];
    }
    return data_num;
}

void PrintTensor(Tensor* tensor) {
    int data_num = 1;
    for(int i = 0; i < tensor->n_dim; i++) {
        data_num *= tensor->shape[i];
    }

    printf("Basic Information:\n ");
    printf("Dimension Num: %d\n", tensor->n_dim);
    printf("Shape: ");
    for(int i = 0; i < tensor->n_dim; i++) {
        printf("%d ", tensor->shape[i]);
    }
    printf("\nData:\n");
    for(int i = 0; i < data_num; i++) {
        printf("%lf ", tensor->data[i]);
    }
    printf("\n");
}

bool IsSameShape(Tensor* tensor1, Tensor* tensor2) {
    if(tensor1->n_dim != tensor2->n_dim) {
        return false;
    }
    for(int i = 0; i < tensor1->n_dim; i++) {
        if(tensor1->shape[i] != tensor2->shape[i]) {
            return false;
        }
    }
    return true;
}

void FreeTensor(Tensor* tensor) {
    free(tensor->data);
    free(tensor->shape);
    free(tensor);
}