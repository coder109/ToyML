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

    double* grad = (double*)calloc(data_num, sizeof(double));
    if(grad == NULL) {
        perror("Grad malloc");
        exit(EXIT_MALLOC_FAILURE);
    }

    tensor->data = data;
    tensor->grad = grad;
    tensor->n_dim = n_dim;
    tensor->shape = shape;
    tensor->fn_id = OP_UNDEFINED;
    tensor->n_prev = 0;
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

Tensor* CreateBaseOnArray(int n_dim, int* shape_elem, double* data) {
    Tensor* tensor = CreateZeroTensor(n_dim, shape_elem);
    int data_num = GetDataNum(tensor);
    for(int i = 0; i < data_num; i++) {
        tensor->data[i] = data[i];
    }
    return tensor;
}

Tensor* CreateRandom(int n_dim, int* shape_elem) {
    Tensor* tensor = CreateZeroTensor(n_dim, shape_elem);
    int data_num = GetDataNum(tensor);
    for(int i = 0; i < data_num; i++) {
        tensor->data[i] = (double)rand() / (double)RAND_MAX;
    }
    return tensor;
}

bool ChangeShape(Tensor* tensor, int n_dim, int* shape_elem) {
    int new_data_num = 1;
    for(int i = 0; i < n_dim; i++) {
        new_data_num *= shape_elem[i];
    }
    if(new_data_num != GetDataNum(tensor)) {
        printf("Unable to change shape\n");
        return false;
    }
    tensor->n_dim = n_dim;
    free(tensor->shape);
    tensor->shape = (int*)malloc(sizeof(int) * n_dim);
    if(tensor->shape == NULL) {
        perror("Shape malloc");
        exit(EXIT_MALLOC_FAILURE);
    }

    for(int i = 0; i < n_dim; i++) {
        tensor->shape[i] = shape_elem[i];
    }
    return true;
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

    result->n_prev = 2;
    result->prev = (Tensor**)malloc(sizeof(Tensor*) * 2);
    if(result->prev == NULL) {
        perror("Prev malloc");
        exit(EXIT_MALLOC_FAILURE);
    }
    result->prev[0] = tensor1;
    result->prev[1] = tensor2;
    result->fn_id = OP_ADD;
    return result;
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
    return true;
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

    result->n_prev = 2;
    result->prev = (Tensor**)malloc(sizeof(Tensor*) * 2);
    if(result->prev == NULL) {
        perror("Prev malloc");
        exit(EXIT_MALLOC_FAILURE);
    }
    result->prev[0] = tensor1;
    result->prev[1] = tensor2;
    result->fn_id = OP_SUB;
    return result;
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
    return true;
}

Tensor* HadamardProduct(Tensor* tensor1, Tensor* tensor2) {
    if(!IsSameShape(tensor1, tensor2)) {
        perror("Tensor shape not same");
        exit(EXIT_OP_SHAPE_FAILURE);
    }
    Tensor* result = CreateZeroTensor(tensor1->n_dim, tensor1->shape);
    int data_num = GetDataNum(result);
    for(int i = 0; i < data_num; i++) {
        result->data[i] = tensor1->data[i] * tensor2->data[i];
    }

    result->n_prev = 2;
    result->prev = (Tensor**)malloc(sizeof(Tensor*) * 2);
    if(result->prev == NULL) {
        perror("Prev malloc");
        exit(EXIT_MALLOC_FAILURE);
    }
    result->prev[0] = tensor1;
    result->prev[1] = tensor2;
    result->fn_id = OP_HADAMARD;
    return result;
}

bool isMatrix(Tensor* tensor) {
    return tensor->n_dim == 2;
}

Tensor* MatMul(Tensor* tensor1, Tensor* tensor2) {
    if(!isMatrix(tensor1) || !isMatrix(tensor2)) {
        perror("Tensor is not matrix");
        exit(EXIT_NOT_MAT_FAILURE);
    }

    if(tensor1->shape[1] != tensor2->shape[0]) {
        perror("Matrix shape not match");
        exit(EXIT_OP_SHAPE_FAILURE);
    }

    Tensor* result = CreateZeroTensor(2, (int[]){tensor1->shape[0], tensor2->shape[1]});

    for(int i = 0; i < tensor1->shape[0]; i++) {
        for(int j = 0; j < tensor2->shape[1]; j++) {
            for(int k = 0; k < tensor1->shape[1]; k++) {
                result->data[i * result->shape[1] + j] += tensor1->data[i * tensor1->shape[1] + k] * tensor2->data[k * tensor2->shape[1] + j];
            }
        }
    }

    result->n_prev = 2;
    result->prev = (Tensor**)malloc(sizeof(Tensor*) * 2);
    if(result->prev == NULL) {
        perror("Prev malloc");
        exit(EXIT_MALLOC_FAILURE);
    }
    result->prev[0] = tensor1;
    result->prev[1] = tensor2;
    result->fn_id = OP_MATMUL;
    return result;
}

Tensor* Transpose(Tensor* tensor) {
    if(!isMatrix(tensor)) {
        perror("Tensor is not matrix");
        exit(EXIT_NOT_MAT_FAILURE);
    }
    Tensor* result = CreateZeroTensor(2, (int[]){tensor->shape[1], tensor->shape[0]});

    for(int i = 0; i < tensor->shape[0]; i++) {
        for(int j = 0; j < tensor->shape[1]; j++) {
            result->data[j * result->shape[1] + i] = tensor->data[i * tensor->shape[1] + j];
        }
    }
    return result;
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
    printf("\nGrad:\n");
    for(int i = 0; i < data_num; i++) {
        printf("%lf ", tensor->grad[i]);
    }
    printf("\nPrev Node Num:%d", tensor->n_prev);
    printf("\nOP ID:%d\n", tensor->fn_id);
}

int GetElemIdxBasedOnRowCol(Tensor* tensor, int row, int col) {
    return row * tensor->shape[1] + col;    
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
    if(tensor->data != NULL) {
        free(tensor->data);
    }
    if(tensor->shape != NULL) {
        free(tensor->shape);
    }
    if(tensor->grad != NULL) {
        free(tensor->grad);
    }
    if(tensor->prev != NULL) {
        free(tensor->prev);
    }
    free(tensor);
}