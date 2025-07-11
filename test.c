#include "include/tensor.h"
#include "include/active_fn.h"
#include "include/backprop.h"
#include "include/loss_fn.h"
#include "include/optimize.h"
#include "include/file.h"

int main(int argc, char* argv[]) {
    Tensor* random_tensor = CreateRandom(2, (int[]){2, 3});
    
    SaveTensor(random_tensor, "test.txt");
    PrintTensor(random_tensor);

    free(random_tensor);

    FILE* fp = fopen("test.txt", "r");
    if(fp == NULL) {
        perror("File open failed");
    }
    char* line = (char*)calloc(1024, sizeof(char));
    // Read without getline
    if(fgets(line, 1024, fp) == NULL) {
        perror("File read failed");
    }

    Tensor* loaded_tensor = LoadTensor(line);   
    PrintTensor(loaded_tensor);

    printf("%lf\n", String2Double("0.112233"));
    printf("%lf\n", String2Double("31.112233"));
    free(line);
    fclose(fp);
    return 0;
}