# ToyML

**ToyML** is a lightweight machine learning library built for educational purposes and fun implemented in C only. 

ToyML maintains minimal dependencies by exclusively using standard C library headers. This intentional design choice keeps the implementation lightweight, easy to build and focused on core ML concepts.

ToyML draws inspiration from PyTorch's design philosophy and tensor operations. To ensure correctness, all test cases are validated against PyTorch's reference implementations.

⚠️ **Note:** This is not a production-ready library. Use it for learning and experimentation only.

## How to use?

On Windows, use the following command to build:

```bash
mkdir build
cd build
cmake .. -G “MinGW Makefiles”
make
```

On Linux:

```bash
mkdir build
cd build
cmake ..
make
```

A simple example is presented below:

```C
#include "include/tensor.h"
#include "include/active_fn.h"
#include "include/backprop.h"
#include "include/loss_fn.h"
#include "include/optimize.h"

int main(int argc, char* argv[]) {
    Tensor* in_tensor = CreateBaseOnArray(2, (int[]){6, 1}, (double[]){1, 2, 3, 4, 5, 6});

    Tensor* layer = CreateBaseOnArray(2, (int[]){1, 1}, (double[]){1});

    Tensor* gt_tensor = CreateBaseOnArray(2, (int[]){6, 1}, (double[]){6, 12, 18, 24, 30, 36});


    double lr = 0.001;
    for(int iter = 0; iter < 200; ++iter) {
        // Forward Process
        Tensor* out_tensor = MatMul(in_tensor, layer);
        Tensor* loss = MSE(gt_tensor, out_tensor);

        // Calculate the gradients
        Backprop(loss);

        // Optimize the parameters
        SGDOptimize(layer, lr);

        // Clear the previous gradients.
        ZeroGradTensor(layer);
        
        // Free
        free(loss);
        free(out_tensor);
    }
    
    PrintTensor(layer);

    FreeTensor(in_tensor);
    FreeTensor(layer);
    FreeTensor(gt_tensor);
    return 0;
}
```

Check `example` for more information.

## Introduction

### Tensor

The core data structure in ToyML is the **Tensor**, which is implemented as a flat array of `double` numbers. The interpretation of this data is determined by two key properties:

- `shape`: Describes the dimensions of the tensor (e.g., [2, 3] for a 2×3 matrix)
- `n_dim`: The number of dimensions of the tensor

For instance, the tensor `[1, 2, 3, 4, 5, 6]` can be interpreted as a 2\*3 matrix or a 2\*1\*3 tensor.

### Backpropagation

In ToyML, backpropagation refers only to the process that computes the gradients, not how the gradients are used.

## TODO

- [x] Backpropagation
- [x] Optimizer
- [ ] Built-in Networks
- [ ] Tokenizers
- [ ] Save and load weights

## Contribution

As a toy project, contributions aren't expected, but feel free to fork and experiment!

## License

GPL-3.0 license