# ToyML

**ToyML** is a lightweight machine learning library built for educational purposes and fun implemented in C only.

⚠️ **Note:** This is not a production-ready library. Use it for learning and experimentation only.

## Tensor

The core data structure in ToyML is the **Tensor**, which is implemented as a flat array of `double` numbers. The interpretation of this data is determined by two key properties:

- `shape`: Describes the dimensions of the tensor (e.g., [2, 3] for a 2×3 matrix)
- `n_dim`: The number of dimensions of the tensor

For instance, the tensor `[1, 2, 3, 4, 5, 6]` can be interpreted as a 2\*3 matrix or a 2\*1\*3 tensor.

## Contribution

As a toy project, contributions aren't expected, but feel free to fork and experiment!

## License

GPL-3.0 license