# Project `cure_test` Codebase Report

This report provides a summary of the functionality of each file in the `cure_test` project codebase.

## Root Directory

### `README.md`

Provides an overview of the `cure_test` project, its focus on implementing functionalities from the CURE paper (Homomorphic Encryption for Deep Neural Networks), current status, structure, and next steps. It mentions the use of the Lattigo library for homomorphic encryption operations.

### `go.mod`

The Go module file for the `cure_test` project. It defines the module path (`cure_test`) and lists its dependencies. Key dependencies include:
- `github.com/tuneinsight/lattigo/v6 v6.1.1`

### `go.sum`

The Go checksum file. It contains the expected cryptographic checksums of the content of specific module versions (and their dependencies). This file is used to ensure that future downloads of these modules retrieve the same content as the first download, ensuring build reproducibility and integrity.

## `data/`

This directory contains dataset files used for training and testing models.

Currently, it holds the MNIST dataset, which consists of handwritten digit images. The files are:
- `train-images-idx3-ubyte.gz`: Training set images (60,000 images).
- `train-labels-idx1-ubyte.gz`: Training set labels.
- `t10k-images-idx3-ubyte.gz`: Test set images (10,000 images).
- `t10k-labels-idx1-ubyte.gz`: Test set labels.

These files are in the IDX file format, compressed with gzip.

## `examples/`

### `examples/activation_demo/main.go`

This file (originally `examples/activation_demo.go` and moved to its own directory to resolve build issues) demonstrates the practical application of homomorphic activation functions (ReLU, Sigmoid, Tanh) using Homomorphic Encryption (HE) with the Lattigo library. 

Key functionalities include:
- Setting up HE parameters (CKKS scheme).
- Generating public and secret keys.
- Creating an encoder, encryptor, decryptor, and evaluator.
- Defining sample input data (a slice of `float64`).
- Encrypting the input data.
- Defining and applying homomorphic versions of ReLU, Sigmoid, and Tanh activation functions (which use polynomial approximations) to the encrypted data.
- Decrypting the results.
- Calculating and printing the approximation error between the homomorphically computed results and the expected plaintext results.

This example serves as a test case and a usage illustration for the homomorphic activation functions implemented in the `pkg/activation_he` package.

## `pkg/activation_he/`

### `pkg/activation_he/README.md`

This README file details the implementation of homomorphic activation functions (ReLU, Sigmoid, Tanh) within the `activation_he` package. It explains that since these functions are non-linear, they cannot be directly computed on encrypted data using typical HE schemes. The approach taken is to use polynomial approximations (specifically minimax polynomials) of these functions, which can then be evaluated homomorphically.

The document likely covers:
- The motivation for using polynomial approximations.
- The choice of minimax approximation for minimizing the maximum error over a given interval.
- How to use the provided functions to evaluate these approximated activation functions on CKKS ciphertexts.

### `pkg/activation_he/activations.go`

This file contains the core logic for evaluating homomorphic activation functions. It defines a `HomomorphicActivationFunction` struct which likely holds the polynomial approximation (coefficients, degree, interval) for a specific activation function.

Key components:
- `HomomorphicActivationFunction` struct: Stores polynomial coefficients, degree, and the approximation interval.
- `NewHomomorphicActivationFunction`: Constructor to create instances of `HomomorphicActivationFunction` (e.g., for ReLU, Sigmoid, Tanh using their pre-computed minimax polynomial coefficients from `minimax.go`).
- `Evaluate(ciphertext *rlwe.Ciphertext, evaluator *ckks.Evaluator) (*rlwe.Ciphertext, error)`: This method takes an encrypted input vector (ciphertext) and applies the approximated activation function to it homomorphically. It uses the Lattigo evaluator to perform polynomial evaluation (scalar multiplications, ciphertext additions, and potentially multiplications followed by relinearization and rescaling).

### `pkg/activation_he/activations_test.go`

This file contains unit tests for the homomorphic activation functions defined in `activations.go`. The tests aim to verify the correctness of the homomorphic evaluation of ReLU, Sigmoid, and Tanh.

A typical test flow would involve:
- Setting up a test context with CKKS parameters, keys, and HE components (encoder, encryptor, decryptor, evaluator) using `setupTestContext` from `test_helpers.go`.
- Defining sample input data (plaintext vectors).
- Encrypting the input data.
- Creating a `HomomorphicActivationFunction` for the function under test (e.g., ReLU).
- Calling the `Evaluate` method on the encrypted input.
- Decrypting the resulting ciphertext.
- Comparing the decrypted output with the expected plaintext output (i.e., applying the standard activation function to the original plaintext input), allowing for a small approximation error using `checkApproximate` from `test_helpers.go`.

### `pkg/activation_he/minimax.go`

This file implements the generation of minimax polynomial approximations for various activation functions. Minimax polynomials are chosen because they minimize the maximum error (L-infinity norm) over a specified interval `[a, b]` for a given degree.

Key functions:
- `MinimaxApproximationReLU(degree int, intervalMin, intervalMax float64) ([]float64, error)`: Computes the coefficients of the minimax polynomial for the ReLU function.
- `MinimaxApproximationSigmoid(degree int, intervalMin, intervalMax float64) ([]float64, error)`: Computes the coefficients for the Sigmoid function.
- `MinimaxApproximationTanh(degree int, intervalMin, intervalMax float64) ([]float64, error)`: Computes the coefficients for the Tanh function.
- Internally, these functions likely use numerical methods (e.g., Remez algorithm or a similar method based on Chebyshev polynomials and solving a linear system) to find the optimal polynomial coefficients.
- Helper functions like `ChebyshevNodes` (to compute points for interpolation/approximation) and `vandermonde` (to construct Vandermonde matrices for solving linear systems) are likely present.

### `pkg/activation_he/test_helpers.go`

This file provides utility functions and structures to support testing within the `activation_he` package, particularly for `activations_test.go`.

Key components:
- `TestContext` struct: A container to hold common Homomorphic Encryption (HE) components needed for tests, such as `ckks.Parameters`, `rlwe.SecretKey`, `rlwe.PublicKey`, `*ckks.Encoder`, `*rlwe.Encryptor`, `*rlwe.Decryptor`, and `*ckks.Evaluator`.
- `setupTestContext(paramsID he.ParameterSetIdentifier) (*TestContext, error)`: A helper function that initializes and returns a `TestContext`. It typically involves:
    - Getting CKKS parameters using `he.GetCKKSParameters`.
    - Generating a new key pair (secret and public keys).
    - Creating an evaluation key set (e.g., relinearization keys, rotation keys if needed).
    - Instantiating the encoder, encryptor, decryptor, and evaluator with these parameters and keys.
- `checkApproximate(t *testing.T, expected, actual []float64, epsilon float64)`: A utility function to compare two slices of `float64` values. It asserts that each corresponding pair of elements is close enough, i.e., their absolute difference is within a specified `epsilon`. This is crucial for testing approximate HE schemes like CKKS.

### `pkg/activation_he/utils.go`

This file provides various utility functions to support the polynomial approximation of activation functions within the `activation_he` package. These functions are primarily used for analyzing the quality of the approximations.

Key functions:
- `ComputeApproximationError(f func(float64) float64, p func(float64) float64, x float64) float64`: Calculates the absolute error `|f(x) - p(x)|` between an original function `f(x)` and its polynomial approximation `p(x)` at a specific point `x`.
- `ComputeMaxApproximationError(f func(float64) float64, p func(float64) float64, intervalMin, intervalMax float64, numSamples int) (maxError float64, xAtMaxError float64)`: Determines the maximum absolute approximation error over a specified interval `[intervalMin, intervalMax]` by sampling `numSamples` points within that interval. It returns both the maximum error and the point `x` at which it occurs.
- `GenerateErrorSamples(f func(float64) float64, p func(float64) float64, intervalMin, intervalMax float64, numSamples int) ([]float64, []float64)`: Produces two slices: one of `x` values and one of corresponding approximation errors `|f(x) - p(x)|` over an interval. This is useful for visualizing the error distribution.
- `GenerateFunctionSamples(f func(float64) float64, p func(float64) float64, intervalMin, intervalMax float64, numSamples int) (xValues []float64, fValues []float64, pValues []float64)`: Generates three slices: `x` values, corresponding `f(x)` values (original function), and `p(x)` values (polynomial approximation). This allows for direct comparison and plotting of the function and its approximation.
- `PrintPolynomialInfo(polyName string, coeffs []float64, degree int, intervalMin, intervalMax float64, maxError float64)`: A utility to print detailed information about a polynomial approximation, including its name (e.g., "ReLU"), degree, approximation interval, maximum error, and a list of its coefficients.
- `FormatPolynomialString(coeffs []float64) string`: Converts a slice of polynomial coefficients into a human-readable string representation (e.g., "0.50 + 0.25x - 0.01x^3").

## `pkg/he/`

This package is central to the homomorphic encryption capabilities of the project.

### `pkg/he/he.go`

This file acts as a high-level facade or an aggregator for the homomorphic encryption functionalities provided by its sub-packages (`pkg/he/params` and `pkg/he/ops`). Its main purpose is to offer a simplified and unified API for other parts of the application that need to use HE.

Key characteristics:
- **Re-exports Types and Constants:** It re-exports types like `ParameterSetIdentifier` and constants such as `DefaultSet`, `Set1`, `Set2`, and `TestSet` directly from the `pkg/he/params` package. This allows users of `pkg/he` to access these without needing to import `pkg/he/params` directly.
- **Wraps Parameter and Component Creation:** It provides wrapper functions for `GetCKKSParameters` and for creating essential HE components like `KeyGenerator`, `NewEncoder`, `NewEncryptor`, `NewDecryptor`, and `NewEvaluator`. These wrappers simply delegate the calls to the corresponding functions in `pkg/he/params`.
- **Re-exports HE Operations:** Similarly, it re-exports core HE operations like `ScalarMultCiphertext`, `MulCiphertexts`, `MulMatricesCiphertexts`, and `MulMatricesCiphertextsParallel` by calling the actual implementations located in the `pkg/he/ops` package.

In essence, `pkg/he/he.go` centralizes access to the HE framework, making it easier to manage imports and use HE primitives throughout the project by providing a single point of entry.

### `pkg/he/params/params.go`

This file is responsible for defining and managing Homomorphic Encryption (HE) parameter sets for the CKKS scheme and providing utility functions for creating standard HE components (encoder, encryptor, etc.). This appears to be the current and actively used version for parameter management.

Key components:
- `ParameterSetIdentifier` type: An enum-like string type (`DefaultSet`, `Set1`, `Set2`, `TestSet`) to distinguish between different pre-defined CKKS parameter configurations.
    - `DefaultSet`: A general-purpose parameter set (e.g., `LogN: 14`).
    - `Set1`, `Set2`: Intended to correspond to specific parameter sets from the CURE paper. Currently, these are placeholders and redirect to `DefaultSet`, with `TODO` comments indicating they need to be updated with the actual parameters from the paper.
    - `TestSet`: A parameter set with smaller values (e.g., `LogN: 12`) designed for faster execution during testing.
- `GetCKKSParameters(paramSetID ParameterSetIdentifier) (ckks.Parameters, error)`: A function that returns a Lattigo `ckks.Parameters` struct initialized according to the provided `paramSetID`. It handles the creation of parameters using `ckks.NewParametersFromLiteral`.
- Utility Functions: Provides simple wrappers around Lattigo's constructors for creating essential HE components:
    - `KeyGenerator(params ckks.Parameters) *rlwe.KeyGenerator`
    - `NewEncoder(params ckks.Parameters) *ckks.Encoder`
    - `NewEncryptor(params ckks.Parameters, pk *rlwe.PublicKey) *rlwe.Encryptor`
    - `NewDecryptor(params ckks.Parameters, sk *rlwe.SecretKey) *rlwe.Decryptor`
    - `NewEvaluator(params ckks.Parameters, evk rlwe.EvaluationKeySet) *ckks.Evaluator`

### `pkg/he/ops/ops.go`

This file contains the implementations of fundamental homomorphic encryption operations using the Lattigo library. This is likely the current and active set of HE operations used by the project.

Key functions:
- `ScalarMultCiphertext(scalar float64, ctIn *rlwe.Ciphertext, evaluator *ckks.Evaluator) (*rlwe.Ciphertext, error)`: Performs the multiplication of an encrypted vector (ciphertext `ctIn`) by a plaintext scalar value (`scalar`). It uses `evaluator.MulNew` (from Lattigo v6) for this operation. Includes error handling for nil inputs.
- `MulCiphertexts(ct1, ct2 *rlwe.Ciphertext, evaluator *ckks.Evaluator) (*rlwe.Ciphertext, error)`: Implements element-wise multiplication of two ciphertexts (`ct1` and `ct2`). The process involves:
    1. Homomorphic multiplication: `evaluator.MulNew(ct1, ct2)` to get a degree-2 ciphertext.
    2. Relinearization: `evaluator.Relinearize(ctOut, ctOut)` to reduce the ciphertext degree back to 1 (requires a relinearization key in the evaluator).
    3. Rescaling: `evaluator.Rescale(ctOut, ctOut)` to manage noise growth and the scale of the ciphertext.
    Includes error handling.
- `MulMatricesCiphertexts(ctaRows []*rlwe.Ciphertext, ctbCols []*rlwe.Ciphertext, k_dimension int, evaluator *ckks.Evaluator) ([][]*rlwe.Ciphertext, error)`: Provides functionality for homomorphic matrix multiplication. Matrix A is represented by its encrypted rows (`ctaRows`), and matrix B by its encrypted columns (`ctbCols`). The core logic computes each element `C_ij` of the resulting matrix C by performing a homomorphic dot product of row `i` from A and column `j` from B. This involves:
    1. Element-wise ciphertext multiplication of the row and column vectors.
    2. Relinearization and Rescaling of the product.
    3. An efficient summation of the `k_dimension` elements in the resulting product vector. This summation is often achieved using rotations in a divide-and-conquer (butterfly network) fashion to sum slots within the ciphertext. Requires appropriate rotation keys in the evaluator.
    Extensive error checking for input validity is included.
- `MulMatricesCiphertextsParallel(ctaRows []*rlwe.Ciphertext, ctbCols []*rlwe.Ciphertext, k_dimension int, evaluator *ckks.Evaluator, numWorkers int) ([][]*rlwe.Ciphertext, error)`: Offers a parallelized version of `MulMatricesCiphertexts`. It utilizes multiple Go worker goroutines to compute parts (e.g., rows or individual elements) of the resulting matrix concurrently. This aims to significantly speed up matrix multiplication for larger matrices. Comments in the code suggest observed speedups based on matrix size and the number of workers.

## `pkg/he/backup/`

This directory contains older or alternative implementations related to homomorphic encryption, likely kept for reference or during a refactoring process.

### `pkg/he/backup/params.go`

This file appears to be an older or backup version for managing Homomorphic Encryption (HE) parameters, similar in purpose to `pkg/he/params/params.go`.

Key features:
- `ParameterSetIdentifier` type: Defines `DefaultSet`, `Set1`, and `Set2` to distinguish CKKS parameter configurations.
    - `DefaultSet`: Uses `LogN: 12` (smaller, for faster testing).
    - `Set1`, `Set2`: Intended for parameters from the CURE paper, but contain placeholder values (e.g., `LogN: 14` for `Set1`, `LogN: 15` for `Set2`) with `TODO` comments to replace them with actual values from the CURE paper.
- `GetCKKSParameters(paramSetID ParameterSetIdentifier) (params ckks.Parameters, err error)`: Function to retrieve Lattigo CKKS parameters based on the identifier.
- Utility functions: `KeyGenerator`, `NewEncoder`, `NewEncryptor`, `NewDecryptor`, `NewEvaluator`. These are wrappers around Lattigo's `ckks` and `rlwe` constructors to easily create the necessary HE components.

This file seems to have been superseded by `pkg/he/params/params.go`, which has a slightly different `DefaultSet` and an additional `TestSet`.

### `pkg/he/backup/conv.go`

This file, located in a `backup` subdirectory, provides an implementation of a homomorphic convolutional layer. It appears to be an earlier or alternative version compared to any `conv.go` that might exist in a main `pkg/layers` directory.

Key components:
- `ConvLayer` struct: Defines the structure of a convolutional layer, storing parameters such as kernel dimensions (`KernelHeight`, `KernelWidth`), input/output channels (`InputChannels`, `OutputChannels`), stride (`StrideHeight`, `StrideWidth`), padding (`PaddingHeight`, `PaddingWidth`), and the layer's weights (`Weights`) and biases (`Biases`).
- `NewConvLayer(...) (*ConvLayer, error)`: A constructor function that initializes a `ConvLayer` struct. It includes validation to ensure that the dimensions of the provided weights and biases match the layer's configuration.
- `CalculateOutputDimensions(inputHeight, inputWidth int) (outputHeight, outputWidth int)`: A helper method to compute the height and width of the output feature map based on input dimensions, kernel size, stride, and padding, using the standard formula for convolutions.
- `ForwardPlaintext(input [][][]float64) ([][][]float64, error)`: Implements the forward pass of the convolution operation on unencrypted (plaintext) data. This serves as a reference implementation and for verification purposes. It applies kernels to the input, adds biases, and handles padding as specified.
- `ForwardHomomorphic(...) ([]*rlwe.Ciphertext, error)`: Implements the convolutional forward pass on homomorphically encrypted data. Input data is expected as encrypted, flattened 2D representations per channel. The function performs convolutions using HE operations (multiplications, additions, and likely rotations for summing/packing data) and adds encrypted biases.
- `ForwardHomomorphicOptimized(...)`: Suggests an optimized version of the homomorphic forward pass, potentially employing more advanced techniques for efficiency, such as better data packing (vectorization) or more optimized HE operation sequences. The implementation details for this optimized version are not fully shown in the provided snippet but would aim to improve performance over the basic homomorphic forward pass.

### `pkg/he/backup/conv_test.go`

This file, also in the `backup` directory, contains tests for the homomorphic convolution functionalities defined in the corresponding `pkg/he/backup/conv.go`.

Key tests:
- `TestConvLayerPlaintext(t *testing.T)`: Verifies the correctness of the `ForwardPlaintext` method of the `ConvLayer`. It typically sets up a `ConvLayer` with predefined weights and biases, provides a sample plaintext input, computes the output using `ForwardPlaintext`, and compares this output against manually calculated or known expected values.
- `TestConvLayerHomomorphic(t *testing.T)`: Designed to test the `ForwardHomomorphic` method. The setup is more involved:
    - Initializes CKKS parameters (e.g., using `he.GetCKKSParameters` with a `TestSet`).
    - Generates cryptographic keys (secret key, public key, relinearization keys, rotation keys).
    - Creates HE components (encoder, encryptor, decryptor, evaluator).
    - Prepares a `ConvLayer` and sample plaintext input.
    - Encrypts the input data channel by channel.
    - Calls `ForwardHomomorphic` with the encrypted inputs and HE components.
    - Decrypts the resulting encrypted output channels.
    - Compares the decrypted results against the output of `ForwardPlaintext` (or a directly computed expected plaintext result), allowing for small discrepancies due to the approximate nature of CKKS encryption using a helper like `checkApproximate`.
    - The test notes that it might be skipped in "short mode" (`if testing.Short()`) due to its potentially longer execution time.

### `pkg/he/backup/ops.go`

This file, found in the `pkg/he/backup/` subdirectory, likely contains earlier or alternative implementations of various homomorphic encryption operations, similar to `pkg/he/ops/ops.go` but potentially differing in implementation details or Lattigo version compatibility.

Key functions:
- `ScalarMultCiphertext(scalar float64, ctIn *rlwe.Ciphertext, evaluator *ckks.Evaluator) (*rlwe.Ciphertext, error)`: Implements the multiplication of an encrypted vector (ciphertext) by a plaintext scalar value (`v * Enc(W)`). The snippet shows it utilizing `evaluator.MulNew(ctIn, scalar)`, which is consistent with Lattigo v6 for scalar multiplication.
- `MulCiphertexts(ct1, ct2 *rlwe.Ciphertext, evaluator *ckks.Evaluator) (*rlwe.Ciphertext, error)`: Performs element-wise multiplication of two encrypted vectors (`Enc(W1) * Enc(W2)`). This operation typically involves a homomorphic multiplication (`evaluator.MulNew`), followed by relinearization (`evaluator.Relinearize`) to reduce the ciphertext degree, and rescaling (`evaluator.Rescale`) to manage noise and scale.
- `MulMatricesCiphertexts(...) ([][]*rlwe.Ciphertext, error)`: Implements homomorphic matrix multiplication. It takes encrypted rows of matrix A and encrypted columns of matrix B. The core idea is to compute each element `C_ij` of the resulting matrix C by homomorphically calculating the dot product of row `i` from A and column `j` from B. This involves element-wise ciphertext multiplication, relinearization, rescaling, and then a sum of the elements in the resulting product vector (often achieved efficiently using rotations).
- `MulMatricesCiphertextsParallel(...) ([][]*rlwe.Ciphertext, error)`: A parallelized version of `MulMatricesCiphertexts`. It uses multiple Go worker goroutines (managed via `sync.WaitGroup` and channels) to compute parts of the resulting matrix concurrently, aiming to speed up matrix multiplication for larger matrices. The comments suggest significant speedups based on matrix size and the number of workers.

### `pkg/he/backup/ops_test.go`

This test file, corresponding to `pkg/he/backup/ops.go` and also in the `backup` directory, provides unit tests for the HE operations defined therein.

Key components and tests:
- `checkCloseEnough(t *testing.T, expected, actual []float64, epsilon float64)`: A helper function to compare two slices of `float64` values, asserting that they are element-wise close within a specified `epsilon`. This is crucial for CKKS scheme tests due to its approximate nature.
- `TestScalarMultCiphertext(t *testing.T)`: Tests the homomorphic scalar multiplication. It encrypts a vector, multiplies it by a scalar homomorphically, decrypts the result, and verifies it against the expected plaintext computation using `checkCloseEnough`.
- `TestMulCiphertexts(t *testing.T)`: Tests the homomorphic element-wise multiplication of two ciphertexts. It encrypts two vectors, multiplies them homomorphically, decrypts, and compares with the plaintext product.
- The file also includes tests (or stubs for them) like `TestMulMatricesCiphertexts`, `TestMatrixPowerCiphertexts`, `TestParallelMatrixMultiplication`, `TestEfficientMatrixPowerCiphertexts`, and `TestMulLargeMatricesCiphertexts`. These tests generally follow a pattern: prepare plaintext data, set up HE context (parameters, keys, evaluator), encrypt data, perform the HE operation, decrypt the result, and compare against expected plaintext results, allowing for approximation errors.

## `pkg/layers/`

### `pkg/layers/conv.go`

This file implements a homomorphic convolutional layer for neural networks, designed to work with the CKKS scheme from the Lattigo library. This appears to be the current and active implementation for convolutional layers.

Key components:
- `ConvLayer` struct: Stores the parameters and data for a convolutional layer, including:
    - Kernel dimensions (`KernelHeight`, `KernelWidth`).
    - Input and output channels (`InputChannels`, `OutputChannels`).
    - Stride (`StrideHeight`, `StrideWidth`) and padding (`PaddingHeight`, `PaddingWidth`).
    - Weights (`Weights` as a 4D slice: `[outputChannels][inputChannels][kernelHeight][kernelWidth]`).
    - Biases (`Biases` as a 1D slice: `[outputChannels]`).
- `NewConvLayer(...) (*ConvLayer, error)`: A constructor function that initializes a `ConvLayer`. It performs validation to ensure the dimensions of the provided `weights` and `biases` match the specified layer configuration (e.g., `len(weights)` must equal `outputChannels`).
- `CalculateOutputDimensions(inputHeight, inputWidth int) (outputHeight, outputWidth int)`: A helper method that computes the height and width of the output feature map based on the input dimensions, kernel size, stride, and padding, using the standard formula: `outputDim = floor((inputDim + 2*padding - kernelSize)/stride + 1)`.
- `ForwardPlaintext(input [][][]float64) ([][][]float64, error)`: Implements the standard forward pass of a convolutional layer on unencrypted (plaintext) data. The input is expected in the shape `[inputChannels][inputHeight][inputWidth]`, and the output is produced in `[outputChannels][outputHeight][outputWidth]`. This serves as a reference implementation and for verification of the homomorphic version.
- `ForwardHomomorphic(inputCiphertexts []*rlwe.Ciphertext, inputHeight, inputWidth int, evaluator *ckks.Evaluator, encryptor *rlwe.Encryptor, params ckks.Parameters) ([]*rlwe.Ciphertext, error)`: Implements the convolutional forward pass on homomorphically encrypted data. 
    - Input ciphertexts are provided as a slice, one ciphertext per input channel, where each ciphertext encrypts a flattened 2D input feature map (`inputHeight * inputWidth`).
    - The output is also a slice of ciphertexts, one per output channel, containing the flattened encrypted output feature maps.
    - The process involves:
        1. Initializing output ciphertexts with encrypted bias values for each output channel.
        2. Iterating through each output channel and then each input channel.
        3. For each input-output channel pair, the corresponding kernel weights are encoded into a plaintext.
        4. The core convolution operation for each output feature map position is performed. This typically involves multiplying the relevant input patch (which might be extracted or constructed using rotations and masking of the input ciphertext) with the kernel plaintext, followed by summation of these products. This process is repeated and accumulated across all input channels for a given output channel.
        5. The Lattigo evaluator is used for HE operations like multiplication by plaintext (`Mul`), addition (`Add`), and potentially rotations (`Rotate`) for aligning data for convolution.
- `ForwardHomomorphicOptimized(...)`: This function signature suggests an optimized version of the homomorphic forward pass, likely employing more advanced techniques such as improved data packing (vectorization), number-theoretic transforms (NTT) for polynomial multiplication if applicable at a lower level, or more efficient sequences of HE operations to enhance performance and reduce noise accumulation or computation time. The implementation details are not fully visible in the snippet but would build upon the principles of `ForwardHomomorphic`.

**Note:** Files `pkg/layers/conv_test.go`, `pkg/layers/fc.go`, `pkg/layers/fc_test.go`, `pkg/utils/utils.go`, and `pkg/utils/utils_test.go` were not found and therefore are not included in this report.
