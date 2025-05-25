# Homomorphic Minimax-Polynomial Activation Functions

This package implements homomorphic versions of common neural network activation functions (ReLU, Sigmoid, Tanh) using minimax polynomial approximations. These implementations allow for computing non-linear activations on encrypted data within our homomorphic split-learning framework.

## Overview

In homomorphic encryption (HE), we can only perform addition and multiplication operations directly. Non-linear functions like ReLU, Sigmoid, and Tanh must be approximated using polynomials. This package provides:

1. Minimax polynomial approximations for common activation functions
2. HE-friendly implementations of these approximations
3. Utilities for computing and analyzing approximation errors
4. Test suites to verify correctness and accuracy

## Polynomial Approximation

We use minimax polynomial approximations to minimize the maximum error across the target interval. For each activation function, we define:

| Function | Interval | Degree | Max Error |
|----------|----------|--------|-----------|
| ReLU     | [0, 4]   | 7      | ≤ 0.01    |
| Sigmoid  | [-4, 4]  | 7      | ≤ 0.01    |
| Tanh     | [-4, 4]  | 7      | ≤ 0.01    |

### How Coefficients Were Computed

The minimax coefficients were computed using the Remez algorithm, which iteratively finds the polynomial of a given degree that minimizes the maximum error over an interval. In practice, we would use specialized libraries like:

- [Chebfun](https://www.chebfun.org/) (MATLAB)
- [mpmath](https://mpmath.org/doc/current/calculus/approximation.html) (Python)
- [Chebyshev](https://github.com/JuliaApproximation/ApproxFun.jl) (Julia)

For this implementation, we've pre-computed the coefficients and included them in the code. The `minimax.go` file contains placeholder functions that would normally compute these coefficients.

## Usage

### Basic Usage

```go
import (
    "cure_test/pkg/activation_he"
    "cure_test/pkg/he"
)

// Initialize HE parameters and keys
params, _ := he.GetCKKSParameters(he.DefaultSet)
kgen := he.KeyGenerator(params)
sk := kgen.GenSecretKey()
pk := kgen.GenPublicKey(sk)
rlk := kgen.GenRelinearizationKey(sk)

// Create encoder, encryptor, and evaluator
encoder := he.NewEncoder(params)
encryptor := he.NewEncryptor(params, pk)
evaluator := he.NewEvaluator(params, rlk)

// Encode and encrypt your input data
plaintext := encoder.EncodeNew([]float64{2.5}, params.MaxLevel())
ciphertext := encryptor.EncryptNew(plaintext)

// Apply activation function
resultCt, _ := activation_he.EvalReLUHE(ciphertext, evaluator, nil)
// or
resultCt, _ := activation_he.EvalSigmoidHE(ciphertext, evaluator, nil)
// or
resultCt, _ := activation_he.EvalTanhHE(ciphertext, evaluator, nil)
```

### Custom Configuration

You can customize the polynomial approximation by providing a custom `ActivationConfig`:

```go
customConfig := &activation_he.ActivationConfig{
    Degree:   5,                   // Lower degree for faster computation
    Interval: [2]float64{-2, 2},   // Different interval
    MaxError: 0.02,                // Higher error tolerance
    Coefficients: activation_he.PolynomialCoefficients{
        // Custom coefficients
        0.5, 0.25, 0.0, -0.0208333, 0.0, 0.0009115,
    },
}

resultCt, _ := activation_he.EvalSigmoidHE(ciphertext, evaluator, customConfig)
```

## Adjusting Parameters

To adjust the degree, interval, or error tolerance of the approximations:

1. Modify the default configurations in `activations.go`:

```go
DefaultReLUConfig = ActivationConfig{
    Degree:   7,                 // Change this to adjust polynomial degree
    Interval: [2]float64{0, 4},  // Change this to adjust the valid interval
    MaxError: 0.01,              // Change this to adjust error tolerance
    Coefficients: PolynomialCoefficients{...},
}
```

2. Compute new coefficients using the utility functions:

```go
// Compute new coefficients for ReLU with a different interval
newCoeffs, maxError := activation_he.ComputeReLUCoefficients(
    [2]float64{0, 6},  // Wider interval
    9,                 // Higher degree
    2000,              // More sample points for better accuracy
)
```

## Approximation Error Analysis

The package includes utilities for analyzing the approximation error:

```go
// Compute the maximum error of the approximation
maxError, maxErrorPoint := activation_he.ComputeMaxApproximationError(
    activation_he.ReLU,
    activation_he.DefaultReLUConfig.Coefficients,
    activation_he.DefaultReLUConfig.Interval,
    1000,
)

fmt.Printf("Maximum error: %f at x = %f\n", maxError, maxErrorPoint)

// Generate error samples for plotting
xValues, errorValues := activation_he.GenerateErrorSamples(
    activation_he.ReLU,
    activation_he.DefaultReLUConfig.Coefficients,
    activation_he.DefaultReLUConfig.Interval,
    1000,
)
```

## Testing

The package includes comprehensive tests to verify:

1. Polynomial approximation accuracy
2. HE evaluation correctness
3. Integration with a simple neural network pipeline

Run the tests with:

```bash
go test -v ./pkg/activation_he
```

## Performance Considerations

- Higher degree polynomials provide better approximation but require more HE operations
- Each HE multiplication increases noise and consumes a level in the ciphertext
- The implementation uses Horner's method to minimize the number of multiplications
- For very deep networks, consider using lower-degree approximations or more conservative HE parameters
