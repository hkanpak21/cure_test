# cure_test

## Current Status

This project is focused on implementing and testing functionalities related to the CURE paper, with an emphasis on Homomorphic Encryption (HE) operations.

The current development efforts include:

*   **HE Layer Setup (Lattigo v5.0.7):**
    *   The Lattigo library (v5.0.7) has been integrated as the primary HE backend.
    *   Dependencies are managed via `go.mod` and `go.sum`, and `go mod tidy` has been run to ensure consistency.

*   **CKKS Parameters & Key Management (`pkg/he/params/params.go`):**
    *   A `DefaultSet` of CKKS parameters has been defined for general use.
    *   Placeholders for `Set1` and `Set2` (as specified in the CURE paper, Sec V-A.2) are included with `TODO` markers for future implementation.
    *   The groundwork for key generation (public, private, relinearization, rotation keys) and helper functions for `Encoder`, `Encryptor`, `Decryptor`, and `Evaluator` is being laid out.

*   **HE Operations (`pkg/he/ops/ops.go`):**
    *   A function `ScalarMultCiphertext` has been implemented, enabling the multiplication of a ciphertext by a scalar value using `evaluator.MultByConstNew`.

## Project Structure

Here is a high-level overview of the project's directory structure:

```
.
├── README.md
├── go.mod
├── go.sum
├── cmd/                # Main applications
│   ├── cure-infer/
│   ├── cure-split/
│   └── cure-train/
├── docs/               # Documentation files
├── examples/           # Example usage scripts
│   └── activation_demo.go
├── internal/           # Internal helper packages
│   ├── math/
│   └── parallel/
├── pkg/                # Core library packages
│   ├── activation_he/  # HE-based activation functions
│   │   ├── README.md
│   │   ├── activations.go
│   │   ├── activations_test.go
│   │   ├── minimax.go
│   │   ├── test_helpers.go
│   │   └── utils.go
│   ├── data/           # Data loading and processing
│   ├── he/             # Homomorphic Encryption core logic
│   │   ├── he.go         # Main HE definitions (if any)
│   │   ├── ops/          # HE operations (e.g., scalar mult)
│   │   │   └── ops.go
│   │   └── params/       # HE parameters and key generation
│   │       └── params.go
│   ├── layers/         # Neural network layers
│   │   └── conv.go
│   ├── matrix/         # Matrix operations
│   │   ├── matrix.go
│   │   └── matrix_test.go
│   ├── model/          # Model definitions
│   └── train/          # Training logic
├── scripts/            # Utility scripts
└── tests/              # Test files
    ├── benchmarks/     # Benchmark tests
    ├── he/             # HE specific tests
    │   └── ops_test.go
    └── integration/    # Integration tests
```

## Next Steps

*   Implement `Set1` and `Set2` CKKS parameters in `pkg/he/params/params.go` based on the CURE paper.
*   Continue developing core HE operations in `pkg/he/ops/ops.go`.
*   Expand test coverage for implemented HE functionalities.