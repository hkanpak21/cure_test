
## 6. Phased Development Plan

To systematically achieve the project's goals, I propose the following phased development plan. Each step includes specific implementation tasks and corresponding testing requirements.

---

### Phase 1: Core Mathematical Operations

#### 1.1. Plaintext Matrix Operations (Target: `pkg/matrix/`)

*   **Step 1.1.1: Implementation - Basic Matrix Multiplication**
    *   Develop Go functions for standard dense matrix multiplication (`C = A * B`).
    *   Represent matrices (e.g., `[][]float64` or a custom `Matrix` struct) and handle dimension compatibility.
*   **Step 1.1.2: Tests - Basic Matrix Multiplication**
    *   Unit tests for correctness with known small matrices, various dimensions (square, rectangular), dimension mismatch errors, and edge cases (1xN, Nx1 matrices).

*   **Step 1.1.3: Implementation - Element-wise Matrix Operations**
    *   Implement functions for element-wise addition, subtraction of matrices.
*   **Step 1.1.4: Tests - Element-wise Matrix Operations**
    *   Unit tests for correctness and dimension mismatch errors.

#### 1.2. Homomorphic Encryption (HE) Aware Matrix Operations (Target: `pkg/he/`)

*   **Context:** Implement matrix operations with Lattigo, focusing on one-level (plaintext-ciphertext) and n-layer (ciphertext-ciphertext) scenarios as per CURE (Sec IV-C).

*   **Step 1.2.1: Setup - Lattigo CKKS Context**
    *   Implement utilities in `pkg/he/params.go` to initialize Lattigo CKKS `Parameters` and key generators, configurable based on paper's `Set 1` / `Set 2` (Sec V-A.2).
*   **Step 1.2.2: Implementation - Scalar-CiphertextVector Multiplication (One-Level Scalar Method, Sec IV-C2)**
    *   Implement `v * [w0, w1, ..., wn] -> [v*w0, v*w1, ..., v*wn]` where `v` is plaintext scalar and `W` is an encrypted vector.
*   **Step 1.2.3: Tests - Scalar-CiphertextVector Multiplication**
    *   Encrypt a vector, perform scalar multiplication, decrypt, and verify. Test various scalars.

*   **Step 1.2.4: Implementation - PlaintextVector-CiphertextVector Element-wise Multiplication (One-Level Batch Method, Sec IV-C2)**
    *   Implement `[v0,v1,...] * [w0,w1,...] -> [v0*w0, v1*w1,...]` where `V` is plaintext, `W` is encrypted.
*   **Step 1.2.5: Tests - PlaintextVector-CiphertextVector Element-wise Multiplication**
    *   Encrypt a vector, prepare plaintext vector, multiply, decrypt, and verify.

*   **Step 1.2.6: Implementation - HE Matrix-Vector Multiplication (Server's First Layer Logic)**
    *   Focus on `Ws_enc * X_plain`. Each column of `Ws_enc` is an encrypted vector.
    *   Implement the "scalar inspired" approach: sum of (plaintext scalar from `X`) * (encrypted column vector from `Ws`).
*   **Step 1.2.7: Tests - HE Matrix-Vector Multiplication**
    *   Test with small encrypted matrix `Ws` and plaintext vector `X`. Decrypt and verify.

*   **Step 1.2.8: Implementation - HE Matrix-Matrix Multiplication (n-encrypted layers, Sec IV-C3 & Supp. VII-D)**
    *   Implement ciphertext-ciphertext multiplication with packing, rotations, and summation for dot products as described.
*   **Step 1.2.9: Tests - HE Matrix-Matrix Multiplication**
    *   Test with small encrypted matrices. Independently test packing and rotation.

---

### Phase 2: Activation Functions (Target: `pkg/nn/activation/`)

#### 2.1. Plaintext Activation Functions

*   **Step 2.1.1: Implementation - Sigmoid**
    *   Implement standard sigmoid, element-wise for matrices/vectors.
*   **Step 2.1.2: Tests - Sigmoid**
    *   Test with various inputs and output range.

*   **Step 2.1.3: Implementation - ReLU**
    *   Implement standard ReLU, element-wise.
*   **Step 2.1.4: Tests - ReLU**
    *   Test with various inputs.

#### 2.2. HE-Approximated Activation Functions

*   **Step 2.2.1: Implementation - Polynomial Approximation for Sigmoid (HE)**
    *   Implement polynomial evaluation on an encrypted input using Lattigo. Use coefficients for a degree ~3-7 polynomial approximating sigmoid (e.g., degree 7 over [-15, 15] as per paper).
*   **Step 2.2.2: Tests - Polynomial Approximation for Sigmoid (HE)**
    *   Encrypt input, compute `p(x)` homomorphically, decrypt, compare to plaintext `p(x)` and `sigmoid(x)`. Monitor HE levels.

*   **Step 2.2.3: Research/Implementation - Polynomial Approximation for ReLU (HE)**
    *   (Optional, if deemed critical for encrypted layers) Research and implement a suitable low-degree polynomial approximation.
*   **Step 2.2.4: Tests - Polynomial Approximation for ReLU (HE)**
    *   (If implemented) Similar tests as for sigmoid.

---

### Phase 3: CURE Training Functionality (Targets: `pkg/nn/`, `pkg/dataloader/`, `pkg/cure/`, `cmd/cure_app/`)

#### 3.1. Model and Data Structures

*   **Step 3.1.1: Implementation - Layer & Model Definition (`pkg/nn/`)**
    *   Define structs for NN Layers (Dense) and the overall NN Model. Include weight initialization.
*   **Step 3.1.2: Implementation - Data Loader (`pkg/dataloader/`)**
    *   Create a CSV data loader (e.g., for MNIST). Implement batching.
*   **Step 3.1.3: Tests - Data Loader and Model Structures**
    *   Test weight initialization, data loading, and batching.

#### 3.2. CURE Protocol Implementation (Simplified Network First - `pkg/cure/`)

*   **Context:** Implement Algorithm 1 from the paper with Client and Server modules. Start simple: 1 HE server layer, 1 plaintext client layer.

*   **Step 3.2.1: Initialization (Client & Server)**
    *   **Client:** Generates Lattigo keys, initializes `Wc`, sends PKc.
    *   **Server:** Receives PKc, initializes `Ws_plain`, encrypts to `Ws_enc`.
*   **Step 3.2.2: Server Forward Pass (Encrypted)**
    *   Server computes `On_enc = Eval_PKc(Ws_enc, X_input)` using Phase 1 ops. Apply HE activations if multi-layer server. Sends `On_enc` to Client.
*   **Step 3.2.3: Client Plaintext Operations (Forward, Loss, Gradients)**
    *   Client decrypts `On_enc` to `On_plain`.
    *   Client: forward pass `Ŷ = fc(Wc, On_plain)`, computes loss `J`, computes gradients `gw_c` and `gw_s_intermediate`.
    *   Client updates `Wc`, encrypts `gw_s_enc = Enc_PKc(gw_s_intermediate)`, sends to Server.
*   **Step 3.2.4: Server Backward Pass (Encrypted Gradient Update)**
    *   Server updates `Ws_enc_new = EvalUpdate_PKc(Ws_enc, gw_s_enc)`.
*   **Step 3.2.5: Tests - Individual Protocol Steps**
    *   Unit test each client/server communication and computation step for correctness.

#### 3.3. Full Training Loop and Experimental Evaluation (`pkg/cure/training.go`, `cmd/cure_app/main.go`)

*   **Step 3.3.1: Implementation - Epoch Loop & Metrics**
    *   Orchestrate steps 3.2.1-3.2.4 over epochs and batches. Log loss/accuracy. Measure time.
*   **Step 3.3.2: Integration Test - End-to-End Training (Simple Network)**
    *   Run full training on a toy dataset. Verify decreasing loss.
*   **Step 3.3.3: Experimental Setup - MNIST (or similar)**
    *   Configure a network architecture from the CURE paper (e.g., Model 1 or Model 6).
*   **Step 3.3.4: Run Experiments & Collect Results**
    *   Train, record accuracy, time. Compare with paper's Table II/IV. Address bootstrapping if HE levels are exhausted (advanced).

---

### Phase 4: Advanced Features & Refinements (Future Work)

*   **Step 4.1:** Advanced Packing Optimizations (Sec IV-C2, IV-C3, Supp. VII-B, VII-D).
*   **Step 4.2:** Full Bootstrapping Integration if required.
*   **Step 4.3:** CURE Estimator/Advisor Implementation (Sec IV-D).
*   **Step 4.4:** Networked Client-Server Communication (e.g., gRPC).
*   **Step 4.5:** Support for more complex network types (e.g., ResNet).

---

## 7. Non-Functional Requirements

*   **Performance:** HE operations should be efficient enough for practical experimentation.
*   **Testability:** Emphasize comprehensive unit and integration testing.
*   **Scalability (Conceptual):** Design with future scalability in mind.

## 8. Definition of Success

The initial success of this reimplementation initiative will be measured by:

*   The successful execution of the complete training loop on a standard dataset (e.g., MNIST) with observable learning (decreasing loss).
*   Achieving model accuracy comparable to plaintext baselines and fully-encrypted approaches reported in the CURE paper (within an acceptable margin due to HE approximations).
*   The completion and successful testing of all development steps outlined in Phases 1 through 3.
*   Demonstrable runtime performance for HE operations that aligns with expectations derived from the CURE paper, considering HE parameter choices and available computational resources.

This proposed plan aims to provide a clear roadmap for the CURE system reimplementation. I am ready to proceed with Phase 1, commencing with the development of plaintext matrix operations.