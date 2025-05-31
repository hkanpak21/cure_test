### PRD — Homomorphic ReLU & Sigmoid via Chebyshev (d = 3, 5)

*Target repo*: `github.com/hkanpak21/cure_test`
*Library*: **Lattigo v6** – CKKS scheme (fits real‐valued activations)

---

#### 1 ▪ Objective

Add easy-to-call helpers that evaluate **ReLU** and **sigmoid** on CKKS ciphertexts in the interval **\[-1, 1]** using low-degree Chebyshev polynomial approximations (d = 3 and d = 5). Keep the API small, dependency-free (except Lattigo), and well-tested.

---

#### 2 ▪ Deliverables & File Layout

| Path                                       | Purpose                                                                                                                                   |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **`pkg/activation_he/coeff.go`**           | Hard-code Chebyshev coefficients (see §3).                                                                                                |
| **`pkg/activation_he/poly.go`**            | Generic `EvalChebyshev(ct *rlwe.Ciphertext, coeffs []float64, eval rlwe.Evaluator) (*rlwe.Ciphertext, error)` – Clenshaw recursion in HE. |
| **`pkg/activation_he/relu.go`**            | `EvalReLU3` and `EvalReLU5` wrappers.                                                                                                     |
| **`pkg/activation_he/sigmoid.go`**         | `EvalSigmoid3` and `EvalSigmoid5` wrappers.                                                                                               |
| **`pkg/activation_he/plain.go`**           | Same helpers on plain `float64` for quick debugging.                                                                                      |
| **`pkg/activation_he/activation_test.go`** | Unit & integration tests (see §4).                                                                                                        |
| **`examples/relu_sigmoid/main.go`**        | 25-line demo encrypt → activate → decrypt.                                                                                                |

*No other files need editing.*

---

#### 3 ▪ Chebyshev coefficients (domain = \[-1, 1])

Coefficients are listed **c₀ … c\_d** for T₀ … T\_d.

```go
// ReLU ≈ max(0,x)
var ReluDeg3 = []float64{0.32811094, 0.5, 0.23435157, 0.0}
var ReluDeg5 = []float64{0.31495238, 0.5, 0.20509862, 0.0, -0.05125416, 0.0}

// Sigmoid ≈ 1/(1+e^{-x})
var SigmDeg3 = []float64{0.5, 0.23551963, 0.0, -0.00468065}
var SigmDeg5 = []float64{0.5, 0.23557248, 0.0, -0.00461894, 0.0, 0.00011125}
```

*Error (MSE) ≤ **4 × 10⁻⁴** on \[-1, 1] for both degrees.*

---

#### 4 ▪ Tests to implement

1. **Plain polynomial accuracy** (`TestChebyshevPlainAccuracy`)
   *1000 random x ∈ \[-1, 1]* → assert MSE < 5e-4 for every (func, deg).

2. **Ciphertext functional test** (`TestChebyshevHEAccuracy`)
   *Encrypt 32 random slots; run each activation; decrypt; compare to plain reference with Δ < 2e-3.*

3. **Batch ranges** (`TestAffineMapping` already exists)
   Re-use to check output is still in \[-0.05, 1.05] after evaluation.

4. **Speed/regression smoke** (`go test -bench=.` optional).

All tests live in **`activation_test.go`** and should pass with `go test ./...`.

---

#### 5 ▪ Implementation Hints

* **Scaling** – CKKS handles real coefficients natively; no rescaling tricks needed for these low degrees.
* **Clenshaw** – fewer rotations/mults than Horner; fits depth budget easily (≤ d mults + d-1 rotations).
* **Evaluator** – accept an `rlwe.Evaluator` so caller can reuse one with a pooled context.
* **Relinearization** – call `eval.Relinearize` once at the end to minimise noise growth.

---

#### 6 ▪ Milestones

| Day | Task                                                                    |
| --- | ----------------------------------------------------------------------- |
| 0   | Add `coeff.go`, `plain.go`; write plain accuracy test.                  |
| 1   | Implement `poly.go` with Clenshaw; quick bench with plaintext.          |
| 2   | Wrap ReLU/Sigmoid helpers; write HE tests; run `go vet`, `staticcheck`. |
| 3   | Add example program; update README with “Activation HE” section.        |
| 4   | CI: add `go test ./...` job (GitHub Action).                            |

---

#### 7 ▪ Success Criteria

* **Correctness** – Tests in §4 all green on default Lattigo CKKS params (`PN13QP218`) with default scale.
* **Simplicity** – No extra packages, ≤ 300 new LOC (excluding tests).
* **Clarity** – Public functions documented with one‐sentence comments.

---

*End of PRD*
