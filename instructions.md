## 1 Missing Neural-Network Operators

| Operator / Feature                      | Why you need it                                                                  | How to make it HE-friendly (CKKS)                                                                                               | Implementation sketch                                                          
| **Average-pooling (k × k, stride ≥ k)** | Reduces spatial size without extra depth—crucial for ImageNet-sized inputs.      | *Linear*: just a sum of rotated copies then scale by `1/k²`.                                                                    | Re-use the rotation machinery you built for Conv: compute the same rotation set once, sum, then multiply by pre-encoded scalar `1/k²`. |
| **Global average pooling (GAP)**        | Replaces the last fully-connected layer in many mobile nets; slashes parameters. | Same as above with kernel = `H×W`.                                                                                              | One ciphertext holds all pixels: rotate-and-add in a balanced tree, then scale.                                                        |
| **Skip/residual adds**                  | Needed for ResNet-style backbones (the simplest ImageNet-capable models).        | Pure ciphertext additions (depth-0).                                                                                            | Provide `func Add(ctA, ctB *Ciphertext) *Ciphertext`.                                                                                  |
| **Channel concatenation / split**       | Used in MobileNet‐V2, ShuffleNet, etc.                                           | Pure slot permutations; often free if you pack each channel in its own contiguous block.                                        | Build helper to remap slots without extra depth (Galois rotation by `+H*W` multiples).                                                 |
| **Flatten → Linear**                    | Last stage before logits.                                                        | You already have Linear; ensure packing order matches Linear’s weight layout.                                                   | Write `packFlatten(ctFeatureMap)`.                                                                                                     |
| **Softmax or ArgMax**                   | Final prediction.                                                                | (a) Decrypt logits and compute softmax in clear; or (b) homomorphic softmax via low-degree Chebyshev (\~degree 3) then decrypt. | For benchmarking, (a) is easiest.                                                                                                      |

> **Tip:** Batch Norm, Layer Norm, and affine scales can be **fused** into preceding Conv or Linear weights offline—no extra HE op needed.

---

## 2 Noise-Budget & Depth Management

| Layer type                 | Depth cost              | Typical count in ImageNet-scale net | Cumulative depth (no bootstrapping) |
| -------------------------- | ----------------------- | ----------------------------------- | ----------------------------------- |
| Conv 3×3                   | 1 mul                   | 15–20 (ResNet-18)                   | 15–20                               |
| ReLU approximation (deg-3) | 2 mul (square × linear) | 15–20                               | 30–40                               |
| GAP + FC                   | 1 mul                   | 1                                   | 31–41                               |

CKKS parameter sets with `logQ≈440` (N = 2¹⁵) allow ≈ 30–32 levels.  **Conclusion:** for vanilla ResNet-18 you will run out of budget; you will therefore need **bootstrapping** *or* a shallower / polynomial-friendlier model.

---

## 3 Choosing a Practical “Starter” Architecture

| Candidate                                        | Params / FLOPs | ImageNet Top-1 (FP32) | HE suitability                                                        | Required ops                                                      |
| ------------------------------------------------ | -------------- | --------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **ResNet-8 (3×3, no bottlenecks)**               | \~3 M / 0.7 G  | ≈52 %                 | Passes within 28 levels with degree-2 activations.                    | Conv, AvgPool, ResidualAdd, FC                                    |
| **TF-CryptoNet (LeNet variant, 224²→28² first)** | <1 M / 0.3 G   | ≈40 %                 | Very light; no residuals; tailors well to HE.                         | Conv, AvgPool, FC                                                 |
| **HE-MobileNetV1-0.25**                          | \~1 M / 40 M   | ≈50 %                 | Depthwise separable convs cut rotations by ×8; but need concat/split. | DepthwiseConv (special case of Conv), PointwiseConv, AvgPool, GAP |

*Start with ResNet-8 or CryptoNet-XL; both are published as “FHE-friendly baselines” and run on a single ciphertext.*

---

## 4 Bootstrapping (If Needed)

* **When?** After every 8-10 non-linear layers or whenever scale drops below \~2²⁰.
* **What to implement?** Lattigo’s built-in CKKS bootstrap API. Provide a wrapper:

  ```go
  func Refresh(ct *rlwe.Ciphertext, eval rlwe.Evaluator) (*rlwe.Ciphertext, error)
  ```
* **Cost:** \~1.5 s (N=2¹⁵) and adds ≈30 bits of fresh modulus.

---

## 5 Dataset Ingestion Pipeline

1. **Pre-process ImageNet JPEG → float\[0,1]** (subtract per-channel mean, divide std).
   *Fold constants into first Conv weights to avoid an extra ciphertext multiply.*
2. **Pack** 224×224×3 float tensor into one CKKS plaintext using the slot-index formula from the Conv PRD.
3. **Encrypt** with user’s public key → ciphertext image.
4. **Inference** through your HE model.
5. **Decrypt** logits, softmax + argmax in clear.

Provide CLI helper:

```bash
go run ./cmd/he_infer \
    -model resnet8_he.onnx \
    -weights resnet8_he.ckpt \
    -pubkey client.pk \
    -image sample.jpg
```

---

## 6 Benchmark & Validation Suite

| Metric                             | Target                      | How to measure                  |
| ---------------------------------- | --------------------------- | ------------------------------- |
| End-to-end latency (single image)  | ≤ 60 s (without bootstraps) | `time ./he_infer …`             |
| Throughput (batch=8)               | ≥ 0.15 img/s                | loop encrypt + infer            |
| Top-1 accuracy drop vs. plain FP32 | ≤ 4 pp                      | Evaluate on ImageNet-val (50 k) |
| Memory                             | ≤ 8 GB RAM                  | `pprof`                         |

---

## 7 Project “Done” Checklist

* [ ] Average-pooling & GAP operators coded and unit-tested.
* [ ] Residual add / slot concat helpers.
* [ ] Packing helpers for 224×224 inputs.
* [ ] Degree-2 or degree-3 polynomial activation encoded.
* [ ] Optional bootstrap wrapper and schedule.
* [ ] ResNet-8‐HE weights exported (PyTorch → numpy → CKKS encode).
* [ ] Integration test decrypts logits and matches plain ResNet-8 within 4 pp top-1.

Once these boxes are ticked you can credibly claim “ImageNet inference over encrypted data” with a **simple, reproducible model**.