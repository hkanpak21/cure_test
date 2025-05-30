// Package he provides homomorphic encryption primitives for secure computations.
// It wraps the Lattigo library functionality with specialized functions for neural network operations.
package he

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"

	"cure_test/pkg/he/ops"
	"cure_test/pkg/he/params"
)

// Re-export types from params package
type ParameterSetIdentifier = params.ParameterSetIdentifier

// Define constants re-exported from params
const (
	DefaultSet = params.DefaultSet
	Set1       = params.Set1
	Set2       = params.Set2
	TestSet    = params.TestSet
)

// GetCKKSParameters initializes and returns Lattigo CKKS parameters based on the identifier.
func GetCKKSParameters(paramSetID ParameterSetIdentifier) (ckks.Parameters, error) {
	return params.GetCKKSParameters(paramSetID)
}

// KeyGenerator returns a new Lattigo CKKS key generator.
func KeyGenerator(parameters ckks.Parameters) *rlwe.KeyGenerator {
	return rlwe.NewKeyGenerator(parameters)
}

// NewEncoder creates and returns a new CKKS encoder.
func NewEncoder(parameters ckks.Parameters) *ckks.Encoder {
	return ckks.NewEncoder(parameters)
}

// NewEncryptor creates and returns a new RLWE encryptor from a public key.
func NewEncryptor(parameters ckks.Parameters, pk *rlwe.PublicKey) *rlwe.Encryptor {
	return rlwe.NewEncryptor(parameters, pk)
}

// NewDecryptor creates and returns a new RLWE decryptor from a secret key.
func NewDecryptor(parameters ckks.Parameters, sk *rlwe.SecretKey) *rlwe.Decryptor {
	return rlwe.NewDecryptor(parameters, sk)
}

// NewEvaluator creates and returns a new CKKS evaluator from an evaluation key set.
func NewEvaluator(parameters ckks.Parameters, evk rlwe.EvaluationKeySet) *ckks.Evaluator {
	return ckks.NewEvaluator(parameters, evk)
}

// ScalarMultCiphertext performs scalar multiplication on an encrypted vector (ciphertext).
func ScalarMultCiphertext(scalar float64, ctIn *rlwe.Ciphertext, evaluator *ckks.Evaluator) (*rlwe.Ciphertext, error) {
	return ops.ScalarMultCiphertext(scalar, ctIn, evaluator)
}

// MulCiphertexts performs homomorphic multiplication of two ciphertexts.
func MulCiphertexts(ct1, ct2 *rlwe.Ciphertext, evaluator *ckks.Evaluator) (*rlwe.Ciphertext, error) {
	return ops.MulCiphertexts(ct1, ct2, evaluator)
}

// MulMatricesCiphertexts performs homomorphic matrix multiplication.
func MulMatricesCiphertexts(
	ctaRows []*rlwe.Ciphertext,
	ctbCols []*rlwe.Ciphertext,
	k_dimension int,
	evaluator *ckks.Evaluator,
) ([][]*rlwe.Ciphertext, error) {
	return ops.MulMatricesCiphertexts(ctaRows, ctbCols, k_dimension, evaluator)
}

// MulMatricesCiphertextsParallel performs homomorphic matrix multiplication in parallel.
func MulMatricesCiphertextsParallel(
	ctaRows []*rlwe.Ciphertext,
	ctbCols []*rlwe.Ciphertext,
	k_dimension int,
	evaluator *ckks.Evaluator,
	numWorkers int,
) ([][]*rlwe.Ciphertext, error) {
	return ops.MulMatricesCiphertextsParallel(ctaRows, ctbCols, k_dimension, evaluator, numWorkers)
}
