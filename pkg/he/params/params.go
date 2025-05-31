// Package params provides homomorphic encryption parameter sets and key generation utilities.
package params

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// ParameterSetIdentifier is a type to distinguish between different HE parameter sets.
// We can define specific sets like "Set1" or "Set2" from the CURE paper, or a general "DefaultSet".
type ParameterSetIdentifier string

const (
	// DefaultSet provides a general-purpose CKKS parameter configuration.
	DefaultSet ParameterSetIdentifier = "DefaultSet"
	// Set1 corresponds to the Set1 parameters from the CURE paper.
	Set1 ParameterSetIdentifier = "Set1"
	// Set2 corresponds to the Set2 parameters from the CURE paper.
	Set2 ParameterSetIdentifier = "Set2"
	// TestSet provides a faster parameter set for testing with log n = 12
	TestSet ParameterSetIdentifier = "TestSet"
)

// GetCKKSParameters initializes and returns Lattigo CKKS parameters based on the identifier.
func GetCKKSParameters(paramSetID ParameterSetIdentifier) (params ckks.Parameters, err error) {
	switch paramSetID {
	case DefaultSet:
		// Default general-purpose parameter set for testing
		params, err = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN:         14,                      // Ring degree: 2^14 = 16384
			LogQ:         []int{55, 40, 40, 45, 45, 45}, // Increased last 3 primes for better depth // Modulus chain for ciphertext operations
			LogP:         []int{60, 60},          // Special modulus for key switching
			LogDefaultScale: 50,                 // Default scale for encoding: 2^50
		})
		if err != nil {
			return params, fmt.Errorf("failed to create default parameters: %w", err)
		}

	case TestSet:
		// Faster parameter set for testing with log n = 12
		// Adding more levels to support higher-degree polynomials (degree 7 requires 9 levels)
		params, err = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN:         12,                      // Ring degree: 2^12 = 4096 (smaller for faster testing)
			LogQ:         []int{40, 30, 30, 30, 30, 30, 30, 30, 30, 30},  // 10 levels for polynomials up to degree 7
			LogP:         []int{45, 45},          // Special modulus for key switching
			LogDefaultScale: 30,                 // Default scale for encoding: 2^30
		})
		if err != nil {
			return params, fmt.Errorf("failed to create test parameters: %w", err)
		}

	case Set1:
		// TODO: Replace with actual Set1 parameters from the CURE paper
		// For now, using the same as DefaultSet
		return GetCKKSParameters(DefaultSet)

	case Set2:
		// TODO: Replace with actual Set2 parameters from the CURE paper
		// For now, using the same as DefaultSet
		return GetCKKSParameters(DefaultSet)

	default:
		return params, fmt.Errorf("unknown parameter set identifier: %s", paramSetID)
	}

	return params, nil
}

// KeyGenerator returns a new Lattigo CKKS key generator.
func KeyGenerator(params ckks.Parameters) *rlwe.KeyGenerator {
	return rlwe.NewKeyGenerator(params)
}

// NewEncoder creates and returns a new CKKS encoder.
func NewEncoder(params ckks.Parameters) *ckks.Encoder {
	return ckks.NewEncoder(params)
}

// NewEncryptor creates and returns a new RLWE encryptor from a public key.
func NewEncryptor(params ckks.Parameters, pk *rlwe.PublicKey) *rlwe.Encryptor {
	return rlwe.NewEncryptor(params, pk)
}

// NewDecryptor creates and returns a new RLWE decryptor from a secret key.
func NewDecryptor(params ckks.Parameters, sk *rlwe.SecretKey) *rlwe.Decryptor {
	return rlwe.NewDecryptor(params, sk)
}

// NewEvaluator creates and returns a new CKKS evaluator from an evaluation key set.
func NewEvaluator(params ckks.Parameters, evk rlwe.EvaluationKeySet) *ckks.Evaluator {
	return ckks.NewEvaluator(params, evk)
}
