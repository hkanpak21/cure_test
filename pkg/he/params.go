package he

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
	// Set1 represents the first parameter set defined in the CURE paper (Sec V-A.2).
	// TODO: Replace placeholder values with actual parameters from CURE paper for Set1.
	Set1 ParameterSetIdentifier = "Set1"
	// Set2 represents the second parameter set defined in the CURE paper (Sec V-A.2).
	// TODO: Replace placeholder values with actual parameters from CURE paper for Set2.
	Set2 ParameterSetIdentifier = "Set2"
)

// GetCKKSParameters initializes and returns Lattigo CKKS parameters based on the identifier.
func GetCKKSParameters(paramSetID ParameterSetIdentifier) (params ckks.Parameters, err error) {
	switch paramSetID {
	case DefaultSet:
		params, err = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN:            12,    // Ring degree N = 2^12 (for faster testing)
			LogQ:            []int{55, 40, 40, 40, 40, 40}, // Ciphertext modulus Q (6 levels)
			LogP:            []int{60, 60},                // Key-switching modulus P
			LogDefaultScale: 40,                         // Default scaling factor 2^40
		})
		if err != nil {
			return ckks.Parameters{}, fmt.Errorf("failed to create CKKS parameters for %s: %w", paramSetID, err)
		}
	case Set1:
		// Placeholder for Set1 parameters from CURE paper (Sec V-A.2)
		// TODO: Replace with actual values from CURE paper.
		params, err = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN:            14,
			LogQ:            []int{55, 40, 40, 40, 40, 40, 40, 40}, // Example: 8 levels
			LogP:            []int{60, 60},
			LogDefaultScale: 40,
		})
		if err != nil {
			return ckks.Parameters{}, fmt.Errorf("failed to create CKKS parameters for %s (placeholder): %w", paramSetID, err)
		}
	case Set2:
		// Placeholder for Set2 parameters from CURE paper (Sec V-A.2)
		// TODO: Replace with actual values from CURE paper.
		params, err = ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
			LogN:            15,
			LogQ:            []int{60, 45, 45, 45, 45, 45, 45, 45, 45, 45}, // Example: 10 levels
			LogP:            []int{61, 61},
			LogDefaultScale: 45,
		})
		if err != nil {
			return ckks.Parameters{}, fmt.Errorf("failed to create CKKS parameters for %s (placeholder): %w", paramSetID, err)
		}
	default:
		return ckks.Parameters{}, fmt.Errorf("unknown parameter set identifier: %s", paramSetID)
	}
	return
}

// KeyGenerator returns a new Lattigo CKKS key generator.
func KeyGenerator(params ckks.Parameters) *rlwe.KeyGenerator {
	return ckks.NewKeyGenerator(params)
}

// NewEncoder creates and returns a new CKKS encoder.
func NewEncoder(params ckks.Parameters) *ckks.Encoder {
	return ckks.NewEncoder(params)
}

// NewEncryptor creates and returns a new RLWE encryptor from a public key.
func NewEncryptor(params ckks.Parameters, pk *rlwe.PublicKey) *rlwe.Encryptor {
	return ckks.NewEncryptor(params, pk)
}

// NewDecryptor creates and returns a new RLWE decryptor from a secret key.
func NewDecryptor(params ckks.Parameters, sk *rlwe.SecretKey) *rlwe.Decryptor {
	return ckks.NewDecryptor(params, sk)
}

// NewEvaluator creates and returns a new CKKS evaluator from an evaluation key set.
func NewEvaluator(params ckks.Parameters, evk rlwe.EvaluationKeySet) *ckks.Evaluator {
	return ckks.NewEvaluator(params, evk)
}
