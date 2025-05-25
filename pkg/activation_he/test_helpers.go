package activation_he

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// createEvaluator creates a CKKS evaluator with the provided parameters and relinearization key
// This is a helper function for tests to ensure compatibility with Lattigo v6 API
func createEvaluator(params ckks.Parameters, rlk *rlwe.RelinearizationKey) *ckks.Evaluator {
	// Create a memory evaluation key set with the relinearization key
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	
	// Create an evaluator with the evaluation key set
	return ckks.NewEvaluator(params, evk)
}
