package main

import (
	"fmt"
	"log"

	"cure_test/pkg/activation_he"
	"cure_test/pkg/he"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

func main() {
	// Initialize CKKS parameters and keys
	params, _ := he.GetCKKSParameters(he.TestSet)
	kgen := he.KeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	evk := rlwe.NewMemEvaluationKeySet(rlk)

	// Create HE components
	encoder := he.NewEncoder(params)
	encryptor := he.NewEncryptor(params, pk)
	decryptor := he.NewDecryptor(params, sk)
	evaluator := he.NewEvaluator(params, evk)

	// Test data: values in [-1, 1] domain
	input := []float64{-0.8, -0.3, 0.0, 0.5, 0.9}
	fmt.Printf("Input:    %v\n", input)

	// Encrypt → activate → decrypt
	pt := ckks.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(input, pt)
	ct, _ := encryptor.EncryptNew(pt)

	// Apply ReLU approximation
	reluCt, err := activation_he.EvalReLU3(ct, evaluator)
	if err != nil {
		log.Fatal("ReLU evaluation failed:", err)
	}
	reluPt := decryptor.DecryptNew(reluCt)
	reluResult := make([]float64, len(input))
	encoder.Decode(reluPt, reluResult)
	fmt.Printf("ReLU:     %v\n", reluResult[:len(input)])

	// Apply Sigmoid approximation
	sigmoidCt, err := activation_he.EvalSigmoid3(ct, evaluator)
	if err != nil {
		log.Fatal("Sigmoid evaluation failed:", err)
	}
	sigmoidPt := decryptor.DecryptNew(sigmoidCt)
	sigmoidResult := make([]float64, len(input))
	encoder.Decode(sigmoidPt, sigmoidResult)
	fmt.Printf("Sigmoid:  %v\n", sigmoidResult[:len(input)])
}
