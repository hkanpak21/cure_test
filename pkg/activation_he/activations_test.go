package activation_he

import (
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

const (
	// Test parameters
	testLogN            = 13    // Ring degree for tests
	testLogDefaultScale = 40    // Scale for tests
	testNumSamples      = 1000  // Number of samples for testing
	testErrorTolerance  = 0.01  // Maximum allowed error for polynomial approximation
	testNoiseThreshold  = 0.05  // Additional error tolerance for HE noise (increased for HE operations)
)

// Test modulus chain configuration
var (
	testLogQ = []int{55, 40, 40, 40, 40} // Multiple ciphertext modulus levels for tests
	testLogP = []int{45, 45}            // Key-switching modulus for tests
)

// setupTestParameters creates CKKS parameters suitable for testing
func setupTestParameters(t *testing.T) ckks.Parameters {
	params, err := ckks.NewParametersFromLiteral(ckks.ParametersLiteral{
		LogN:            testLogN,
		LogQ:            testLogQ, // Multiple levels for operations
		LogP:            testLogP,
		LogDefaultScale: testLogDefaultScale,
	})
	require.NoError(t, err, "Failed to create CKKS parameters")
	return params
}

// setupTestKeys generates keys for testing
func setupTestKeys(t *testing.T, params ckks.Parameters) (*rlwe.SecretKey, *rlwe.PublicKey, *rlwe.RelinearizationKey) {
	kgen := ckks.NewKeyGenerator(params)
	// Generate a new secret key
	sk := kgen.GenSecretKeyNew()
	
	// Generate a new public key
	pk := kgen.GenPublicKeyNew(sk)
	
	// Generate a new relinearization key
	rlk := kgen.GenRelinearizationKeyNew(sk)
	
	return sk, pk, rlk
}

// setupEvaluator is a helper function to create a CKKS evaluator and related components
func setupEvaluator(t *testing.T, params ckks.Parameters) (*ckks.Evaluator, *ckks.Encoder, *rlwe.Encryptor, *rlwe.Decryptor) {
	// Generate keys
	sk, pk, rlk := setupTestKeys(t, params)
	
	// Create encoder, encryptor, decryptor
	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, pk)
	decryptor := ckks.NewDecryptor(params, sk)
	
	// Create evaluation key set
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	
	// Create evaluator
	evaluator := ckks.NewEvaluator(params, evk)
	
	return evaluator, encoder, encryptor, decryptor
}

// generateTestSamples generates uniformly distributed test samples in the given interval
func generateTestSamples(interval [2]float64, numSamples int) []float64 {
	a, b := interval[0], interval[1]
	step := (b - a) / float64(numSamples-1)
	samples := make([]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		samples[i] = a + float64(i)*step
	}
	return samples
}

// TestBasicHEOperations tests simple homomorphic operations to verify Lattigo v6 compatibility
func TestBasicHEOperations(t *testing.T) {
	// Create parameters with extra levels for multiple operations
	params := setupTestParameters(t)
	
	// Create evaluator and other components
	evaluator, encoder, encryptor, decryptor := setupEvaluator(t, params)
	
	// Test simple operations on a single value
	testValue := 3.5
	
	// Encode and encrypt the test value
	values := []float64{testValue}
	plaintext := ckks.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(values, plaintext)
	ciphertext, err := encryptor.EncryptNew(plaintext)
	require.NoError(t, err, "Failed to encrypt test value")
	
	// 1. Test addition: x + 2
	addConst := 2.0
	addPt := ckks.NewPlaintext(params, ciphertext.Level())
	encoder.Encode([]float64{addConst}, addPt)
	
	addResult, err := evaluator.AddNew(ciphertext, addPt)
	require.NoError(t, err, "Failed to add constant")
	
	// Decrypt and verify
	addDecrypted := decryptor.DecryptNew(addResult)
	addDecryptedValues := make([]float64, 1)
	encoder.Decode(addDecrypted, addDecryptedValues)
	
	expectedAdd := testValue + addConst
	assert.InDelta(t, expectedAdd, addDecryptedValues[0], 0.01, 
		"Addition failed: expected %f, got %f", expectedAdd, addDecryptedValues[0])
	t.Logf("Addition test: %f + %f = %f (expected %f)", 
		testValue, addConst, addDecryptedValues[0], expectedAdd)
	
	// 2. Test multiplication: x * 3
	mulConst := 3.0
	mulPt := ckks.NewPlaintext(params, ciphertext.Level())
	encoder.Encode([]float64{mulConst}, mulPt)
	
	mulResult, err := evaluator.MulNew(ciphertext, mulPt)
	require.NoError(t, err, "Failed to multiply by constant")
	
	// Note: In Lattigo v6, relinearization is only needed after multiplying two ciphertexts
	// Since we're multiplying a ciphertext by a plaintext, the degree stays at 1, so no relinearization is needed
	
	// Rescale if needed
	if mulResult.Level() > 0 {
		err = evaluator.Rescale(mulResult, mulResult)
		require.NoError(t, err, "Failed to rescale")
	}
	
	// Decrypt and verify
	mulDecrypted := decryptor.DecryptNew(mulResult)
	mulDecryptedValues := make([]float64, 1)
	encoder.Decode(mulDecrypted, mulDecryptedValues)
	
	expectedMul := testValue * mulConst
	assert.InDelta(t, expectedMul, mulDecryptedValues[0], 0.01, 
		"Multiplication failed: expected %f, got %f", expectedMul, mulDecryptedValues[0])
	t.Logf("Multiplication test: %f * %f = %f (expected %f)", 
		testValue, mulConst, mulDecryptedValues[0], expectedMul)
	
	// 3. Test polynomial: 2x² + 3x + 1 using EvaluatePolynomialHE
	poly := PolynomialCoefficients{1.0, 3.0, 2.0} // 1 + 3x + 2x²
	
	polyResult, err := EvaluatePolynomialHE(poly, ciphertext, evaluator, params)
	require.NoError(t, err, "Failed to evaluate polynomial")
	
	// Decrypt and verify
	polyDecrypted := decryptor.DecryptNew(polyResult)
	polyDecryptedValues := make([]float64, 1)
	encoder.Decode(polyDecrypted, polyDecryptedValues)
	
	// Calculate expected result: 2x² + 3x + 1
	expectedPoly := 2*testValue*testValue + 3*testValue + 1
	assert.InDelta(t, expectedPoly, polyDecryptedValues[0], 0.1, 
		"Polynomial evaluation failed: expected %f, got %f", expectedPoly, polyDecryptedValues[0])
	t.Logf("Polynomial test: 2*(%f)² + 3*(%f) + 1 = %f (expected %f)", 
		testValue, testValue, polyDecryptedValues[0], expectedPoly)
}

// TestPolynomialApproximation tests the polynomial approximation accuracy for each activation function
func TestPolynomialApproximation(t *testing.T) {
	tests := []struct {
		name           string
		actualFunc     func(float64) float64
		approxFunc     func(float64) float64
		config         *ActivationConfig
		errorTolerance float64
	}{
		{
			name:       "ReLU Approximation",
			actualFunc: ReLU,
			approxFunc: func(x float64) float64 { return EvaluatePolynomial(DefaultReLUConfig.Coefficients, x) },
			config:     &DefaultReLUConfig,
			errorTolerance: testErrorTolerance,
		},
		{
			name:       "Sigmoid Approximation",
			actualFunc: Sigmoid,
			approxFunc: func(x float64) float64 { return EvaluatePolynomial(DefaultSigmoidConfig.Coefficients, x) },
			config:     &DefaultSigmoidConfig,
			errorTolerance: testErrorTolerance,
		},
		{
			name:       "Tanh Approximation",
			actualFunc: Tanh,
			approxFunc: func(x float64) float64 { return EvaluatePolynomial(DefaultTanhConfig.Coefficients, x) },
			config:     &DefaultTanhConfig,
			errorTolerance: testErrorTolerance,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			samples := generateTestSamples(tc.config.Interval, testNumSamples)
			maxError := 0.0

			for _, x := range samples {
				expected := tc.actualFunc(x)
				actual := tc.approxFunc(x)
				err := math.Abs(expected - actual)
				if err > maxError {
					maxError = err
				}
			}

			assert.LessOrEqual(t, maxError, tc.errorTolerance, 
				"Maximum approximation error exceeds tolerance")
			t.Logf("Maximum approximation error: %f", maxError)
		})
	}
}

// TestHEActivationFunctions tests the homomorphic evaluation of activation functions
func TestHEActivationFunctions(t *testing.T) {
	params := setupTestParameters(t)
	sk, pk, rlk := setupTestKeys(t, params)

	// Create encoder, encryptor, decryptor, and evaluator
	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, pk)
	decryptor := ckks.NewDecryptor(params, sk)
	
	// Create evaluation key set with the relinearization key
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	
	// Create evaluator with the evaluation key set
	evaluator := ckks.NewEvaluator(params, evk)

	tests := []struct {
		name       string
		// Use our polynomial approximation for the expected values instead of actual function
		polyFunc   func(float64) float64 
		config     *ActivationConfig
		// Define valid test values within the approximation range
		testValues []float64
	}{
		{
			name:       "ReLU HE Evaluation",
			// Use the polynomial approximation as the expected function
			polyFunc: func(x float64) float64 {
				return EvaluatePolynomial(DefaultReLUConfig.Coefficients, x)
			},
			config:     &DefaultReLUConfig,
			testValues: []float64{0.0, 0.5, 1.0, 2.0, 3.0, 4.0},
		},
		{
			name:       "Sigmoid HE Evaluation",
			// Use the polynomial approximation as the expected function
			polyFunc: func(x float64) float64 {
				return EvaluatePolynomial(DefaultSigmoidConfig.Coefficients, x)
			},
			config:     &DefaultSigmoidConfig,
			testValues: []float64{-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0},
		},
		{
			name:       "Tanh HE Evaluation",
			// Use the polynomial approximation as the expected function
			polyFunc: func(x float64) float64 {
				return EvaluatePolynomial(DefaultTanhConfig.Coefficients, x)
			},
			config:     &DefaultTanhConfig,
			testValues: []float64{-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Use the predefined test values instead of generating samples
			for _, x := range tc.testValues {
				// Encode the input
				values := []float64{x}
				plaintext := ckks.NewPlaintext(params, params.MaxLevel())
				encoder.Encode(values, plaintext)
				
				// Encrypt the plaintext
				ciphertext, err := encryptor.EncryptNew(plaintext)
				require.NoError(t, err, "Failed to encrypt plaintext")

				// Evaluate the activation function homomorphically based on the test case
				var resultCt *rlwe.Ciphertext
				switch tc.name {
				case "ReLU HE Evaluation":
					resultCt, err = EvalReLUHE(ciphertext, evaluator, params, tc.config)
				case "Sigmoid HE Evaluation":
					resultCt, err = EvalSigmoidHE(ciphertext, evaluator, params, tc.config)
				case "Tanh HE Evaluation":
					resultCt, err = EvalTanhHE(ciphertext, evaluator, params, tc.config)
				}
				require.NoError(t, err, "Failed to evaluate activation function")

				// Decrypt the result
				resultPlaintext := decryptor.DecryptNew(resultCt)
				
				// Decode the result - in Lattigo v6, we need to create the slice first with appropriate size
				resultValues := make([]float64, 1)
				encoder.Decode(resultPlaintext, resultValues)

				// Compare with the expected result from polynomial approximation
				expected := tc.polyFunc(x)
				actual := resultValues[0]
				
				// Account for both approximation error and HE noise
				totalErrorTolerance := testErrorTolerance + testNoiseThreshold
				
				assert.InDelta(t, expected, actual, totalErrorTolerance,
					"HE evaluation result differs from expected value by more than allowed tolerance")
				
				t.Logf("Input: %f, Expected: %f, Actual: %f, Error: %f",
					x, expected, actual, math.Abs(expected-actual))
			}
		})
	}
}

// TestActivationPipeline tests a simple pipeline using the activation functions
func TestActivationPipeline(t *testing.T) {
	// Skip this test in short mode as it's more of an integration test
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	params := setupTestParameters(t)
	sk, pk, rlk := setupTestKeys(t, params)

	// Create encoder, encryptor, decryptor, and evaluator
	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, pk)
	decryptor := ckks.NewDecryptor(params, sk)
	
	// Create evaluation key set with the relinearization key
	evk := rlwe.NewMemEvaluationKeySet(rlk)
	
	// Create evaluator with the evaluation key set
	evaluator := ckks.NewEvaluator(params, evk)

	// Define a simple neural network layer with ReLU activation
	// y = ReLU(w*x + b)
	weight := 2.0
	bias := -1.0
	
	// Choose input values such that the linear outputs fall within the valid range for ReLU [0, 4]
	// For ReLU, with weight=2.0 and bias=-1.0, inputs between 0.5 and 2.5 give outputs in [0, 4]
	inputs := []float64{0.5, 1.0, 1.5, 2.0, 2.5}
	
	for _, x := range inputs {
		// Compute expected output in plaintext
		linearOutput := weight*x + bias
		expected := ReLU(linearOutput)
		
		// Encode the input
		values := []float64{x}
		plaintext := ckks.NewPlaintext(params, params.MaxLevel())
		encoder.Encode(values, plaintext)
		
		// Encrypt the plaintext
		ciphertext, err := encryptor.EncryptNew(plaintext)
		require.NoError(t, err, "Failed to encrypt plaintext")
		
		// Perform linear transformation: w*x + b
		weightedCt, err := evaluator.MulNew(ciphertext, weight)
		require.NoError(t, err, "Failed to multiply by weight")
		
		// Create a plaintext with the bias value
		biasPt := ckks.NewPlaintext(params, weightedCt.Level())
		biasValues := []float64{bias}
		encoder.Encode(biasValues, biasPt)
		
		// Add the bias to the weighted ciphertext
		err = evaluator.Add(weightedCt, biasPt, weightedCt)
		require.NoError(t, err, "Failed to add bias")
		
		// Apply ReLU activation
		activatedCt, err := EvalReLUHE(weightedCt, evaluator, params, nil)
		require.NoError(t, err, "Failed to apply ReLU activation")
		
		// Decrypt the result
		resultPlaintext := decryptor.DecryptNew(activatedCt)
		
		// Decode the result - in Lattigo v6, we need to create the slice first with appropriate size
		resultValues := make([]float64, 1)
		encoder.Decode(resultPlaintext, resultValues)
		
		// Compare with expected result
		actual := resultValues[0]
		totalErrorTolerance := testErrorTolerance + testNoiseThreshold
		
		assert.InDelta(t, expected, actual, totalErrorTolerance,
			fmt.Sprintf("Pipeline result differs from expected for input %f", x))
		
		t.Logf("Input: %f, Linear: %f, Expected: %f, Actual: %f, Error: %f",
			x, linearOutput, expected, actual, math.Abs(expected-actual))
	}
}
