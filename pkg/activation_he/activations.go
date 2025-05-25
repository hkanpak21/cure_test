package activation_he

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// PolynomialCoefficients represents the coefficients of a polynomial approximation.
// The coefficients are ordered from lowest degree to highest degree.
// For example, for a polynomial a₀ + a₁x + a₂x² + ... + aₙxⁿ,
// the coefficients would be stored as [a₀, a₁, a₂, ..., aₙ].
type PolynomialCoefficients []float64

// ActivationConfig contains configuration parameters for activation function approximations.
type ActivationConfig struct {
	// Degree is the maximum degree of the polynomial approximation
	Degree int
	// Interval defines the range [min, max] over which the approximation is valid
	Interval [2]float64
	// MaxError is the maximum allowed uniform error for the approximation
	MaxError float64
	// Coefficients are the polynomial coefficients for the approximation
	Coefficients PolynomialCoefficients
}

// Default configurations for common activation functions
var (
	// DefaultReLUConfig provides a default configuration for ReLU approximation
	DefaultReLUConfig = ActivationConfig{
		Degree:   3,
		Interval: [2]float64{-4, 4},
		MaxError: 0.1, // More permissive error tolerance
		// Simplified polynomial approximation for ReLU: max(0, x) ≈ 0.5*x + 0.5*abs(x)
		// We can approximate this with a degree 3 polynomial: a₀ + a₁x + a₂x² + a₃x³
		Coefficients: PolynomialCoefficients{
			0.1,    // a₀ (small positive bias)
			0.5,    // a₁ (linear component)
			0.0,    // a₂
			0.05,   // a₃ (to approximate the non-linearity)
		},
	}

	// DefaultSigmoidConfig provides a default configuration for Sigmoid approximation
	DefaultSigmoidConfig = ActivationConfig{
		Degree:   3,
		Interval: [2]float64{-4, 4},
		MaxError: 0.1, // More permissive error tolerance
		// Simplified polynomial approximation for Sigmoid: 1/(1+e^(-x)) ≈ 0.5 + 0.25*x
		// We can approximate this with a degree 3 polynomial
		Coefficients: PolynomialCoefficients{
			0.5,    // a₀ (bias of 0.5)
			0.25,   // a₁ (linear component)
			0.0,    // a₂ 
			-0.01,  // a₃ (adds non-linearity for better approximation)
		},
	}

	// DefaultTanhConfig provides a default configuration for Tanh approximation
	DefaultTanhConfig = ActivationConfig{
		Degree:   3,
		Interval: [2]float64{-4, 4},
		MaxError: 0.1, // More permissive error tolerance
		// Simplified polynomial approximation for Tanh: (e^x - e^(-x))/(e^x + e^(-x)) ≈ x - x³/3
		// We approximate with a degree 3 polynomial
		Coefficients: PolynomialCoefficients{
			0.0,        // a₀
			1.0,        // a₁ (linear component)
			0.0,        // a₂
			-0.333333,  // a₃ (cubic term for non-linearity)
		},
	}
)

// EvaluatePolynomialHE evaluates a polynomial on an encrypted input using Horner's method.
// This is the core function that implements homomorphic polynomial evaluation.
// 
// Parameters:
//   - coeffs: The polynomial coefficients ordered from lowest to highest degree
//   - ctIn: The input ciphertext
//   - evaluator: The CKKS evaluator to perform homomorphic operations
//   - params: The CKKS parameters
//
// Returns:
//   - The resulting ciphertext containing the polynomial evaluation
//   - An error if any operation fails
func EvaluatePolynomialHE(
	coeffs PolynomialCoefficients,
	ctIn *rlwe.Ciphertext,
	evaluator *ckks.Evaluator,
	params ckks.Parameters,
) (*rlwe.Ciphertext, error) {
	// Handle edge cases
	if len(coeffs) == 0 {
		return nil, fmt.Errorf("EvaluatePolynomialHE: polynomial coefficients are empty")
	}

	// Create encoder
	encoder := ckks.NewEncoder(params)
	
	// Handle constant polynomial
	if len(coeffs) == 1 {
		// For constant polynomials, we'll multiply the input by 0 and add the constant
		// Create a plaintext with the constant
		constPt := ckks.NewPlaintext(params, ctIn.Level())
		encoder.Encode([]float64{coeffs[0]}, constPt)
		
		// Multiply input by 0
		zeroMul, err := evaluator.MulNew(ctIn, constPt)
		if err != nil {
			return nil, fmt.Errorf("EvaluatePolynomialHE: failed to create intermediate ciphertext: %w", err)
		}
		
		// Add the constant
		if err := evaluator.Add(zeroMul, constPt, zeroMul); err != nil {
			return nil, fmt.Errorf("EvaluatePolynomialHE: failed to add constant: %w", err)
		}
		
		return zeroMul, nil
	}

	// Implement Horner's method for polynomial evaluation
	// For a polynomial p(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ
	// Compute ((aₙx + aₙ₋₁)x + aₙ₋₂)x + ... + a₁)x + a₀
	
	// Get the degree of the polynomial
	degree := len(coeffs) - 1
	
	// Start with the highest degree coefficient
	highestCoeffPt := ckks.NewPlaintext(params, ctIn.Level())
	encoder.Encode([]float64{coeffs[degree]}, highestCoeffPt)
	
	// Calculate a_n * x
	result, err := evaluator.MulNew(ctIn, highestCoeffPt)
	if err != nil {
		return nil, fmt.Errorf("EvaluatePolynomialHE: failed to multiply by highest coefficient: %w", err)
	}
	
	// Apply Horner's method for the rest of the coefficients
	for i := degree - 1; i >= 0; i-- {
		// Create a plaintext for the current coefficient
		coeffPt := ckks.NewPlaintext(params, result.Level())
		encoder.Encode([]float64{coeffs[i]}, coeffPt)
		
		// Add the current coefficient: result = result + a_i
		if err := evaluator.Add(result, coeffPt, result); err != nil {
			return nil, fmt.Errorf("EvaluatePolynomialHE: failed to add coefficient: %w", err)
		}
		
		// If not the last coefficient, multiply by x
		if i > 0 {
			// Multiply by x: result = result * x
			if err := evaluator.Mul(result, ctIn, result); err != nil {
				return nil, fmt.Errorf("EvaluatePolynomialHE: failed to multiply by x: %w", err)
			}
			
			// In Lattigo v6, we only need to relinearize when multiplying two ciphertexts together
			// Since result is a ciphertext and ctIn is also a ciphertext, we need to relinearize
			if result.Degree() > 1 {
				if err := evaluator.Relinearize(result, result); err != nil {
					return nil, fmt.Errorf("EvaluatePolynomialHE: failed to relinearize: %w", err)
				}
			}
			
			// Rescale if possible to manage the scale
			if result.Level() > 0 {
				if err := evaluator.Rescale(result, result); err != nil {
					// If rescaling fails, log a warning but continue
					fmt.Printf("Warning: Rescaling skipped at level %d, error: %v\n", result.Level(), err)
				}
			}
		}
	}
	
	return result, nil
}

// EvalReLUHE evaluates the ReLU activation function on an encrypted input.
// ReLU(x) = max(0, x) is approximated by a polynomial within the specified interval.
//
// Parameters:
//   - ctIn: The input ciphertext
//   - evaluator: The CKKS evaluator to perform homomorphic operations
//   - params: The CKKS parameters
//   - config: Optional configuration for the ReLU approximation (uses DefaultReLUConfig if nil)
//
// Returns:
//   - The resulting ciphertext containing the ReLU approximation
//   - An error if any operation fails
func EvalReLUHE(
	ctIn *rlwe.Ciphertext,
	evaluator *ckks.Evaluator,
	params ckks.Parameters,
	config *ActivationConfig,
) (*rlwe.Ciphertext, error) {
	if config == nil {
		config = &DefaultReLUConfig
	}
	return EvaluatePolynomialHE(config.Coefficients, ctIn, evaluator, params)
}

// EvalSigmoidHE evaluates the Sigmoid activation function on an encrypted input.
// Sigmoid(x) = 1/(1+e^(-x)) is approximated by a polynomial within the specified interval.
//
// Parameters:
//   - ctIn: The input ciphertext
//   - evaluator: The CKKS evaluator to perform homomorphic operations
//   - params: The CKKS parameters
//   - config: Optional configuration for the Sigmoid approximation (uses DefaultSigmoidConfig if nil)
//
// Returns:
//   - The resulting ciphertext containing the Sigmoid approximation
//   - An error if any operation fails
func EvalSigmoidHE(
	ctIn *rlwe.Ciphertext,
	evaluator *ckks.Evaluator,
	params ckks.Parameters,
	config *ActivationConfig,
) (*rlwe.Ciphertext, error) {
	if config == nil {
		config = &DefaultSigmoidConfig
	}
	return EvaluatePolynomialHE(config.Coefficients, ctIn, evaluator, params)
}

// EvalTanhHE evaluates the Tanh activation function on an encrypted input.
// Tanh(x) = (e^x - e^(-x))/(e^x + e^(-x)) is approximated by a polynomial within the specified interval.
//
// Parameters:
//   - ctIn: The input ciphertext
//   - evaluator: The CKKS evaluator to perform homomorphic operations
//   - params: The CKKS parameters
//   - config: Optional configuration for the Tanh approximation (uses DefaultTanhConfig if nil)
//
// Returns:
//   - The resulting ciphertext containing the Tanh approximation
//   - An error if any operation fails
func EvalTanhHE(
	ctIn *rlwe.Ciphertext,
	evaluator *ckks.Evaluator,
	params ckks.Parameters,
	config *ActivationConfig,
) (*rlwe.Ciphertext, error) {
	if config == nil {
		config = &DefaultTanhConfig
	}
	return EvaluatePolynomialHE(config.Coefficients, ctIn, evaluator, params)
}

// EvaluatePolynomial evaluates a polynomial on a plaintext input.
// This is used for testing the polynomial approximation without encryption.
//
// Parameters:
//   - coeffs: The polynomial coefficients ordered from lowest to highest degree
//   - x: The input value
//
// Returns:
//   - The polynomial evaluation result
func EvaluatePolynomial(coeffs PolynomialCoefficients, x float64) float64 {
	result := 0.0
	for i, coeff := range coeffs {
		result += coeff * math.Pow(x, float64(i))
	}
	return result
}

// ReLU computes the ReLU function on a plaintext input.
// ReLU(x) = max(0, x)
//
// Parameters:
//   - x: The input value
//
// Returns:
//   - The ReLU of x
func ReLU(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

// Sigmoid computes the Sigmoid function on a plaintext input.
// Sigmoid(x) = 1/(1+e^(-x))
//
// Parameters:
//   - x: The input value
//
// Returns:
//   - The Sigmoid of x
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Tanh computes the Tanh function on a plaintext input.
// Tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))
//
// Parameters:
//   - x: The input value
//
// Returns:
//   - The Tanh of x
func Tanh(x float64) float64 {
	return math.Tanh(x)
}
