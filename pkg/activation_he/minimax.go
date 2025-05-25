package activation_he

import (
	"fmt"
	"math"
)

// MinimaxApproximation computes the minimax polynomial approximation of a function.
// This is a simplified implementation for demonstration purposes.
// In practice, you would use a more sophisticated algorithm like the Remez algorithm
// or leverage libraries like Chebyshev approximation.
//
// Parameters:
//   - f: The function to approximate
//   - interval: The interval [a, b] over which to approximate
//   - degree: The degree of the polynomial approximation
//   - numSamples: The number of sample points to use for the approximation
//
// Returns:
//   - The coefficients of the minimax polynomial approximation
//   - The maximum error of the approximation
func MinimaxApproximation(
	f func(float64) float64,
	interval [2]float64,
	degree int,
	numSamples int,
) (PolynomialCoefficients, float64) {
	// This is a placeholder implementation that returns pre-computed coefficients
	// In a real implementation, you would use the Remez algorithm or a similar method
	
	// For now, we'll return the default coefficients based on the function
	switch {
	case f(0) == 0 && f(1) == 1 && f(-1) == 0: // ReLU-like
		return DefaultReLUConfig.Coefficients, DefaultReLUConfig.MaxError
	case math.Abs(f(0)-0.5) < 0.01 && f(4) > 0.95 && f(-4) < 0.05: // Sigmoid-like
		return DefaultSigmoidConfig.Coefficients, DefaultSigmoidConfig.MaxError
	case math.Abs(f(0)) < 0.01 && f(4) > 0.95 && f(-4) < -0.95: // Tanh-like
		return DefaultTanhConfig.Coefficients, DefaultTanhConfig.MaxError
	default:
		// Return a simple linear approximation as fallback
		a := interval[0]
		b := interval[1]
		fa := f(a)
		fb := f(b)
		slope := (fb - fa) / (b - a)
		intercept := fa - slope*a
		return PolynomialCoefficients{intercept, slope}, math.Max(math.Abs(f((a+b)/2)-(intercept+slope*(a+b)/2)), 
			math.Max(math.Abs(f(a)-(intercept+slope*a)), math.Abs(f(b)-(intercept+slope*b))))
	}
}

// ComputeReLUCoefficients computes the minimax polynomial approximation for ReLU.
// This function uses the MinimaxApproximation function with the ReLU function.
//
// Parameters:
//   - interval: The interval [a, b] over which to approximate
//   - degree: The degree of the polynomial approximation
//   - numSamples: The number of sample points to use for the approximation
//
// Returns:
//   - The coefficients of the minimax polynomial approximation
//   - The maximum error of the approximation
func ComputeReLUCoefficients(
	interval [2]float64,
	degree int,
	numSamples int,
) (PolynomialCoefficients, float64) {
	return MinimaxApproximation(ReLU, interval, degree, numSamples)
}

// ComputeSigmoidCoefficients computes the minimax polynomial approximation for Sigmoid.
// This function uses the MinimaxApproximation function with the Sigmoid function.
//
// Parameters:
//   - interval: The interval [a, b] over which to approximate
//   - degree: The degree of the polynomial approximation
//   - numSamples: The number of sample points to use for the approximation
//
// Returns:
//   - The coefficients of the minimax polynomial approximation
//   - The maximum error of the approximation
func ComputeSigmoidCoefficients(
	interval [2]float64,
	degree int,
	numSamples int,
) (PolynomialCoefficients, float64) {
	return MinimaxApproximation(Sigmoid, interval, degree, numSamples)
}

// ComputeTanhCoefficients computes the minimax polynomial approximation for Tanh.
// This function uses the MinimaxApproximation function with the Tanh function.
//
// Parameters:
//   - interval: The interval [a, b] over which to approximate
//   - degree: The degree of the polynomial approximation
//   - numSamples: The number of sample points to use for the approximation
//
// Returns:
//   - The coefficients of the minimax polynomial approximation
//   - The maximum error of the approximation
func ComputeTanhCoefficients(
	interval [2]float64,
	degree int,
	numSamples int,
) (PolynomialCoefficients, float64) {
	return MinimaxApproximation(Tanh, interval, degree, numSamples)
}

// GetMinimaxCoefficients returns the minimax polynomial coefficients for a specific activation function.
// This is a convenience function that computes or retrieves pre-computed coefficients.
//
// Parameters:
//   - activationType: The type of activation function ("relu", "sigmoid", or "tanh")
//   - config: Optional configuration for the approximation
//
// Returns:
//   - The coefficients of the minimax polynomial approximation
//   - An error if the activation type is invalid
func GetMinimaxCoefficients(
	activationType string,
	config *ActivationConfig,
) (PolynomialCoefficients, error) {
	if config == nil {
		switch activationType {
		case "relu":
			return DefaultReLUConfig.Coefficients, nil
		case "sigmoid":
			return DefaultSigmoidConfig.Coefficients, nil
		case "tanh":
			return DefaultTanhConfig.Coefficients, nil
		default:
			return nil, fmt.Errorf("unknown activation type: %s", activationType)
		}
	}

	// Use the provided configuration to compute the coefficients
	numSamples := 1000 // Default number of samples
	switch activationType {
	case "relu":
		coeffs, _ := ComputeReLUCoefficients(config.Interval, config.Degree, numSamples)
		return coeffs, nil
	case "sigmoid":
		coeffs, _ := ComputeSigmoidCoefficients(config.Interval, config.Degree, numSamples)
		return coeffs, nil
	case "tanh":
		coeffs, _ := ComputeTanhCoefficients(config.Interval, config.Degree, numSamples)
		return coeffs, nil
	default:
		return nil, fmt.Errorf("unknown activation type: %s", activationType)
	}
}
