package activation_he

import (
	"fmt"
	"math"
)

// ComputeApproximationError calculates the error between a function and its polynomial approximation
// at a given point.
//
// Parameters:
//   - f: The original function
//   - coeffs: The polynomial coefficients for the approximation
//   - x: The point at which to calculate the error
//
// Returns:
//   - The absolute error |f(x) - p(x)|
func ComputeApproximationError(f func(float64) float64, coeffs PolynomialCoefficients, x float64) float64 {
	return math.Abs(f(x) - EvaluatePolynomial(coeffs, x))
}

// ComputeMaxApproximationError calculates the maximum approximation error over a given interval.
//
// Parameters:
//   - f: The original function
//   - coeffs: The polynomial coefficients for the approximation
//   - interval: The interval [a, b] over which to calculate the error
//   - numSamples: The number of sample points to use
//
// Returns:
//   - The maximum absolute error over the interval
//   - The point at which the maximum error occurs
func ComputeMaxApproximationError(
	f func(float64) float64,
	coeffs PolynomialCoefficients,
	interval [2]float64,
	numSamples int,
) (float64, float64) {
	a, b := interval[0], interval[1]
	step := (b - a) / float64(numSamples-1)
	
	maxError := 0.0
	maxErrorPoint := a
	
	for i := 0; i < numSamples; i++ {
		x := a + float64(i)*step
		error := ComputeApproximationError(f, coeffs, x)
		
		if error > maxError {
			maxError = error
			maxErrorPoint = x
		}
	}
	
	return maxError, maxErrorPoint
}

// GenerateErrorSamples generates samples of the approximation error over an interval.
// This is useful for plotting the error distribution.
//
// Parameters:
//   - f: The original function
//   - coeffs: The polynomial coefficients for the approximation
//   - interval: The interval [a, b] over which to generate samples
//   - numSamples: The number of sample points to generate
//
// Returns:
//   - A slice of x values
//   - A slice of corresponding error values
func GenerateErrorSamples(
	f func(float64) float64,
	coeffs PolynomialCoefficients,
	interval [2]float64,
	numSamples int,
) ([]float64, []float64) {
	a, b := interval[0], interval[1]
	step := (b - a) / float64(numSamples-1)
	
	xValues := make([]float64, numSamples)
	errorValues := make([]float64, numSamples)
	
	for i := 0; i < numSamples; i++ {
		x := a + float64(i)*step
		xValues[i] = x
		errorValues[i] = ComputeApproximationError(f, coeffs, x)
	}
	
	return xValues, errorValues
}

// GenerateFunctionSamples generates samples of both the original function and its
// polynomial approximation over an interval.
//
// Parameters:
//   - f: The original function
//   - coeffs: The polynomial coefficients for the approximation
//   - interval: The interval [a, b] over which to generate samples
//   - numSamples: The number of sample points to generate
//
// Returns:
//   - A slice of x values
//   - A slice of corresponding original function values
//   - A slice of corresponding approximation values
func GenerateFunctionSamples(
	f func(float64) float64,
	coeffs PolynomialCoefficients,
	interval [2]float64,
	numSamples int,
) ([]float64, []float64, []float64) {
	a, b := interval[0], interval[1]
	step := (b - a) / float64(numSamples-1)
	
	xValues := make([]float64, numSamples)
	fValues := make([]float64, numSamples)
	pValues := make([]float64, numSamples)
	
	for i := 0; i < numSamples; i++ {
		x := a + float64(i)*step
		xValues[i] = x
		fValues[i] = f(x)
		pValues[i] = EvaluatePolynomial(coeffs, x)
	}
	
	return xValues, fValues, pValues
}

// PrintPolynomialInfo prints information about a polynomial approximation.
//
// Parameters:
//   - name: The name of the activation function
//   - coeffs: The polynomial coefficients
//   - interval: The interval over which the approximation is valid
//   - maxError: The maximum approximation error
func PrintPolynomialInfo(name string, coeffs PolynomialCoefficients, interval [2]float64, maxError float64) {
	fmt.Printf("=== %s Polynomial Approximation ===\n", name)
	fmt.Printf("Degree: %d\n", len(coeffs)-1)
	fmt.Printf("Interval: [%.2f, %.2f]\n", interval[0], interval[1])
	fmt.Printf("Max Error: %.6f\n", maxError)
	fmt.Println("Coefficients:")
	
	for i, coeff := range coeffs {
		fmt.Printf("  a%d = %.10f\n", i, coeff)
	}
	
	fmt.Println()
}

// FormatPolynomialString formats a polynomial as a human-readable string.
//
// Parameters:
//   - coeffs: The polynomial coefficients
//   - precision: The number of decimal places to show
//
// Returns:
//   - A formatted string representation of the polynomial
func FormatPolynomialString(coeffs PolynomialCoefficients, precision int) string {
	if len(coeffs) == 0 {
		return "0"
	}
	
	format := fmt.Sprintf("%%.%df", precision)
	result := ""
	
	for i, coeff := range coeffs {
		if math.Abs(coeff) < 1e-10 {
			continue // Skip terms with zero coefficients
		}
		
		term := ""
		if i == 0 {
			// Constant term
			term = fmt.Sprintf(format, coeff)
		} else if i == 1 {
			// Linear term
			if coeff == 1 {
				term = "x"
			} else if coeff == -1 {
				term = "-x"
			} else {
				term = fmt.Sprintf(format+"x", coeff)
			}
		} else {
			// Higher-degree terms
			if coeff == 1 {
				term = fmt.Sprintf("x^%d", i)
			} else if coeff == -1 {
				term = fmt.Sprintf("-x^%d", i)
			} else {
				term = fmt.Sprintf(format+"x^%d", coeff, i)
			}
		}
		
		// Add the term to the result with the appropriate sign
		if result == "" {
			// First term
			result = term
		} else if coeff > 0 {
			// Positive term
			result += " + " + term
		} else {
			// Negative term (the sign is already included in the term)
			result += " " + term
		}
	}
	
	return result
}
