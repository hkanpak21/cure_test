package matrix

import (
	"fmt"
)

// Multiply performs matrix multiplication C = A * B.
// A is an m x k matrix, B is a k x n matrix. Result C is an m x n matrix.
// It handles matrices with zero dimensions according to mathematical conventions
// for [][]float64 representations (e.g., a 0-row matrix is 0x0, an M-row matrix with empty inner slices is Mx0).
func Multiply(a, b [][]float64) ([][]float64, error) {
	rowsA := len(a)
	colsA := 0
	if rowsA > 0 {
		colsA = len(a[0]) // Assumes a[0] exists and is representative for a rectangular matrix
	}

	rowsB := len(b)
	colsB := 0
	if rowsB > 0 {
		colsB = len(b[0]) // Assumes b[0] exists and is representative for a rectangular matrix
	}

	if colsA != rowsB {
		return nil, fmt.Errorf("matrix: incompatible dimensions for multiplication, A_cols(%d) != B_rows(%d)", colsA, rowsB)
	}

	c := make([][]float64, rowsA)
	for i := 0; i < rowsA; i++ {
		c[i] = make([]float64, colsB) // If colsB is 0, this creates an empty slice, which is correct for an Mx0 matrix.
		for j := 0; j < colsB; j++ {
			sum := 0.0
			// If colsA (inner dimension) is 0, this loop doesn't run, sum remains 0.
			// This correctly produces a zero matrix for cases like (Mx0) * (0xN) -> (MxN of zeros).
			for k := 0; k < colsA; k++ {
				sum += a[i][k] * b[k][j]
			}
			c[i][j] = sum
		}
	}
	return c, nil
}

// Add performs element-wise addition C = A + B.
// Matrices A and B must have identical dimensions.
func Add(a, b [][]float64) ([][]float64, error) {
	rowsA := len(a)
	colsA := 0
	if rowsA > 0 {
		// Assumes a[0] exists and is representative for a rectangular matrix
		colsA = len(a[0])
	}

	rowsB := len(b)
	colsB := 0
	if rowsB > 0 {
		// Assumes b[0] exists and is representative for a rectangular matrix
		colsB = len(b[0])
	}

	if rowsA != rowsB || colsA != colsB {
		return nil, fmt.Errorf("matrix: dimensions must be identical for addition, A is %dx%d, B is %dx%d", rowsA, colsA, rowsB, colsB)
	}

	c := make([][]float64, rowsA)
	for i := 0; i < rowsA; i++ {
		c[i] = make([]float64, colsA)
		for j := 0; j < colsA; j++ {
			c[i][j] = a[i][j] + b[i][j]
		}
	}
	return c, nil
}

// Subtract performs element-wise subtraction C = A - B.
// Matrices A and B must have identical dimensions.
func Subtract(a, b [][]float64) ([][]float64, error) {
	rowsA := len(a)
	colsA := 0
	if rowsA > 0 {
		// Assumes a[0] exists and is representative for a rectangular matrix
		colsA = len(a[0])
	}

	rowsB := len(b)
	colsB := 0
	if rowsB > 0 {
		// Assumes b[0] exists and is representative for a rectangular matrix
		colsB = len(b[0])
	}

	if rowsA != rowsB || colsA != colsB {
		return nil, fmt.Errorf("matrix: dimensions must be identical for subtraction, A is %dx%d, B is %dx%d", rowsA, colsA, rowsB, colsB)
	}

	c := make([][]float64, rowsA)
	for i := 0; i < rowsA; i++ {
		c[i] = make([]float64, colsA)
		for j := 0; j < colsA; j++ {
			c[i][j] = a[i][j] - b[i][j]
		}
	}
	return c, nil
}
