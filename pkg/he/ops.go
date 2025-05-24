package he

import (
	"fmt"
	"sync"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// ScalarMultCiphertext performs scalar multiplication on an encrypted vector (ciphertext).
// It computes v * Enc(W) = Enc(v * W), where v is a plaintext scalar and W is a vector.
// The operation is performed as ctOut = evaluator.MultByConstNew(ctIn, v).
//
// Parameters:
//   - scalar: The plaintext float64 scalar v.
//   - ctIn: The input *rlwe.Ciphertext, representing Enc(W).
//   - evaluator: The *ckks.Evaluator to perform the multiplication.
//
// Returns:
//   - *rlwe.Ciphertext: The resulting ciphertext Enc(v * W).
//   - error: An error if the operation fails (e.g., nil inputs).
func ScalarMultCiphertext(scalar float64, ctIn *rlwe.Ciphertext, evaluator *ckks.Evaluator) (ctOut *rlwe.Ciphertext, err error) {
	if evaluator == nil {
		return nil, fmt.Errorf("ScalarMultCiphertext: evaluator cannot be nil")
	}
	if ctIn == nil {
		return nil, fmt.Errorf("ScalarMultCiphertext: input ciphertext cannot be nil")
	}

	// MulNew creates a new ciphertext for the result.
	// The scalar argument for MulNew can be various types, including float64.
	ctOut, err = evaluator.MulNew(ctIn, scalar)
	if err != nil {
		return nil, fmt.Errorf("ScalarMultCiphertext: MulNew failed: %w", err)
	}

	return ctOut, nil
}

// MulCiphertexts performs homomorphic multiplication of two ciphertexts ct1 and ct2.
// It computes Enc(W1 * W2) = Mul(Enc(W1), Enc(W2)), where W1 and W2 are vectors.
// The operation involves:
//  1. Multiplication: ctOut = evaluator.Mul(ct1, ct2)
//  2. Relinearization: evaluator.Relinearize(ctOut, ctOut) (requires RelinearizationKey in evaluator)
//  3. Rescaling: evaluator.Rescale(ctOut, ctOut)
//
// Parameters:
//   - ct1: The first input *rlwe.Ciphertext, representing Enc(W1).
//   - ct2: The second input *rlwe.Ciphertext, representing Enc(W2).
//   - evaluator: The *ckks.Evaluator to perform the operations. Must be initialized with a RelinearizationKey.
//
// Returns:
//   - *rlwe.Ciphertext: The resulting ciphertext Enc(W1 * W2).
//   - error: An error if any operation fails.
func MulCiphertexts(ct1, ct2 *rlwe.Ciphertext, evaluator *ckks.Evaluator) (ctOut *rlwe.Ciphertext, err error) {
	if evaluator == nil {
		return nil, fmt.Errorf("MulCiphertexts: evaluator cannot be nil")
	}
	if ct1 == nil || ct2 == nil {
		return nil, fmt.Errorf("MulCiphertexts: input ciphertexts cannot be nil")
	}

	// Create a new ciphertext to store the multiplication result.
	// The Mul operation can be done in-place if ctOut is one of the inputs,
	// but creating a new one is safer and clearer.
	// Initialize ctOut with the correct parameters, degree 2, and level.
	// The level should match the input ciphertexts' level.
	// Note: evaluator.GetRLWEParameters() is not a direct method. We get params from one of the inputs or pre-setup.
	// Assuming ct1.GetParameters() or similar exists or params are passed if needed.
	// For now, let's assume params can be derived or are implicitly handled by evaluator methods for ctOut allocation.
	// Lattigo's Mul typically handles ctOut allocation or expects it to be pre-allocated correctly.
	// Let's try evaluator.MulNew(ct1, ct2) which returns a new ciphertext.

	// 1. Multiplication (ctOut will be degree 2)
	ctOut, err = evaluator.MulNew(ct1, ct2)
	if err != nil {
		return nil, fmt.Errorf("MulCiphertexts: MulNew failed: %w", err)
	}

	// 2. Relinearization (ctOut degree 2 -> 1)
	// This step requires a RelinearizationKey to be available in the evaluator.
	if err = evaluator.Relinearize(ctOut, ctOut); err != nil {
		return nil, fmt.Errorf("MulCiphertexts: Relinearize failed: %w", err)
	}

	// 3. Rescaling (adjusts scale and reduces noise)
	if err = evaluator.Rescale(ctOut, ctOut); err != nil {
		return nil, fmt.Errorf("MulCiphertexts: Rescale failed: %w", err)
	}

	return ctOut, nil
}

// MulMatricesCiphertexts performs homomorphic matrix multiplication of A and B.
// A is represented by its encrypted rows (ctaRows), and B by its encrypted columns (ctbCols).
// C_ij = DotProduct(Row_i_A, Col_j_B).
//
// Parameters:
//   - ctaRows: A slice of *rlwe.Ciphertext, where each ciphertext encrypts a row of matrix A.
//     If A is m x k, len(ctaRows) is m.
//   - ctbCols: A slice of *rlwe.Ciphertext, where each ciphertext encrypts a column of matrix B.
//     If B is k x n, len(ctbCols) is n.
//   - k_dimension: The common dimension (number of columns in A / number of rows in B).
//     This is the number of elements in each row/column vector that are multiplied and summed.
//     It must be <= params.MaxSlots().
//   - evaluator: The *ckks.Evaluator to perform the operations.
//     It MUST be initialized with appropriate rotation keys for the inner sum operations.
//
// Returns:
//   - [][]*rlwe.Ciphertext: A 2D slice representing the encrypted result matrix C (m x n).
//   - error: An error if any operation fails or inputs are invalid.
func MulMatricesCiphertexts(
	ctaRows []*rlwe.Ciphertext,
	ctbCols []*rlwe.Ciphertext,
	k_dimension int,
	evaluator *ckks.Evaluator,
) ([][]*rlwe.Ciphertext, error) {

	if evaluator == nil {
		return nil, fmt.Errorf("MulMatricesCiphertexts: evaluator cannot be nil")
	}
	if len(ctaRows) == 0 {
		// If A has 0 rows, the result C also has 0 rows.
		return [][]*rlwe.Ciphertext{}, nil
	}
	if len(ctbCols) == 0 {
		// If B has 0 columns, the result C also has 0 columns.
		m := len(ctaRows)
		resultCiphertexts := make([][]*rlwe.Ciphertext, m)
		for i := 0; i < m; i++ {
			resultCiphertexts[i] = []*rlwe.Ciphertext{}
		}
		return resultCiphertexts, nil
	}

	m := len(ctaRows)
	n := len(ctbCols)

	if k_dimension <= 0 {
		return nil, fmt.Errorf("MulMatricesCiphertexts: k_dimension must be positive, got %d", k_dimension)
	}

	resultCiphertexts := make([][]*rlwe.Ciphertext, m)
	for i := 0; i < m; i++ {
		resultCiphertexts[i] = make([]*rlwe.Ciphertext, n)
		if ctaRows[i] == nil {
			return nil, fmt.Errorf("MulMatricesCiphertexts: ctaRows[%d] cannot be nil", i)
		}
		for j := 0; j < n; j++ {
			if ctbCols[j] == nil {
				return nil, fmt.Errorf("MulMatricesCiphertexts: ctbCols[%d] cannot be nil", j)
			}

			// Step 1: Multiply the row vector from A with the column vector from B element-wise
			// This creates a ciphertext containing the products of corresponding elements
			mulCt, err := evaluator.MulNew(ctaRows[i], ctbCols[j])
			if err != nil {
				return nil, fmt.Errorf("MulMatricesCiphertexts: MulNew failed for C[%d][%d]: %w", i, j, err)
			}

			// Step 2: Relinearize the product ciphertext to reduce its degree back to 1
			if err = evaluator.Relinearize(mulCt, mulCt); err != nil {
				return nil, fmt.Errorf("MulMatricesCiphertexts: Relinearize failed for C[%d][%d]: %w", i, j, err)
			}

			// Step 3: Rescale to manage the noise and scale
			if err = evaluator.Rescale(mulCt, mulCt); err != nil {
				return nil, fmt.Errorf("MulMatricesCiphertexts: Rescale failed for C[%d][%d]: %w", i, j, err)
			}

			// Step 4: Compute the sum of the first k_dimension elements using a more efficient approach
			// We'll use a divide-and-conquer strategy with powers of 2 rotations
			// This is much more efficient for large dimensions
			resultCt := mulCt.CopyNew()
			
			// Create a temporary ciphertext for rotations
			tempCt := mulCt.CopyNew()
			
			// Calculate log2(k_dimension) to determine the number of rotation steps needed
			logK := 0
			for k := k_dimension; k > 1; k >>= 1 {
				logK++
			}
			
			// Use a divide-and-conquer approach with powers of 2 rotations
			// This reduces the number of rotations from O(k_dimension) to O(log(k_dimension))
			for rotStep := 0; rotStep < logK; rotStep++ {
				// Rotate by 2^rotStep positions
				rotation := 1 << rotStep
				if err = evaluator.Rotate(resultCt, rotation, tempCt); err != nil {
					return nil, fmt.Errorf("MulMatricesCiphertexts: Rotate failed for C[%d][%d], rotation %d: %w", i, j, rotation, err)
				}
				
				// Add the rotated ciphertext to the result
				if err = evaluator.Add(resultCt, tempCt, resultCt); err != nil {
					return nil, fmt.Errorf("MulMatricesCiphertexts: Add failed for C[%d][%d], rotation %d: %w", i, j, rotation, err)
				}
			}
			
			resultCiphertexts[i][j] = resultCt
		}
	}

	return resultCiphertexts, nil
}

// MulMatricesCiphertextsParallel performs homomorphic matrix multiplication of two ciphertext matrices in parallel.
// The input matrices are represented as slices of ciphertexts, where each ciphertext in ctaRows
// represents a row of matrix A, and each ciphertext in ctbCols represents a column of matrix B.
// The result is a matrix C where each element C[i][j] is the dot product of row i from A and column j from B.
// This function uses multiple worker goroutines to process rows in parallel.
//
// Parameters:
//   - ctaRows: A slice of *rlwe.Ciphertext, where each ciphertext encrypts a row of matrix A.
//     If A is m x k, len(ctaRows) is m.
//   - ctbCols: A slice of *rlwe.Ciphertext, where each ciphertext encrypts a column of matrix B.
//     If B is k x n, len(ctbCols) is n.
//   - k_dimension: The common dimension (number of columns in A / number of rows in B).
//     This is the number of elements in each row/column vector that are multiplied and summed.
//     It must be <= params.MaxSlots().
//   - evaluator: The *ckks.Evaluator to perform the operations.
//     It MUST be initialized with appropriate rotation keys for the inner sum operations.
//   - numWorkers: The number of parallel workers to use. If <= 0, it will default to 1.
//
// Returns:
//   - [][]*rlwe.Ciphertext: A 2D slice representing the encrypted result matrix C (m x n).
//   - error: An error if any operation fails or inputs are invalid.
func MulMatricesCiphertextsParallel(
	ctaRows []*rlwe.Ciphertext,
	ctbCols []*rlwe.Ciphertext,
	k_dimension int,
	evaluator *ckks.Evaluator,
	numWorkers int,
) ([][]*rlwe.Ciphertext, error) {
	if evaluator == nil {
		return nil, fmt.Errorf("MulMatricesCiphertextsParallel: evaluator cannot be nil")
	}

	if ctaRows == nil || ctbCols == nil {
		return nil, fmt.Errorf("MulMatricesCiphertextsParallel: input matrices cannot be nil")
	}

	if len(ctaRows) == 0 || len(ctbCols) == 0 {
		return nil, fmt.Errorf("MulMatricesCiphertextsParallel: input matrices cannot be empty")
	}

	if k_dimension <= 0 {
		return nil, fmt.Errorf("MulMatricesCiphertextsParallel: k_dimension must be positive")
	}

	if numWorkers <= 0 {
		return nil, fmt.Errorf("MulMatricesCiphertextsParallel: numWorkers must be positive")
	}

	// Get the dimensions of the result matrix
	m := len(ctaRows)    // Number of rows in A
	n := len(ctbCols)    // Number of columns in B
	// k_dimension is the shared dimension

	// Initialize the result matrix
	resultCiphertexts := make([][]*rlwe.Ciphertext, m)
	for i := 0; i < m; i++ {
		resultCiphertexts[i] = make([]*rlwe.Ciphertext, n)
	}

	// Create worker-specific evaluators with the same parameters and keys
	workerEvaluators := make([]*ckks.Evaluator, numWorkers)
	for w := 0; w < numWorkers; w++ {
		// Clone the evaluator for thread safety
		workerEvaluators[w] = evaluator.ShallowCopy()
	}

	// Create a wait group to synchronize goroutines
	var wg sync.WaitGroup

	// Create a mutex for synchronizing error handling
	errMutex := &sync.Mutex{}
	var firstErr error

	// Calculate how many rows each worker should process
	rowsPerWorker := m / numWorkers
	if rowsPerWorker == 0 {
		rowsPerWorker = 1
		numWorkers = m // Adjust number of workers if we have fewer rows than workers
	}

	// Start worker goroutines
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			// Get this worker's evaluator
			workerEval := workerEvaluators[workerID]

			// Calculate the range of rows this worker will process
			startRow := workerID * rowsPerWorker
			endRow := (workerID + 1) * rowsPerWorker
			if workerID == numWorkers-1 {
				endRow = m // Last worker takes any remaining rows
			}

			// Process assigned rows
			for i := startRow; i < endRow; i++ {
				// Check if an error has already occurred
				errMutex.Lock()
				if firstErr != nil {
					errMutex.Unlock()
					return
				}
				errMutex.Unlock()

				// Process each column for this row
				for j := 0; j < n; j++ {
					// Step 1: Multiply the row vector from A with the column vector from B element-wise
					mulCt, err := workerEval.MulNew(ctaRows[i], ctbCols[j])
					if err != nil {
						errMutex.Lock()
						if firstErr == nil {
							firstErr = fmt.Errorf("parallel multiplication failed for C[%d][%d]: %w", i, j, err)
						}
						errMutex.Unlock()
						return
					}

					// Step 2: Relinearize the product ciphertext
					if err = workerEval.Relinearize(mulCt, mulCt); err != nil {
						errMutex.Lock()
						if firstErr == nil {
							firstErr = fmt.Errorf("parallel relinearization failed for C[%d][%d]: %w", i, j, err)
						}
						errMutex.Unlock()
						return
					}

					// Step 3: Rescale to manage the noise and scale
					if err = workerEval.Rescale(mulCt, mulCt); err != nil {
						errMutex.Lock()
						if firstErr == nil {
							firstErr = fmt.Errorf("parallel rescale failed for C[%d][%d]: %w", i, j, err)
						}
						errMutex.Unlock()
						return
					}

					// Step 4: Compute the sum using the divide-and-conquer approach with powers of 2 rotations
					resultCt := mulCt.CopyNew()
					tempCt := mulCt.CopyNew()

					// Calculate log2(k_dimension)
					logK := 0
					for k := k_dimension; k > 1; k >>= 1 {
						logK++
					}

					// Use rotations with powers of 2
					for rotStep := 0; rotStep < logK; rotStep++ {
						rotation := 1 << rotStep
						if err = workerEval.Rotate(resultCt, rotation, tempCt); err != nil {
							errMutex.Lock()
							if firstErr == nil {
								firstErr = fmt.Errorf("parallel rotation failed for C[%d][%d], rotation %d: %w", i, j, rotation, err)
							}
							errMutex.Unlock()
							return
						}

						// Add the rotated ciphertext to the result
						if err = workerEval.Add(resultCt, tempCt, resultCt); err != nil {
							errMutex.Lock()
							if firstErr == nil {
								firstErr = fmt.Errorf("parallel addition failed for C[%d][%d], rotation %d: %w", i, j, rotation, err)
							}
							errMutex.Unlock()
							return
						}
					}

					// Store the result
					resultCiphertexts[i][j] = resultCt
				}
			}
		}(w)
	}

	// Wait for all workers to finish
	wg.Wait()

	// Check if any errors occurred
	if firstErr != nil {
		return nil, firstErr
	}

	return resultCiphertexts, nil
}
