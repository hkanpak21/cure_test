package he

import (
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// ConvLayer represents a homomorphic convolutional layer in a neural network.
// It implements both plaintext and homomorphic forward passes using the CKKS scheme.
// The implementation follows the approach described in the CURE paper, supporting
// configurable kernel sizes, strides, and padding.
type ConvLayer struct {
	// Kernel dimensions
	KernelHeight int
	KernelWidth  int
	// Input dimensions
	InputChannels  int
	OutputChannels int
	// Stride and padding
	StrideHeight int
	StrideWidth  int
	PaddingHeight int
	PaddingWidth  int
	// Weights and biases
	Weights [][][][]float64 // [outputChannels][inputChannels][kernelHeight][kernelWidth]
	Biases  []float64       // [outputChannels]
}

// NewConvLayer creates a new homomorphic convolutional layer with the given parameters.
// This constructor validates that the weight dimensions match the specified layer configuration
// to prevent errors during forward passes.
func NewConvLayer(
	inputChannels, outputChannels int,
	kernelHeight, kernelWidth int,
	strideHeight, strideWidth int,
	paddingHeight, paddingWidth int,
	weights [][][][]float64,
	biases []float64,
) (*ConvLayer, error) {
	// Validate dimensions
	if len(weights) != outputChannels {
		return nil, fmt.Errorf("weights dimension mismatch: expected first dimension %d, got %d", outputChannels, len(weights))
	}

	for i := 0; i < outputChannels; i++ {
		if len(weights[i]) != inputChannels {
			return nil, fmt.Errorf("weights dimension mismatch: expected second dimension %d, got %d", inputChannels, len(weights[i]))
		}
		for j := 0; j < inputChannels; j++ {
			if len(weights[i][j]) != kernelHeight {
				return nil, fmt.Errorf("weights dimension mismatch: expected third dimension %d, got %d", kernelHeight, len(weights[i][j]))
			}
			for k := 0; k < kernelHeight; k++ {
				if len(weights[i][j][k]) != kernelWidth {
					return nil, fmt.Errorf("weights dimension mismatch: expected fourth dimension %d, got %d", kernelWidth, len(weights[i][j][k]))
				}
			}
		}
	}

	if len(biases) != outputChannels {
		return nil, fmt.Errorf("biases dimension mismatch: expected length %d, got %d", outputChannels, len(biases))
	}

	return &ConvLayer{
		KernelHeight:   kernelHeight,
		KernelWidth:    kernelWidth,
		InputChannels:  inputChannels,
		OutputChannels: outputChannels,
		StrideHeight:   strideHeight,
		StrideWidth:    strideWidth,
		PaddingHeight:  paddingHeight,
		PaddingWidth:   paddingWidth,
		Weights:        weights,
		Biases:         biases,
	}, nil
}

// CalculateOutputDimensions calculates the output dimensions of the convolution operation
// using the standard formula for convolutional layers:
// outputDim = floor((inputDim + 2*padding - kernelSize)/stride + 1)
func (c *ConvLayer) CalculateOutputDimensions(inputHeight, inputWidth int) (outputHeight, outputWidth int) {
	outputHeight = int(math.Floor(float64(inputHeight+2*c.PaddingHeight-c.KernelHeight)/float64(c.StrideHeight)) + 1)
	outputWidth = int(math.Floor(float64(inputWidth+2*c.PaddingWidth-c.KernelWidth)/float64(c.StrideWidth)) + 1)
	return
}

// ForwardPlaintext performs the forward pass of the convolutional layer in plaintext.
// This serves as a reference implementation and for verification of homomorphic operations.
//
// The implementation follows standard convolution operations in CNNs, applying each kernel
// to the input and adding the bias term. Zero-padding is applied when specified.
//
// Input shape: [inputChannels][inputHeight][inputWidth]
// Output shape: [outputChannels][outputHeight][outputWidth]
//
// Performance note: This implementation prioritizes clarity over performance.
// For production use with large inputs, consider using optimized libraries.
func (c *ConvLayer) ForwardPlaintext(input [][][]float64) ([][][]float64, error) {
	if len(input) != c.InputChannels {
		return nil, fmt.Errorf("input channels mismatch: expected %d, got %d", c.InputChannels, len(input))
	}

	inputHeight := len(input[0])
	inputWidth := len(input[0][0])

	outputHeight, outputWidth := c.CalculateOutputDimensions(inputHeight, inputWidth)

	// Initialize output
	output := make([][][]float64, c.OutputChannels)
	for i := 0; i < c.OutputChannels; i++ {
		output[i] = make([][]float64, outputHeight)
		for j := 0; j < outputHeight; j++ {
			output[i][j] = make([]float64, outputWidth)
			// Initialize with bias
			for k := 0; k < outputWidth; k++ {
				output[i][j][k] = c.Biases[i]
			}
		}
	}

	// Perform convolution
	for oc := 0; oc < c.OutputChannels; oc++ {
		for oh := 0; oh < outputHeight; oh++ {
			for ow := 0; ow < outputWidth; ow++ {
				// Calculate input region (with padding consideration)
				inputStartH := oh*c.StrideHeight - c.PaddingHeight
				inputStartW := ow*c.StrideWidth - c.PaddingWidth

				// Apply kernel
				for ic := 0; ic < c.InputChannels; ic++ {
					for kh := 0; kh < c.KernelHeight; kh++ {
						ih := inputStartH + kh
						if ih < 0 || ih >= inputHeight {
							continue // Skip if outside input (zero padding)
						}

						for kw := 0; kw < c.KernelWidth; kw++ {
							iw := inputStartW + kw
							if iw < 0 || iw >= inputWidth {
								continue // Skip if outside input (zero padding)
							}

							output[oc][oh][ow] += input[ic][ih][iw] * c.Weights[oc][ic][kh][kw]
						}
					}
				}
			}
		}
	}

	return output, nil
}

// ForwardHomomorphic performs the forward pass of the convolutional layer on homomorphically encrypted data.
// This implementation follows the CURE paper approach for privacy-preserving neural network inference.
//
// The function operates on encrypted inputs, applying convolution operations without decrypting
// the data. This preserves privacy while allowing computation on the encrypted values.
//
// Input shape: [inputChannels][inputHeight*inputWidth] (flattened 2D input per channel)
// Output shape: [outputChannels][outputHeight*outputWidth] (flattened 2D output per channel)
//
// Memory usage: This implementation requires memory proportional to:
//   - inputChannels * outputChannels * outputHeight * outputWidth
// Consider using the optimized version for large networks.
func (c *ConvLayer) ForwardHomomorphic(
	inputCiphertexts []*rlwe.Ciphertext, // One ciphertext per input channel
	inputHeight, inputWidth int,
	evaluator *ckks.Evaluator,
	encryptor *rlwe.Encryptor,
	params ckks.Parameters,
) ([]*rlwe.Ciphertext, error) {
	if len(inputCiphertexts) != c.InputChannels {
		return nil, fmt.Errorf("input channels mismatch: expected %d, got %d", c.InputChannels, len(inputCiphertexts))
	}

	outputHeight, outputWidth := c.CalculateOutputDimensions(inputHeight, inputWidth)
	outputSize := outputHeight * outputWidth

	// Create output ciphertexts (one per output channel)
	outputCiphertexts := make([]*rlwe.Ciphertext, c.OutputChannels)

	// Process each output channel
	for oc := 0; oc < c.OutputChannels; oc++ {
		// Initialize with bias
		// Create a plaintext with the bias value
		biasPt := ckks.NewPlaintext(params, params.MaxLevel())
		
		// Create a slice with the bias value repeated for each output position
		biasValues := make([]float64, outputSize)
		for i := 0; i < outputSize; i++ {
			biasValues[i] = c.Biases[oc]
		}
		
		// Encode the bias values
		encoder := NewEncoder(params)
		encoder.Encode(biasValues, biasPt)
		
		// Initialize the output ciphertext with the encoded bias
		// In Lattigo v6, we need to create a new ciphertext first
		outputCiphertexts[oc] = ckks.NewCiphertext(params, 1, params.MaxLevel())
		
		// Create a temporary ciphertext for the bias
		biasCt, err := encryptor.EncryptNew(biasPt)
		if err != nil {
			return nil, fmt.Errorf("error encrypting bias: %w", err)
		}
		
		// Copy the bias ciphertext to the output
		outputCiphertexts[oc].Copy(biasCt)

		// For each input channel, convolve with the corresponding kernel and add to output
		for ic := 0; ic < c.InputChannels; ic++ {
			// Extract the kernel for this input-output channel pair
			kernel := c.Weights[oc][ic]

			// For each position in the output
			for oh := 0; oh < outputHeight; oh++ {
				for ow := 0; ow < outputWidth; ow++ {
					// Calculate the position in the output (for tracking progress and debugging)
					// This part of the code could be optimized further by vectorizing operations
					// to reduce the number of homomorphic operations

					// Calculate input region (with padding consideration)
					inputStartH := oh*c.StrideHeight - c.PaddingHeight
					inputStartW := ow*c.StrideWidth - c.PaddingWidth

					// For each position in the kernel
					for kh := 0; kh < c.KernelHeight; kh++ {
						ih := inputStartH + kh
						if ih < 0 || ih >= inputHeight {
							continue // Skip if outside input (zero padding)
						}

						for kw := 0; kw < c.KernelWidth; kw++ {
							iw := inputStartW + kw
							if iw < 0 || iw >= inputWidth {
								continue // Skip if outside input (zero padding)
							}

							// Get the weight for this kernel position
							weight := kernel[kh][kw]

							// Calculate the input index
							inputIdx := ih*inputWidth + iw

							// Create a mask to extract the input value at this position
							mask := make([]float64, params.N())
							mask[inputIdx] = 1.0

							// Encode the mask
							maskPt := ckks.NewPlaintext(params, params.MaxLevel())
							encoder.Encode(mask, maskPt)

							// Multiply the input ciphertext by the mask to extract the value
							// In Lattigo v6, we use Mul instead of MulPlainNew
							extractedCt := ckks.NewCiphertext(params, 1, inputCiphertexts[ic].Level())
							var err error
							err = evaluator.Mul(inputCiphertexts[ic], maskPt, extractedCt)
							if err != nil {
								return nil, fmt.Errorf("error multiplying by mask: %w", err)
							}

							// Multiply by the weight
							weightedCt, err := evaluator.MulNew(extractedCt, weight)
							if err != nil {
								return nil, fmt.Errorf("error multiplying by weight: %w", err)
							}

							// Add to the output
							// In Lattigo v6, we use Add instead of AddNew
							tempCt := ckks.NewCiphertext(params, 1, outputCiphertexts[oc].Level())
							err = evaluator.Add(outputCiphertexts[oc], weightedCt, tempCt)
							if err != nil {
								return nil, fmt.Errorf("error adding to output: %w", err)
							}
							outputCiphertexts[oc] = tempCt
						}
					}
				}
			}
		}
	}

	return outputCiphertexts, nil
}

// ForwardHomomorphicOptimized performs an optimized forward pass of the convolutional layer
// This implementation uses a more efficient approach by vectorizing operations
func (c *ConvLayer) ForwardHomomorphicOptimized(
	inputCiphertexts []*rlwe.Ciphertext, // One ciphertext per input channel
	inputHeight, inputWidth int,
	evaluator *ckks.Evaluator,
	encryptor *rlwe.Encryptor,
	params ckks.Parameters,
) ([]*rlwe.Ciphertext, error) {
	if len(inputCiphertexts) != c.InputChannels {
		return nil, fmt.Errorf("input channels mismatch: expected %d, got %d", c.InputChannels, len(inputCiphertexts))
	}

	outputHeight, outputWidth := c.CalculateOutputDimensions(inputHeight, inputWidth)
	outputSize := outputHeight * outputWidth

	// Create output ciphertexts (one per output channel)
	outputCiphertexts := make([]*rlwe.Ciphertext, c.OutputChannels)
	for oc := 0; oc < c.OutputChannels; oc++ {
		// Initialize with zeros
		outputCiphertexts[oc] = ckks.NewCiphertext(params, 1, params.MaxLevel())
	}

	// Create encoder
	encoder := NewEncoder(params)

	// Process each output channel
	for oc := 0; oc < c.OutputChannels; oc++ {
		// Create a result accumulator for this output channel
		resultCt := ckks.NewCiphertext(params, 1, params.MaxLevel())

		// For each input channel
		for ic := 0; ic < c.InputChannels; ic++ {
			// For each position in the kernel
			for kh := 0; kh < c.KernelHeight; kh++ {
				for kw := 0; kw < c.KernelWidth; kw++ {
					// Get the weight for this kernel position
					weight := c.Weights[oc][ic][kh][kw]

					// Skip if weight is zero (optimization)
					if weight == 0 {
						continue
					}

					// Create a rotation matrix for this kernel position
					rotationMatrix := make([]float64, outputSize*params.N())

					// For each position in the output
					for oh := 0; oh < outputHeight; oh++ {
						for ow := 0; ow < outputWidth; ow++ {
							// Calculate corresponding input position
							ih := oh*c.StrideHeight + kh - c.PaddingHeight
							iw := ow*c.StrideWidth + kw - c.PaddingWidth

							// Skip if outside input boundaries (zero padding)
							if ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth {
								inputIdx := ih*inputWidth + iw
								rotationMatrix[oh*outputWidth*params.N()+ow*params.N()+inputIdx] = weight
							}
						}
					}

					// Encode the rotation matrix
					rotationPt := ckks.NewPlaintext(params, params.MaxLevel())
					encoder.Encode(rotationMatrix, rotationPt)

					// Apply the rotation matrix to the input
					rotatedCt := ckks.NewCiphertext(params, 1, inputCiphertexts[ic].Level())
					var err error
					err = evaluator.Mul(inputCiphertexts[ic], rotationPt, rotatedCt)
					if err != nil {
						return nil, fmt.Errorf("error applying rotation matrix: %w", err)
					}

					// Add to the result accumulator
					tempCt := ckks.NewCiphertext(params, 1, resultCt.Level())
					err = evaluator.Add(resultCt, rotatedCt, tempCt)
					if err != nil {
						return nil, fmt.Errorf("error adding to result accumulator: %w", err)
					}
					resultCt = tempCt
				}
			}
		}

		// Add bias
		biasValues := make([]float64, params.N())
		for i := 0; i < outputSize; i++ {
			biasValues[i] = c.Biases[oc]
		}
		biasPt := ckks.NewPlaintext(params, params.MaxLevel())
		encoder.Encode(biasValues, biasPt)

		// Add bias to result
		// In Lattigo v6, we use Add instead of AddPlainNew
		tempCt := ckks.NewCiphertext(params, 1, resultCt.Level())
		// Create a temporary ciphertext for the bias
		biasCt, err := encryptor.EncryptNew(biasPt)
		if err != nil {
			return nil, fmt.Errorf("error encrypting bias: %w", err)
		}
		
		// Add the bias ciphertext to the result
		err = evaluator.Add(resultCt, biasCt, tempCt)
		if err != nil {
			return nil, fmt.Errorf("error adding bias: %w", err)
		}
		resultCt = tempCt

		// Store the result for this output channel
		outputCiphertexts[oc] = resultCt
	}

	return outputCiphertexts, nil
}
