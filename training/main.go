package main

import (
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

// --- Configuration ---
const (
	numEpochs     = 10
	learningRate  = 0.001 // Adjusted learning rate
	batchSize     = 64
	mnistDataPath = "../data" // Relative to training executable
	numClasses    = 10
)

// --- Data Loading ---

// MNISTImage represents a single MNIST image.
type MNISTImage struct {
	Pixels []float64 // Flattened image data, normalized to [0,1]
	Rows   int
	Cols   int
}

// MNISTLabel is the corresponding label (0-9).
type MNISTLabel uint8

func loadMNISTImages(filePath string) ([]MNISTImage, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open image file %s: %w", filePath, err)
	}
	defer file.Close()

	var magic, numImages, numRows, numCols int32
	if err := binary.Read(file, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("failed to read magic number: %w", err)
	}
	if magic != 2051 { // Magic number for image files
		return nil, fmt.Errorf("invalid magic number for image file: %d", magic)
	}

	if err := binary.Read(file, binary.BigEndian, &numImages); err != nil {
		return nil, fmt.Errorf("failed to read number of images: %w", err)
	}
	if err := binary.Read(file, binary.BigEndian, &numRows); err != nil {
		return nil, fmt.Errorf("failed to read number of rows: %w", err)
	}
	if err := binary.Read(file, binary.BigEndian, &numCols); err != nil {
		return nil, fmt.Errorf("failed to read number of columns: %w", err)
	}

	images := make([]MNISTImage, numImages)
	imageSize := int(numRows * numCols)
	buf := make([]byte, imageSize)

	for i := 0; i < int(numImages); i++ {
		_, err := io.ReadFull(file, buf)
		if err != nil {
			return nil, fmt.Errorf("failed to read image data for image %d: %w", i, err)
		}
		pixels := make([]float64, imageSize)
		for j, val := range buf {
			pixels[j] = float64(val) / 255.0 // Normalize to [0,1]
		}
		images[i] = MNISTImage{Pixels: pixels, Rows: int(numRows), Cols: int(numCols)}
	}
	return images, nil
}

func loadMNISTLabels(filePath string) ([]MNISTLabel, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open label file %s: %w", filePath, err)
	}
	defer file.Close()

	var magic, numLabels int32
	if err := binary.Read(file, binary.BigEndian, &magic); err != nil {
		return nil, fmt.Errorf("failed to read magic number for labels: %w", err)
	}
	if magic != 2049 { // Magic number for label files
		return nil, fmt.Errorf("invalid magic number for label file: %d", magic)
	}

	if err := binary.Read(file, binary.BigEndian, &numLabels); err != nil {
		return nil, fmt.Errorf("failed to read number of labels: %w", err)
	}

	labels := make([]MNISTLabel, numLabels)
	buf := make([]byte, numLabels)
	_, err = io.ReadFull(file, buf)
	if err != nil {
		return nil, fmt.Errorf("failed to read label data: %w", err)
	}
	for i, val := range buf {
		labels[i] = MNISTLabel(val)
	}
	return labels, nil
}

// --- Plaintext Layer Definitions ---

// PlainLayer interface for all plaintext layers
type PlainLayer interface {
	Forward(input [][]float64) [][]float64
	Backward(dOutput [][]float64) [][]float64
	UpdateWeights(learningRate float64)
	Name() string
}

// PlainDenseLayer
type PlainDenseLayer struct {
	name        string
	weights     [][]float64 // [inputSize][outputSize]
	biases      []float64   // [outputSize]
	inputCache  [][]float64 // Cache input for backward pass
	dWeights    [][]float64
	dBiases     []float64
	inputSize   int
	outputSize  int
}

func NewPlainDenseLayer(name string, inputSize, outputSize int) *PlainDenseLayer {
	weights := make([][]float64, inputSize)
	// Xavier/Glorot initialization
	limit := math.Sqrt(6.0 / float64(inputSize+outputSize))
	for i := range weights {
		weights[i] = make([]float64, outputSize)
		for j := range weights[i] {
			weights[i][j] = (rand.Float64()*2 - 1) * limit
		}
	}
	biases := make([]float64, outputSize) // Initialize biases to zero or small value

	return &PlainDenseLayer{
		name:       name,
		weights:    weights,
		biases:     biases,
		inputSize:  inputSize,
		outputSize: outputSize,
	}
}

func (l *PlainDenseLayer) Forward(input [][]float64) [][]float64 {
	l.inputCache = input
	batchSize := len(input)
	output := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		output[i] = make([]float64, l.outputSize)
		for j := 0; j < l.outputSize; j++ {
			sum := 0.0
			for k := 0; k < l.inputSize; k++ {
				sum += input[i][k] * l.weights[k][j]
			}
			output[i][j] = sum + l.biases[j]
		}
	}
	return output
}

func (l *PlainDenseLayer) Backward(dOutput [][]float64) [][]float64 {
	batchSize := len(l.inputCache)
	dInput := make([][]float64, batchSize)
	l.dWeights = make([][]float64, l.inputSize)
	for i := range l.dWeights {
		l.dWeights[i] = make([]float64, l.outputSize)
	}
	l.dBiases = make([]float64, l.outputSize)

	for i := 0; i < batchSize; i++ {
		dInput[i] = make([]float64, l.inputSize)
		// Gradient w.r.t. input
		for k := 0; k < l.inputSize; k++ {
			sum := 0.0
			for j := 0; j < l.outputSize; j++ {
				sum += dOutput[i][j] * l.weights[k][j]
			}
			dInput[i][k] = sum
		}
		// Gradient w.r.t. weights and biases
		for j := 0; j < l.outputSize; j++ {
			for k := 0; k < l.inputSize; k++ {
				l.dWeights[k][j] += l.inputCache[i][k] * dOutput[i][j]
			}
			l.dBiases[j] += dOutput[i][j]
		}
	}

	// Average gradients over batch
	for i := range l.dWeights {
		for j := range l.dWeights[i] {
			l.dWeights[i][j] /= float64(batchSize)
		}
	}
	for i := range l.dBiases {
		l.dBiases[i] /= float64(batchSize)
	}
	return dInput
}

func (l *PlainDenseLayer) UpdateWeights(learningRate float64) {
	for i := 0; i < l.inputSize; i++ {
		for j := 0; j < l.outputSize; j++ {
			l.weights[i][j] -= learningRate * l.dWeights[i][j]
		}
	}
	for i := 0; i < l.outputSize; i++ {
		l.biases[i] -= learningRate * l.dBiases[i]
	}
}
func (l *PlainDenseLayer) Name() string { return l.name }

// PlainReLULayer
type PlainReLULayer struct {
	name       string
	inputCache [][]float64
}

func NewPlainReLULayer(name string) *PlainReLULayer {
	return &PlainReLULayer{name: name}
}

func (l *PlainReLULayer) Forward(input [][]float64) [][]float64 {
	l.inputCache = input
	output := make([][]float64, len(input))
	for i, batch := range input {
		output[i] = make([]float64, len(batch))
		for j, val := range batch {
			if val > 0 {
				output[i][j] = val
			} else {
				output[i][j] = 0
			}
		}
	}
	return output
}

func (l *PlainReLULayer) Backward(dOutput [][]float64) [][]float64 {
	dInput := make([][]float64, len(dOutput))
	for i, batchGrad := range dOutput {
		dInput[i] = make([]float64, len(batchGrad))
		for j, grad := range batchGrad {
			if l.inputCache[i][j] > 0 {
				dInput[i][j] = grad
			} else {
				dInput[i][j] = 0
			}
		}
	}
	return dInput
}

func (l *PlainReLULayer) UpdateWeights(learningRate float64) {} // No weights
func (l *PlainReLULayer) Name() string                       { return l.name }

// PlainSoftmaxOutputLayer (combines Softmax and CrossEntropyLoss)
type PlainSoftmaxOutputLayer struct {
	name            string
	predictionsCache [][]float64 // Cache softmax predictions for backward pass
}

func NewPlainSoftmaxOutputLayer(name string) *PlainSoftmaxOutputLayer {
	return &PlainSoftmaxOutputLayer{name: name}
}

func (l *PlainSoftmaxOutputLayer) Forward(input [][]float64) [][]float64 {
	batchSize := len(input)
	output := make([][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		row := input[i]
		maxVal := row[0]
		for _, val := range row {
			if val > maxVal {
				maxVal = val
			}
		}
		sumExp := 0.0
		output[i] = make([]float64, len(row))
		for j, val := range row {
			expVal := math.Exp(val - maxVal) // Stability trick
			output[i][j] = expVal
			sumExp += expVal
		}
		for j := range output[i] {
			output[i][j] /= sumExp
		}
	}
	l.predictionsCache = output
	return output
}

// Backward for Softmax with Cross-Entropy expects true labels (one-hot or indices)
// This simplified version calculates dLoss/dInput_to_softmax directly.
// CalculateInitialGradient computes the gradient of the loss function
// with respect to the input of the softmax layer (dLoss/dZ).
// It uses the true labels to compute this initial gradient.
func (l *PlainSoftmaxOutputLayer) CalculateInitialGradient(trueLabelsBatch [][]MNISTLabel) [][]float64 {
	batchSize := len(l.predictionsCache)
	if batchSize == 0 {
		// log.Println("Warning: CalculateInitialGradient called with empty predictions cache.")
		return [][]float64{}
	}
	numClasses := 0
	if len(l.predictionsCache[0]) > 0 { // Check if the first prediction has elements
		numClasses = len(l.predictionsCache[0])
	} else if batchSize > 0 { // If there are predictions but the first one is empty
		// log.Println("Warning: CalculateInitialGradient called with zero classes in the first prediction sample.")
		return [][]float64{} // Or handle as an error: all samples should have same numClasses
	} else { // batchSize is 0, already handled
		return [][]float64{}
	}

	dInput := make([][]float64, batchSize)

	for i := 0; i < batchSize; i++ {
		dInput[i] = make([]float64, numClasses)
		// Ensure trueLabelsBatch[i] and trueLabelsBatch[i][0] are safe to access
		if len(trueLabelsBatch) <= i || len(trueLabelsBatch[i]) == 0 {
			log.Printf("Warning: Missing true label for sample %d in batch during CalculateInitialGradient. Gradient for this sample will be zeros.", i)
			continue // dInput[i] will remain zeros
		}
		trueLabel := trueLabelsBatch[i][0]
		for j := 0; j < numClasses; j++ {
			indicator := 0.0
			if j == int(trueLabel) {
				indicator = 1.0
			}
			// Gradient of CE Loss w.r.t Softmax input (logits): p_j - y_j
			// The division by batchSize here means the gradient is already averaged.
			dInput[i][j] = (l.predictionsCache[i][j] - indicator) / float64(batchSize)
		}
	}
	return dInput
}

// Backward satisfies the PlainLayer interface. For the PlainSoftmaxOutputLayer,
// the gradient is initiated by CalculateInitialGradient using true labels,
// not by a dOutput from a subsequent layer (as there is none).
// This method should ideally not be called directly in the standard backpropagation flow
// if the output layer is handled as a special case in the training loop.
func (l *PlainSoftmaxOutputLayer) Backward(dOutput [][]float64) [][]float64 {
	log.Println("Warning: PlainSoftmaxOutputLayer.Backward(dOutput [][]float64) called. This is generally not expected for the final output layer if gradients are initiated using true labels. The gradient should be initiated via CalculateInitialGradient.")
	// This method is required to satisfy the PlainLayer interface.
	// In a typical scenario where this layer is the last, dOutput would be dLoss/dActivation_of_this_layer.
	// However, for Softmax with CrossEntropy, it's more direct to compute dLoss/dPreActivation_of_this_layer.
	// Returning nil as a safeguard, as this path indicates a potential misunderstanding in the training loop logic.
	return nil
}

func (l *PlainSoftmaxOutputLayer) UpdateWeights(learningRate float64) {} // No weights
func (l *PlainSoftmaxOutputLayer) Name() string                       { return l.name }

func calculateCrossEntropyLoss(predictions [][]float64, trueLabelsBatch [][]MNISTLabel) float64 {
	batchSize := len(predictions)
	totalLoss := 0.0
	for i := 0; i < batchSize; i++ {
		trueLabel := trueLabelsBatch[i][0]
		predictedProb := predictions[i][trueLabel]
		if predictedProb < 1e-9 { // Avoid log(0)
			predictedProb = 1e-9
		}
		totalLoss -= math.Log(predictedProb)
	}
	return totalLoss / float64(batchSize)
}

func calculateAccuracy(predictions [][]float64, trueLabelsBatch [][]MNISTLabel) float64 {
	correct := 0
	batchSize := len(predictions)
	for i := 0; i < batchSize; i++ {
		trueLabel := trueLabelsBatch[i][0]
		maxProb := -1.0
		predictedLabel := -1
		for j, prob := range predictions[i] {
			if prob > maxProb {
				maxProb = prob
				predictedLabel = j
			}
		}
		if predictedLabel == int(trueLabel) {
			correct++
		}
	}
	return float64(correct) / float64(batchSize)
}

// --- Model Definition ---
type PlainModel struct {
	layers []PlainLayer
}

func buildPlainDNN() *PlainModel {
	log.Println("Building Plain DNN Model...")
	model := &PlainModel{}
	// 784 -> 128 ReLU -> 128 ReLU -> 32 ReLU -> 10 Softmax
	model.layers = append(model.layers, NewPlainDenseLayer("dense1", 28*28, 128))
	model.layers = append(model.layers, NewPlainReLULayer("relu1"))
	model.layers = append(model.layers, NewPlainDenseLayer("dense2", 128, 128))
	model.layers = append(model.layers, NewPlainReLULayer("relu2"))
	model.layers = append(model.layers, NewPlainDenseLayer("dense3", 128, 32))
	model.layers = append(model.layers, NewPlainReLULayer("relu3"))
	model.layers = append(model.layers, NewPlainDenseLayer("dense_output", 32, numClasses))
	model.layers = append(model.layers, NewPlainSoftmaxOutputLayer("softmax_output"))
	log.Println("Plain DNN Model built.")
	return model
}

func main() {
	rand.Seed(time.Now().UnixNano())
	log.Println("Starting PLAINTEXT DNN training process...")

	// 1. Load Data
	log.Println("Loading MNIST data...")
	trainImages, err := loadMNISTImages(mnistDataPath + "/train-images-idx3-ubyte")
	if err != nil {
		log.Fatalf("Failed to load training images: %v", err)
	}
	actualTrainLabels, err := loadMNISTLabels(mnistDataPath + "/train-labels-idx1-ubyte")
	if err != nil {
		log.Fatalf("Failed to load training labels: %v", err)
	}
	if len(trainImages) != len(actualTrainLabels) {
		log.Fatalf("Mismatch in number of training images (%d) and labels (%d)", len(trainImages), len(actualTrainLabels))
	}
	log.Printf("Loaded %d training images and %d labels.\n", len(trainImages), len(actualTrainLabels))

	// For simplicity, using training data also for per-epoch evaluation. Ideally, use a separate validation set.

	// 2. Build Model
	model := buildPlainDNN()

	// 3. Training Loop
	log.Println("Starting training loop...")
	trainingResultsCSV := [][]string{{"Epoch", "AvgLoss", "AvgAccuracy", "EpochTimeSeconds"}}

	numTrainSamples := len(trainImages)

	for epoch := 0; epoch < numEpochs; epoch++ {
		epochStartTime := time.Now()
		log.Printf("--- Epoch %d/%d ---\n", epoch+1, numEpochs)
		totalEpochLoss := 0.0
		totalEpochAccuracy := 0.0
		batchesProcessed := 0

		// Shuffle data (indices) each epoch
		perm := rand.Perm(numTrainSamples)

		for i := 0; i < numTrainSamples; i += batchSize {
			end := i + batchSize
			if end > numTrainSamples {
				end = numTrainSamples
			}
			if i == end { continue } // Skip if batch is empty

			// Create batch
			currentBatchSize := end - i
			inputBatch := make([][]float64, currentBatchSize)
			labelBatch := make([][]MNISTLabel, currentBatchSize) // For SoftmaxOutputLayer

			for batchIdx := 0; batchIdx < currentBatchSize; batchIdx++ {
				originalIdx := perm[i+batchIdx]
				inputBatch[batchIdx] = trainImages[originalIdx].Pixels
				labelBatch[batchIdx] = []MNISTLabel{actualTrainLabels[originalIdx]}
			}

			// Forward pass
			currentData := inputBatch
			for _, layer := range model.layers {
				currentData = layer.Forward(currentData)
			}
			predictions := currentData // Output of Softmax

			// Calculate loss and accuracy for the batch
			batchLoss := calculateCrossEntropyLoss(predictions, labelBatch)
			batchAccuracy := calculateAccuracy(predictions, labelBatch)
			totalEpochLoss += batchLoss * float64(currentBatchSize)
			totalEpochAccuracy += batchAccuracy * float64(currentBatchSize)
			batchesProcessed++

			if batchesProcessed%100 == 0 { // Log progress every 100 batches
				log.Printf("Epoch %d, Batch %d/%d, Batch Loss: %f, Batch Acc: %f\n", epoch+1, batchesProcessed, numTrainSamples/batchSize, batchLoss, batchAccuracy)
			}

			// Backward pass
			// The SoftmaxOutputLayer's Backward method is special, it takes true labels
			var grad [][]float64
			softmaxLayer, ok := model.layers[len(model.layers)-1].(*PlainSoftmaxOutputLayer)
			if !ok {
				log.Fatalf("Last layer is not PlainSoftmaxOutputLayer")
			}
			grad = softmaxLayer.CalculateInitialGradient(labelBatch) // This computes dLoss/dInput_to_softmax

			for lIdx := len(model.layers) - 2; lIdx >= 0; lIdx-- { // Start from layer before softmax
				layer := model.layers[lIdx]
				grad = layer.Backward(grad)
			}

			// Update weights
			for _, layer := range model.layers {
				layer.UpdateWeights(learningRate)
			}
		} // End of batch loop

		avgEpochLoss := totalEpochLoss / float64(numTrainSamples)
		avgEpochAccuracy := totalEpochAccuracy / float64(numTrainSamples)
		epochEndTime := time.Now()
		epochDurationSeconds := epochEndTime.Sub(epochStartTime).Seconds()
		log.Printf("Epoch %d COMPLETED. Avg Loss: %f, Avg Accuracy: %f, Time: %.2fs\n", epoch+1, avgEpochLoss, avgEpochAccuracy, epochDurationSeconds)
		trainingResultsCSV = append(trainingResultsCSV, []string{
			strconv.Itoa(epoch + 1),
			fmt.Sprintf("%f", avgEpochLoss),
			fmt.Sprintf("%f", avgEpochAccuracy),
			fmt.Sprintf("%.2f", epochDurationSeconds),
		})

	} // End of epoch loop

	log.Println("Training loop finished.")

	// 4. Save Results
	resultsFile, err := os.Create("plaintext_training_results.csv")
	if err != nil {
		log.Fatalf("Failed to create results CSV file: %v", err)
	}
	defer resultsFile.Close()

	csvWriter := csv.NewWriter(resultsFile)
	if err := csvWriter.WriteAll(trainingResultsCSV); err != nil {
		log.Fatalf("Failed to write to CSV file: %v", err)
	}
	csvWriter.Flush()
	log.Println("Plaintext training results saved to plaintext_training_results.csv")

	log.Println("Plaintext DNN training process completed.")
}
