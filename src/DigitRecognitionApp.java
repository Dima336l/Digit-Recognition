import java.io.*;
import java.util.*;
import java.util.Arrays;

/**
 * Main application class for the Digit Recognition System.
 * Orchestrates the machine learning pipeline including data loading,
 * algorithm training, evaluation, and results reporting.
 * 
 * CST 3170 Machine Learning Coursework - Digit Recognition System
 * Author: Dumitru Nirca
 * Date: December 2025
 * 
 * This system implements multiple machine learning algorithms for optical character recognition
 * of handwritten digits using the UCI Machine Learning Repository dataset.
 * 
 * Algorithms implemented:
 * 1. Nearest Neighbor (baseline) with Euclidean distance
 * 2. k-Nearest Neighbors with various k values and hyperparameter tuning
 * 3. Weighted k-NN with distance-based voting
 * 4. Linear Support Vector Machine (SVM) using Pegasos-style stochastic gradient descent
 * 
 * The system performs two-fold cross-validation and generates comprehensive performance metrics
 * including accuracy, confusion matrices, and classification reports.
 * 
 * @author Dumitru Nirca
 */
public class DigitRecognitionApp {
    
    // Configuration constants
    private static final String DATASET1_PATH = "dataSet1.csv";
    private static final String DATASET2_PATH = "dataSet2.csv";
    private static final int[] K_VALUES = {1, 3, 5, 7, 9, 11}; // k values to test
    
    /**
     * Main method - entry point of the program.
     * 
     * @param args Command line arguments (not used)
     */
    public static void main(String[] args) {
        System.out.println("=== CST 3170 Digit Recognition System ===");
        System.out.println("Loading datasets and performing comprehensive evaluation...\n");
        
        try {
            // Load datasets
            List<DigitSample> dataset1 = DatasetLoader.loadDataset(DATASET1_PATH);
            List<DigitSample> dataset2 = DatasetLoader.loadDataset(DATASET2_PATH);
            
            System.out.println("Dataset 1 size: " + dataset1.size() + " samples");
            System.out.println("Dataset 2 size: " + dataset2.size() + " samples\n");
            
            // Analyze dataset distributions
            DatasetLoader.printDatasetAnalysis(dataset1, "Dataset 1");
            DatasetLoader.printDatasetAnalysis(dataset2, "Dataset 2");
            
            // Perform two-fold cross-validation with multiple algorithms
            performTwoFoldValidation(dataset1, dataset2);
            
        } catch (Exception exception) {
            System.err.println("Error: " + exception.getMessage());
            exception.printStackTrace();
        }
    }
    
    /**
     * Performs comprehensive two-fold cross-validation with multiple algorithms.
     * 
     * @param dataset1 First dataset
     * @param dataset2 Second dataset
     */
    private static void performTwoFoldValidation(List<DigitSample> dataset1, List<DigitSample> dataset2) {
        System.out.println("\n=== TWO-FOLD CROSS-VALIDATION RESULTS ===\n");
        
        // Fold 1: Train on dataset1, test on dataset2
        System.out.println("--- FOLD 1: Training on Dataset 1, Testing on Dataset 2 ---");
        FoldResults fold1Results = evaluateAllAlgorithms(dataset1, dataset2, "Fold 1");
        
        // Fold 2: Train on dataset2, test on dataset1
        System.out.println("\n--- FOLD 2: Training on Dataset 2, Testing on Dataset 1 ---");
        FoldResults fold2Results = evaluateAllAlgorithms(dataset2, dataset1, "Fold 2");
        
        // Print overall summary
        printOverallSummary(fold1Results, fold2Results);
    }
    
    /**
     * Evaluates all implemented algorithms on given training and test sets.
     * 
     * @param trainSet Training dataset
     * @param testSet Test dataset
     * @param foldName Name of the fold for display purposes
     * @return Results summary for this fold
     */
    private static final double[] SVM_LAMBDA_CANDIDATES = {
        0.0003, 0.00035, 0.0004, 0.00045, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.0010, 0.0012
    };
    private static final int[] SVM_EPOCH_CANDIDATES = {100, 120, 140, 160, 180, 200, 220, 240};
    private static final double[] SVM_MIN_LR_CANDIDATES = {5e-8, 7.5e-8, 1e-7, 1.25e-7, 1.5e-7};
    private static final int SVM_ENSEMBLE_SIZE = 5;
    private static final int SVM_VALIDATION_REPEATS = 6;
    private static final double VALIDATION_SPLIT_RATIO = 0.85;
    
    private static FoldResults evaluateAllAlgorithms(List<DigitSample> trainSet, List<DigitSample> testSet, String foldName) {
        System.out.println("\n1. NEAREST NEIGHBOR (k=1) - BASELINE:");
        NearestNeighbor nnClassifier = new NearestNeighbor();
        nnClassifier.train(trainSet);
        double nnAccuracy = EvaluationMetrics.evaluateClassifier(nnClassifier, testSet);
        
        System.out.println("\n2. k-NEAREST NEIGHBORS HYPERPARAMETER ANALYSIS:");
        double bestAccuracy = 0;
        int bestK = K_VALUES[0];
        for (int k : K_VALUES) {
            KNearestNeighbors knnClassifier = new KNearestNeighbors(k, false);
            knnClassifier.train(trainSet);
            double accuracy = knnClassifier.evaluate(testSet);
            System.out.printf("  k=%d: Accuracy = %.2f%%\n", k, accuracy);
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestK = k;
            }
        }
        System.out.printf("\nBest k-NN performance: k=%d with %.2f%% accuracy\n\n", bestK, bestAccuracy);
        
        System.out.println("3. WEIGHTED k-NN (distance-based voting):");
        KNearestNeighbors weightedKNN = new KNearestNeighbors(bestK, true);
        weightedKNN.train(trainSet);
        double weightedAccuracy = EvaluationMetrics.evaluateClassifier(weightedKNN, testSet);
        
        System.out.println("\n4. SUPPORT VECTOR MACHINE (Linear SVM):");
        SVMTrainingResult svmTrainingResult = trainBestLinearSVM(trainSet);
        System.out.printf("Selected Linear SVM config: %s (validation accuracy %.2f%%)\n",
                          svmTrainingResult.hyperparameterSummary,
                          svmTrainingResult.validationAccuracy);
        double svmAccuracy = EvaluationMetrics.evaluateClassifier(svmTrainingResult.classifier, testSet);
        
        System.out.println("\n" + foldName + " SUMMARY:");
        System.out.printf("  Nearest Neighbor (k=1): %.2f%%\n", nnAccuracy);
        System.out.printf("  k-NN (best k=%d): %.2f%%\n", bestK, bestAccuracy);
        System.out.printf("  Weighted k-NN (k=%d): %.2f%%\n", bestK, weightedAccuracy);
        System.out.printf("  Linear SVM [%s]: %.2f%%\n", svmTrainingResult.hyperparameterSummary, svmAccuracy);
        System.out.println();
        
        return new FoldResults(nnAccuracy, bestK, bestAccuracy, weightedAccuracy, svmAccuracy, svmTrainingResult.hyperparameterSummary);
    }
    
    /**
     * Prints the overall summary of both folds.
     * 
     * @param fold1Results Results from fold 1
     * @param fold2Results Results from fold 2
     */
    private static void printOverallSummary(FoldResults fold1Results, FoldResults fold2Results) {
        System.out.println("\n=== OVERALL SUMMARY (Two-Fold Cross-Validation) ===\n");
        
        double avgNNAccuracy = (fold1Results.nnAccuracy + fold2Results.nnAccuracy) / 2.0;
        double avgKnnAccuracy = (fold1Results.bestKnnAccuracy + fold2Results.bestKnnAccuracy) / 2.0;
        double avgWeightedKnnAccuracy = (fold1Results.weightedKnnAccuracy + fold2Results.weightedKnnAccuracy) / 2.0;
        double avgSVMAccuracy = (fold1Results.svmAccuracy + fold2Results.svmAccuracy) / 2.0;
        
        System.out.println("Average Performance Across Both Folds:");
        System.out.printf("  Nearest Neighbor (k=1): %.2f%%\n", avgNNAccuracy);
        System.out.printf("  k-NN (best k per fold): %.2f%%\n", avgKnnAccuracy);
        System.out.printf("  Weighted k-NN: %.2f%%\n", avgWeightedKnnAccuracy);
        System.out.printf("  Linear SVM: %.2f%%\n", avgSVMAccuracy);
        
        System.out.println("\nFold Hyperparameter Selections:");
        System.out.printf("  Fold 1 - k-NN best k: %d, Weighted k-NN k: %d, SVM: %s\n", 
                         fold1Results.bestK, fold1Results.bestK, fold1Results.svmHyperparameterSummary);
        System.out.printf("  Fold 2 - k-NN best k: %d, Weighted k-NN k: %d, SVM: %s\n", 
                         fold2Results.bestK, fold2Results.bestK, fold2Results.svmHyperparameterSummary);
        
        System.out.println("\n=== Analysis Complete ===");
    }
    
    /**
     * Helper class to store results from a single fold evaluation.
     */
    private static class FoldResults {
        final double nnAccuracy;
        final int bestK;
        final double bestKnnAccuracy;
        final double weightedKnnAccuracy;
        final double svmAccuracy;
        final String svmHyperparameterSummary;
        
        FoldResults(double nnAccuracy, int bestK, double bestKnnAccuracy, double weightedKnnAccuracy, 
                   double svmAccuracy, String svmHyperparameterSummary) {
            this.nnAccuracy = nnAccuracy;
            this.bestK = bestK;
            this.bestKnnAccuracy = bestKnnAccuracy;
            this.weightedKnnAccuracy = weightedKnnAccuracy;
            this.svmAccuracy = svmAccuracy;
            this.svmHyperparameterSummary = svmHyperparameterSummary;
        }
    }
    
    private static SVMTrainingResult trainBestLinearSVM(List<DigitSample> trainingData) {
        if (trainingData == null || trainingData.size() < 2) {
            throw new IllegalArgumentException("Training data must contain at least two samples for SVM training");
        }
        
        List<HyperparameterCandidate> candidateResults = new ArrayList<>();
        
        System.out.println("Performing internal validation for Linear SVM hyperparameters...");
        
        for (double lambda : SVM_LAMBDA_CANDIDATES) {
            for (int epochs : SVM_EPOCH_CANDIDATES) {
                for (double minLR : SVM_MIN_LR_CANDIDATES) {
                    double accuracy = evaluateSVMHyperparameters(trainingData, lambda, epochs, minLR);
                    System.out.printf("  Candidate λ=%.5f, epochs=%d, η_min=%.1e -> validation accuracy %.2f%%\n",
                                      lambda, epochs, minLR, accuracy);
                    
                    candidateResults.add(new HyperparameterCandidate(lambda, epochs, minLR, accuracy));
                }
            }
        }
        
        Collections.sort(candidateResults);
        List<LinearSVM> ensembleModels = new ArrayList<>();
        List<String> ensembleSummaries = new ArrayList<>();
        
        int modelsToTrain = Math.min(SVM_ENSEMBLE_SIZE, candidateResults.size());
        for (int i = 0; i < modelsToTrain; i++) {
            HyperparameterCandidate candidate = candidateResults.get(i);
            LinearSVM model = new LinearSVM(candidate.lambda, candidate.epochs, candidate.minLearningRate);
            model.train(trainingData);
            ensembleModels.add(model);
            ensembleSummaries.add(candidate.describe());
        }
        
        Classifier classifier;
        String summaryText;
        double referenceAccuracy = candidateResults.isEmpty() ? 0.0 : candidateResults.get(0).validationAccuracy;
        
        if (ensembleModels.size() == 1) {
            classifier = ensembleModels.get(0);
            summaryText = ensembleSummaries.get(0);
        } else {
            classifier = new LinearSVMEnsemble(ensembleModels);
            summaryText = String.format("Ensemble of %d models: %s", 
                                        ensembleModels.size(), String.join("; ", ensembleSummaries));
        }
        
        return new SVMTrainingResult(classifier, summaryText, referenceAccuracy);
    }
    
    private static double evaluateSVMHyperparameters(List<DigitSample> trainingData,
                                                     double lambda,
                                                     int epochs,
                                                     double minLearningRate) {
        double accuracySum = 0.0;
        
        for (int repeat = 0; repeat < SVM_VALIDATION_REPEATS; repeat++) {
            List<DigitSample> shuffled = new ArrayList<>(trainingData);
            Collections.shuffle(shuffled, new Random(Objects.hash(lambda, epochs, minLearningRate, repeat)));
            
            int validationStartIndex = (int) (shuffled.size() * VALIDATION_SPLIT_RATIO);
            validationStartIndex = Math.min(Math.max(validationStartIndex, 1), shuffled.size() - 1);
            
            List<DigitSample> innerTrain = new ArrayList<>(shuffled.subList(0, validationStartIndex));
            List<DigitSample> validationSet = new ArrayList<>(shuffled.subList(validationStartIndex, shuffled.size()));
            
            LinearSVM candidate = new LinearSVM(lambda, epochs, minLearningRate);
            candidate.train(innerTrain);
            double accuracy = candidate.evaluate(validationSet);
            accuracySum += accuracy;
        }
        
        return accuracySum / SVM_VALIDATION_REPEATS;
    }
    
    private static class SVMTrainingResult {
        final Classifier classifier;
        final String hyperparameterSummary;
        final double validationAccuracy;
        
        SVMTrainingResult(Classifier classifier, String hyperparameterSummary, double validationAccuracy) {
            this.classifier = classifier;
            this.hyperparameterSummary = hyperparameterSummary;
            this.validationAccuracy = validationAccuracy;
        }
    }
    
    private static class HyperparameterCandidate implements Comparable<HyperparameterCandidate> {
        final double lambda;
        final int epochs;
        final double minLearningRate;
        final double validationAccuracy;
        
        HyperparameterCandidate(double lambda, int epochs, double minLearningRate, double validationAccuracy) {
            this.lambda = lambda;
            this.epochs = epochs;
            this.minLearningRate = minLearningRate;
            this.validationAccuracy = validationAccuracy;
        }
        
        @Override
        public int compareTo(HyperparameterCandidate other) {
            return Double.compare(other.validationAccuracy, this.validationAccuracy);
        }
        
        String describe() {
            return String.format("λ=%.5f, epochs=%d, η_min=%.1e (val %.2f%%)", 
                                 lambda, epochs, minLearningRate, validationAccuracy);
        }
    }
}

// ============================================================================
// MODEL CLASSES
// ============================================================================

/**
 * Data structure to represent a single digit sample with features and label.
 * Each sample contains 64 features (8x8 pixel values) and a digit label (0-9).
 */
class DigitSample {
    private final double[] features;
    private final int label;
    
    /**
     * Creates a new DigitSample with the given features and label.
     * 
     * @param features Array of 64 pixel values representing the digit
     * @param label The actual digit (0-9) this sample represents
     */
    public DigitSample(double[] features, int label) {
        this.features = features.clone(); // Defensive copy
        this.label = label;
    }
    
    /**
     * Gets the feature vector for this sample.
     * 
     * @return A copy of the feature array to maintain immutability
     */
    public double[] getFeatures() {
        return features.clone();
    }
    
    /**
     * Gets the label for this sample.
     * 
     * @return The digit label (0-9)
     */
    public int getLabel() {
        return label;
    }
    
    /**
     * Gets the number of features in this sample.
     * 
     * @return The feature vector length (should be 64)
     */
    public int getFeatureCount() {
        return features.length;
    }
    
    @Override
    public String toString() {
        return String.format("DigitSample{label=%d, features=%d}", label, features.length);
    }
}

/**
 * Data structure to hold the results of a classification operation.
 * Contains the predicted label, confidence score, and optional additional metrics.
 */
class ClassificationResult {
    private final int predictedLabel;
    private final double confidence;
    private final double[] distances; // Optional: for analysis purposes
    
    /**
     * Creates a classification result with predicted label and confidence.
     * 
     * @param predictedLabel The predicted digit (0-9)
     * @param confidence The confidence score of the prediction (0.0 to 1.0)
     */
    public ClassificationResult(int predictedLabel, double confidence) {
        this.predictedLabel = predictedLabel;
        this.confidence = confidence;
        this.distances = null;
    }
    
    /**
     * Creates a classification result with predicted label, confidence, and distance information.
     * 
     * @param predictedLabel The predicted digit (0-9)
     * @param confidence The confidence score of the prediction (0.0 to 1.0)
     * @param distances Array of distances to neighbors (for analysis)
     */
    public ClassificationResult(int predictedLabel, double confidence, double[] distances) {
        this.predictedLabel = predictedLabel;
        this.confidence = confidence;
        this.distances = distances != null ? distances.clone() : null;
    }
    
    /**
     * Gets the predicted label.
     * 
     * @return The predicted digit (0-9)
     */
    public int getPredictedLabel() {
        return predictedLabel;
    }
    
    /**
     * Gets the confidence score.
     * 
     * @return Confidence value between 0.0 and 1.0
     */
    public double getConfidence() {
        return confidence;
    }
    
    /**
     * Gets the distance information if available.
     * 
     * @return Array of distances or null if not available
     */
    public double[] getDistances() {
        return distances != null ? distances.clone() : null;
    }
    
    /**
     * Checks if distance information is available.
     * 
     * @return true if distances are available, false otherwise
     */
    public boolean hasDistances() {
        return distances != null;
    }
    
    @Override
    public String toString() {
        return String.format("ClassificationResult{predicted=%d, confidence=%.3f}", 
                           predictedLabel, confidence);
    }
}

/**
 * Helper class to store neighbor information for k-NN algorithms.
 * Contains the label of a neighbor and its distance from the query point.
 * Implements Comparable to enable sorting by distance.
 */
class NeighborDistance implements Comparable<NeighborDistance> {
    private final int label;
    private final double distance;
    
    /**
     * Creates a new NeighborDistance with the given label and distance.
     * 
     * @param label The class label of the neighbor
     * @param distance The distance from the query point
     */
    public NeighborDistance(int label, double distance) {
        this.label = label;
        this.distance = distance;
    }
    
    /**
     * Gets the label of this neighbor.
     * 
     * @return The class label
     */
    public int getLabel() {
        return label;
    }
    
    /**
     * Gets the distance of this neighbor from the query point.
     * 
     * @return The distance value
     */
    public double getDistance() {
        return distance;
    }
    
    /**
     * Compares this neighbor with another based on distance.
     * Used for sorting neighbors by distance in ascending order.
     * 
     * @param other The other NeighborDistance to compare with
     * @return Negative if this distance is smaller, positive if larger, 0 if equal
     */
    @Override
    public int compareTo(NeighborDistance other) {
        return Double.compare(this.distance, other.distance);
    }
    
    @Override
    public String toString() {
        return String.format("NeighborDistance{label=%d, distance=%.3f}", label, distance);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        NeighborDistance that = (NeighborDistance) obj;
        return label == that.label && 
               Double.compare(that.distance, distance) == 0;
    }
    
    @Override
    public int hashCode() {
        int result = label;
        long temp = Double.doubleToLongBits(distance);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        return result;
    }
}

// ============================================================================
// ALGORITHM CLASSES
// ============================================================================

/**
 * Interface for digit classification algorithms.
 * Defines the contract that all classification algorithms must implement.
 */
interface Classifier {
    
    /**
     * Trains the classifier on the given training dataset.
     * 
     * @param trainingData List of training samples
     */
    void train(List<DigitSample> trainingData);
    
    /**
     * Classifies a single test sample.
     * 
     * @param testSample The sample to classify
     * @return Classification result with predicted label and confidence
     */
    ClassificationResult classify(DigitSample testSample);
    
    /**
     * Evaluates the classifier on a test dataset.
     * 
     * @param testData List of test samples
     * @return Accuracy percentage (0.0 to 100.0)
     */
    default double evaluate(List<DigitSample> testData) {
        int correct = 0;
        
        for (DigitSample testSample : testData) {
            ClassificationResult result = classify(testSample);
            if (result.getPredictedLabel() == testSample.getLabel()) {
                correct++;
            }
        }
        
        return (correct * 100.0) / testData.size();
    }
    
    /**
     * Gets the name of this classification algorithm.
     * 
     * @return Algorithm name for display purposes
     */
    String getAlgorithmName();
}

/**
 * Nearest Neighbor classifier (k=1) implementation.
 * This is a specialized version of k-NN for the baseline case.
 */
class NearestNeighbor implements Classifier {
    private final KNearestNeighbors knnClassifier;
    
    /**
     * Creates a Nearest Neighbor classifier (k=1).
     */
    public NearestNeighbor() {
        this.knnClassifier = new KNearestNeighbors(1, false);
    }
    
    @Override
    public void train(List<DigitSample> trainingData) {
        knnClassifier.train(trainingData);
    }
    
    @Override
    public ClassificationResult classify(DigitSample testSample) {
        return knnClassifier.classify(testSample);
    }
    
    @Override
    public String getAlgorithmName() {
        return "Nearest Neighbor (k=1)";
    }
}

/**
 * k-Nearest Neighbors classifier implementation.
 * Supports both majority voting and distance-weighted voting.
 */
class KNearestNeighbors implements Classifier {
    private static final int NUM_CLASSES = 10;
    
    private final int k;
    private final boolean weighted;
    private List<DigitSample> trainingData;
    
    /**
     * Creates a k-NN classifier with the specified parameters.
     * 
     * @param k Number of neighbors to consider
     * @param weighted Whether to use distance-weighted voting
     */
    public KNearestNeighbors(int k, boolean weighted) {
        this.k = k;
        this.weighted = weighted;
    }
    
    /**
     * Creates a k-NN classifier with majority voting (non-weighted).
     * 
     * @param k Number of neighbors to consider
     */
    public KNearestNeighbors(int k) {
        this(k, false);
    }
    
    @Override
    public void train(List<DigitSample> trainingData) {
        // k-NN is a lazy learning algorithm - just store the training data
        this.trainingData = new ArrayList<>(trainingData);
    }
    
    @Override
    public ClassificationResult classify(DigitSample testSample) {
        if (trainingData == null || trainingData.isEmpty()) {
            throw new IllegalStateException("Classifier must be trained before classification");
        }
        
        // Calculate distances to all training samples
        List<NeighborDistance> distances = new ArrayList<>();
        
        for (DigitSample trainSample : trainingData) {
            double distance = DistanceCalculator.euclideanDistance(
                testSample.getFeatures(), trainSample.getFeatures());
            distances.add(new NeighborDistance(trainSample.getLabel(), distance));
        }
        
        // Sort by distance and take k nearest neighbors
        Collections.sort(distances);
        List<NeighborDistance> kNearest = distances.subList(0, Math.min(k, distances.size()));
        
        // Vote for class label
        if (weighted) {
            return weightedVote(kNearest);
        } else {
            return majorityVote(kNearest);
        }
    }
    
    /**
     * Performs majority voting among k nearest neighbors.
     */
    private ClassificationResult majorityVote(List<NeighborDistance> neighbors) {
        int[] votes = new int[NUM_CLASSES];
        
        for (NeighborDistance neighbor : neighbors) {
            votes[neighbor.getLabel()]++;
        }
        
        int maxVotes = 0;
        int predictedLabel = 0;
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (votes[i] > maxVotes) {
                maxVotes = votes[i];
                predictedLabel = i;
            }
        }
        
        double confidence = (double) maxVotes / neighbors.size();
        return new ClassificationResult(predictedLabel, confidence);
    }
    
    /**
     * Performs distance-weighted voting among k nearest neighbors.
     */
    private ClassificationResult weightedVote(List<NeighborDistance> neighbors) {
        double[] weights = new double[NUM_CLASSES];
        
        for (NeighborDistance neighbor : neighbors) {
            // Use inverse distance as weight (add small epsilon to avoid division by zero)
            double weight = 1.0 / (neighbor.getDistance() + 1e-8);
            weights[neighbor.getLabel()] += weight;
        }
        
        double maxWeight = 0;
        int predictedLabel = 0;
        double totalWeight = 0;
        
        for (int i = 0; i < NUM_CLASSES; i++) {
            totalWeight += weights[i];
            if (weights[i] > maxWeight) {
                maxWeight = weights[i];
                predictedLabel = i;
            }
        }
        
        double confidence = maxWeight / totalWeight;
        return new ClassificationResult(predictedLabel, confidence);
    }
    
    @Override
    public String getAlgorithmName() {
        if (weighted) {
            return String.format("Weighted k-NN (k=%d)", k);
        } else {
            return String.format("k-NN (k=%d)", k);
        }
    }
    
    /**
     * Gets the k value used by this classifier.
     * 
     * @return The number of neighbors considered
     */
    public int getK() {
        return k;
    }
    
    /**
     * Checks if this classifier uses weighted voting.
     * 
     * @return true if weighted, false for majority voting
     */
    public boolean isWeighted() {
        return weighted;
    }
}

/**
 * Linear Support Vector Machine (SVM) classifier implementation.
 * Uses one-vs-one (OvO) approach for multiclass classification.
 * Implements soft-margin SVM with gradient descent optimization.
 * 
 * This is a simplified but functional implementation of linear SVM.
 * For multiclass problems, it trains one binary classifier per pair
 * of classes (45 pairs for 10 classes), then uses voting to determine
 * the final classification.
 * 
 * @author Dumitru Nirca
 */
class LinearSVM implements Classifier {
    private static final int NUM_CLASSES = 10;
    public static final double DEFAULT_REGULARIZATION_PARAMETER = 0.001;
    public static final int DEFAULT_MAX_EPOCHS = 50;
    public static final double DEFAULT_MIN_LEARNING_RATE = 1e-7;
    private static final boolean ENABLE_SPATIAL_AUGMENTATION = true;
    private static final boolean ENABLE_RANDOM_FOURIER_FEATURES = true;
    private static final boolean ENABLE_POLYNOMIAL_FEATURES = true;
    private static final int RANDOM_FOURIER_FEATURE_COUNT = 512; // Increased for >98% accuracy target
    private static final double RANDOM_FOURIER_GAMMA = 0.012;
    private static final long RANDOM_FOURIER_SEED = 1337L;
    private static final int POLYNOMIAL_DEGREE = 2;
    private static final boolean USE_CLASS_WEIGHTING = true;
    private static final double MAX_CLASS_WEIGHT = 5.0; // Maximum class weight multiplier to avoid extreme weights
    private static final int POLYNOMIAL_NEARBY_FEATURE_RANGE = 12; // Range for local feature interactions
    private static final double EARLY_STOPPING_UPDATE_THRESHOLD = 0.0003; // Stop if updates < 0.03% of samples
    private static final double NORMALIZATION_EPSILON = 1e-12; // Epsilon for avoiding division by zero in normalization
    
    private final double regularizationParameter;
    private final int maxEpochs;
    private final double minLearningRate;
    
    // One-vs-one strategy: store weight vectors for each pair
    // For 10 classes, we have 45 pairs (10 choose 2)
    // Store as: pairIndex -> (weightVector, bias)
    // Pair (i,j) where i < j is stored at index = i * (2*NUM_CLASSES - i - 1) / 2 + j - i - 1
    private double[][] weightVectors; // [pairIndex][feature]
    private double[] biasTerms; // [pairIndex]
    private int[][] pairMapping; // Maps (class1, class2) to pairIndex
    private static final int NUM_PAIRS = (NUM_CLASSES * (NUM_CLASSES - 1)) / 2; // 45 pairs
    private int numFeatures;
    private int baseFeatureCount;
    private int rawFeatureCount;
    private int gridSize;
    private boolean useSpatialAugmentation;
    private boolean useRandomFourierFeatures;
    private double[][] randomFourierWeights;
    private double[] randomFourierBiases;
    private int randomFourierFeatureCount;
    private double randomFourierGamma;
    private boolean isTrained;
    
    /**
     * Creates a new Linear SVM classifier.
     */
    public LinearSVM() {
        this(DEFAULT_REGULARIZATION_PARAMETER, DEFAULT_MAX_EPOCHS, DEFAULT_MIN_LEARNING_RATE);
    }
    
    public LinearSVM(double regularizationParameter, int maxEpochs, double minLearningRate) {
        this.regularizationParameter = regularizationParameter;
        this.maxEpochs = maxEpochs;
        this.minLearningRate = minLearningRate;
        this.isTrained = false;
    }
    
    @Override
    public void train(List<DigitSample> trainingData) {
        if (trainingData == null || trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        rawFeatureCount = trainingData.get(0).getFeatureCount();
        gridSize = (int) Math.round(Math.sqrt(rawFeatureCount));
        useSpatialAugmentation = ENABLE_SPATIAL_AUGMENTATION && gridSize * gridSize == rawFeatureCount;
        baseFeatureCount = useSpatialAugmentation ? rawFeatureCount + 2 * gridSize : rawFeatureCount;
        useRandomFourierFeatures = ENABLE_RANDOM_FOURIER_FEATURES;
        randomFourierFeatureCount = useRandomFourierFeatures ? RANDOM_FOURIER_FEATURE_COUNT : 0;
        randomFourierGamma = RANDOM_FOURIER_GAMMA;
        initializeRandomFourierFeatures();
        int polynomialFeatureCount = ENABLE_POLYNOMIAL_FEATURES ? computePolynomialFeatureCount(baseFeatureCount) : 0;
        numFeatures = baseFeatureCount + randomFourierFeatureCount + polynomialFeatureCount;
        weightVectors = new double[NUM_PAIRS][numFeatures];
        biasTerms = new double[NUM_PAIRS];
        initializePairMapping();
        
        // Normalize features using z-score normalization (standardization) for better SVM performance
        double[] featureMeans = computeFeatureMeans(trainingData);
        double[] featureStds = computeFeatureStdDeviations(trainingData, featureMeans);
        storedMeans = featureMeans;
        storedStds = featureStds;
        List<DigitSample> normalizedData = normalizeFeaturesZScore(trainingData, featureMeans, featureStds);
        
        // Train one binary classifier per pair (one-vs-one)
        int pairIndex = 0;
        for (int class1 = 0; class1 < NUM_CLASSES; class1++) {
            for (int class2 = class1 + 1; class2 < NUM_CLASSES; class2++) {
                trainBinaryClassifier(normalizedData, class1, class2, pairIndex);
                pairIndex++;
            }
        }
        
        this.isTrained = true;
    }
    
    /**
     * Initializes the mapping from (class1, class2) to pair index.
     * For 10 classes, pairs are: (0,1), (0,2), ..., (0,9), (1,2), ..., (8,9)
     * Total: 45 pairs
     */
    private void initializePairMapping() {
        pairMapping = new int[NUM_CLASSES][NUM_CLASSES];
        int pairIndex = 0;
        for (int class1 = 0; class1 < NUM_CLASSES; class1++) {
            for (int class2 = class1 + 1; class2 < NUM_CLASSES; class2++) {
                pairMapping[class1][class2] = pairIndex;
                pairMapping[class2][class1] = pairIndex; // Symmetric
                pairIndex++;
            }
        }
    }
    
    /**
     * Gets the pair index for two classes.
     */
    private int getPairIndex(int class1, int class2) {
        if (class1 > class2) {
            int temp = class1;
            class1 = class2;
            class2 = temp;
        }
        return pairMapping[class1][class2];
    }
    
    /**
     * Trains a binary classifier for one pair (one-vs-one).
     * Uses Pegasos-style stochastic gradient descent for SVM optimization.
     * This is a proven algorithm for linear SVMs that converges well.
     */
    private void trainBinaryClassifier(List<DigitSample> trainingData, int class1, int class2, int pairIndex) {
        // Filter training data to only include samples from class1 or class2
        List<DigitSample> pairData = new ArrayList<>();
        for (DigitSample sample : trainingData) {
            int label = sample.getLabel();
            if (label == class1 || label == class2) {
                pairData.add(sample);
            }
        }
        
        // Compute class weights for balanced learning
        int class1Count = 0;
        int class2Count = 0;
        for (DigitSample sample : pairData) {
            if (sample.getLabel() == class1) {
                class1Count++;
            } else {
                class2Count++;
            }
        }
        double classWeight = USE_CLASS_WEIGHTING && class1Count > 0 && class2Count > 0 ?
            (double) class2Count / class1Count : 1.0;
        classWeight = Math.min(classWeight, MAX_CLASS_WEIGHT);
        
        // Initialize weights to zero
        double[] weights = new double[numFeatures];
        double bias = 0.0;
        double[] cumulativeWeights = new double[numFeatures];
        double cumulativeBias = 0.0;
        int snapshotCount = 0;
        
        // Prepare data
        List<DigitSample> shuffledData = new ArrayList<>(pairData);
        int numSamples = pairData.size();
        int globalIteration = 0;
        
        // Pegasos-style training with adaptive learning rate (per-update schedule)
        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            Collections.shuffle(shuffledData);
            int numUpdates = 0;
            
            // One epoch: process all samples
            for (DigitSample sample : shuffledData) {
                globalIteration++;
                double learningRate = 1.0 / (regularizationParameter * globalIteration);
                learningRate = Math.max(minLearningRate, learningRate); // Clamp minimum
                
                double[] features = sample.getFeatures();
                int label = sample.getLabel();
                int binaryLabel = (label == class1) ? 1 : -1; // class1 = +1, class2 = -1
                
                // Apply class weight: positive class gets higher weight if it's minority
                double sampleWeight = (binaryLabel == 1) ? classWeight : 1.0;
                
                // Compute margin
                double margin = binaryLabel * computeDecisionValue(features, weights, bias);
                
                // Hinge loss subgradient update
                if (margin < 1.0) {
                    numUpdates++;
                    
                    // Step 1: Gradient step for hinge loss with class weighting
                    // w = (1 - η*λ) * w + η * weight * y * x (when margin < 1)
                    double scale = 1.0 - learningRate * regularizationParameter;
                    double weightedLR = learningRate * sampleWeight;
                    for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                        weights[featureIndex] = scale * weights[featureIndex] + 
                                               weightedLR * binaryLabel * features[featureIndex];
                    }
                    bias = scale * bias + weightedLR * binaryLabel;
                } else {
                    // Step 2: Regularization step (when margin >= 1)
                    // Just apply weight decay: w = (1 - η*λ) * w
                    double scale = 1.0 - learningRate * regularizationParameter;
                    for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                        weights[featureIndex] *= scale;
                    }
                    bias *= scale;
                }
                
                // Projection step: scale weights to ensure ||w|| <= 1/sqrt(λ)
                // This is optional but can help stability
                double weightNorm = 0.0;
                for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                    weightNorm += weights[featureIndex] * weights[featureIndex];
                }
                weightNorm = Math.sqrt(weightNorm);
                double maxNorm = 1.0 / Math.sqrt(regularizationParameter);
                if (weightNorm > maxNorm) {
                    double scale = maxNorm / weightNorm;
                    for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                        weights[featureIndex] *= scale;
                    }
                }
            }
            
            // Early stopping: if very few updates, model has converged
            // More lenient threshold to allow longer training for better convergence
            if (numUpdates < numSamples * EARLY_STOPPING_UPDATE_THRESHOLD && epoch > 10) {
                break;
            }
            
            for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                cumulativeWeights[featureIndex] += weights[featureIndex];
            }
            cumulativeBias += bias;
            snapshotCount++;
        }
        
        if (snapshotCount > 0) {
            for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                weights[featureIndex] = cumulativeWeights[featureIndex] / snapshotCount;
            }
            bias = cumulativeBias / snapshotCount;
        }
        
        weightVectors[pairIndex] = weights;
        biasTerms[pairIndex] = bias;
    }
    
    /**
     * Computes the decision value for a feature vector.
     */
    private double computeDecisionValue(double[] features, double[] weights, double bias) {
        double result = bias;
        for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            result += weights[featureIndex] * features[featureIndex];
        }
        return result;
    }
    
    /**
     * Computes mean values for each feature (for z-score normalization).
     */
    private double[] computeFeatureMeans(List<DigitSample> trainingData) {
        double[] means = new double[numFeatures];
        
        for (DigitSample sample : trainingData) {
            double[] features = getAugmentedFeatures(sample.getFeatures());
            for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                means[featureIndex] += features[featureIndex];
            }
        }
        
        int numSamples = trainingData.size();
        for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            means[featureIndex] /= numSamples;
        }
        
        return means;
    }
    
    /**
     * Computes standard deviation for each feature (for z-score normalization).
     */
    private double[] computeFeatureStdDeviations(List<DigitSample> trainingData, double[] means) {
        double[] variances = new double[numFeatures];
        
        for (DigitSample sample : trainingData) {
            double[] features = getAugmentedFeatures(sample.getFeatures());
            for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                double diff = features[featureIndex] - means[featureIndex];
                variances[featureIndex] += diff * diff;
            }
        }
        
        int numSamples = trainingData.size();
        double[] stds = new double[numFeatures];
        for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            variances[featureIndex] /= numSamples;
            stds[featureIndex] = Math.sqrt(variances[featureIndex]);
            // Add small epsilon to avoid division by zero
            if (stds[featureIndex] < 1e-8) {
                stds[featureIndex] = 1.0;
            }
        }
        
        return stds;
    }
    
    /**
     * Normalizes features using z-score normalization (standardization).
     * Transforms features to have mean=0 and std=1.
     */
    private List<DigitSample> normalizeFeaturesZScore(List<DigitSample> trainingData, 
                                                       double[] means, double[] stds) {
        List<DigitSample> normalizedData = new ArrayList<>();
        
        for (DigitSample sample : trainingData) {
            double[] augmentedFeatures = getAugmentedFeatures(sample.getFeatures());
            double[] normalizedFeatures = new double[numFeatures];
            
            for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
                normalizedFeatures[featureIndex] = 
                    (augmentedFeatures[featureIndex] - means[featureIndex]) / stds[featureIndex];
            }
            normalizeVectorInPlace(normalizedFeatures);
            
            normalizedData.add(new DigitSample(normalizedFeatures, sample.getLabel()));
        }
        
        return normalizedData;
    }
    
    // Store normalization parameters for test data
    private double[] storedMeans;
    private double[] storedStds;
    
    @Override
    public ClassificationResult classify(DigitSample testSample) {
        if (!isTrained) {
            throw new IllegalStateException("Classifier must be trained before classification");
        }
        
        // Normalize test sample using training statistics (z-score normalization)
        double[] features = testSample.getFeatures();
        double[] augmentedFeatures = getAugmentedFeatures(features);
        double[] normalizedFeatures = new double[numFeatures];
        
        for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            normalizedFeatures[featureIndex] = 
                (augmentedFeatures[featureIndex] - storedMeans[featureIndex]) / storedStds[featureIndex];
        }
        normalizeVectorInPlace(normalizedFeatures);
        
        // One-vs-one: Run all pairwise classifiers and count votes
        int[] votes = new int[NUM_CLASSES];
        double[] confidences = new double[NUM_CLASSES];
        
        for (int class1 = 0; class1 < NUM_CLASSES; class1++) {
            for (int class2 = class1 + 1; class2 < NUM_CLASSES; class2++) {
                int pairIdx = getPairIndex(class1, class2);
                double decisionValue = computeDecisionValue(normalizedFeatures,
                                                           weightVectors[pairIdx],
                                                           biasTerms[pairIdx]);
                
                // Positive decision value means class1 wins, negative means class2 wins
                if (decisionValue > 0) {
                    votes[class1]++;
                    confidences[class1] += Math.max(0, decisionValue);
                } else {
                    votes[class2]++;
                    confidences[class2] += Math.max(0, -decisionValue);
                }
            }
        }
        
        // Find class with most votes (one-vs-one voting)
        int predictedClass = 0;
        int maxVotes = votes[0];
        for (int classLabel = 1; classLabel < NUM_CLASSES; classLabel++) {
            if (votes[classLabel] > maxVotes || 
                (votes[classLabel] == maxVotes && confidences[classLabel] > confidences[predictedClass])) {
                maxVotes = votes[classLabel];
                predictedClass = classLabel;
            }
        }
        
        // Convert votes to confidence
        double totalVotes = NUM_CLASSES - 1; // Each class participates in 9 comparisons
        double confidence = (double) votes[predictedClass] / totalVotes;
        
        return new ClassificationResult(predictedClass, confidence);
    }
    
    @Override
    public String getAlgorithmName() {
        return "Linear SVM";
    }
    
    public String getHyperparameterSummary() {
        return String.format("λ=%.5f, epochs=%d, η_min=%.1e", 
                             regularizationParameter, maxEpochs, minLearningRate);
    }
    
    private double[] getAugmentedFeatures(double[] originalFeatures) {
        double[] baseFeatures = buildBaseFeatures(originalFeatures);
        int offset = 0;
        double[] augmented = new double[numFeatures];
        
        // Base features (original + spatial)
        System.arraycopy(baseFeatures, 0, augmented, offset, baseFeatureCount);
        offset += baseFeatureCount;
        
        // Random Fourier features
        if (useRandomFourierFeatures) {
            double[] randomFeatures = computeRandomFourierFeatures(originalFeatures);
            System.arraycopy(randomFeatures, 0, augmented, offset, randomFourierFeatureCount);
            offset += randomFourierFeatureCount;
        }
        
        // Polynomial features (degree 2)
        if (ENABLE_POLYNOMIAL_FEATURES) {
            double[] polyFeatures = computePolynomialFeatures(baseFeatures);
            System.arraycopy(polyFeatures, 0, augmented, offset, polyFeatures.length);
        }
        
        return augmented;
    }
    
    private int computePolynomialFeatureCount(int baseCount) {
        // For degree 2: include all pairwise products (x_i * x_j where i <= j)
        // This gives us baseCount + baseCount*(baseCount+1)/2 features
        // But to keep it manageable, we'll use a subset: top features only
        int maxPolyFeatures = Math.min(896, baseCount * (baseCount + 1) / 2); // Increased for >98% accuracy
        return maxPolyFeatures;
    }
    
    private double[] computePolynomialFeatures(double[] baseFeatures) {
        // For degree 2 polynomial: include x_i^2 and selected x_i * x_j products
        // We'll use a sampling strategy to keep feature count manageable
        int polyCount = computePolynomialFeatureCount(baseFeatureCount);
        double[] polyFeatures = new double[polyCount];
        int idx = 0;
        
        // Add squared features (x_i^2)
        for (int i = 0; i < Math.min(baseFeatureCount, polyCount / 2); i++) {
            polyFeatures[idx++] = baseFeatures[i] * baseFeatures[i];
        }
        
        // Add selected pairwise products (x_i * x_j) for nearby features
        // This captures local interactions which are important for images
        // First pass: nearby features (local interactions)
        for (int i = 0; i < baseFeatureCount && idx < polyCount; i++) {
            for (int j = i + 1; j < Math.min(i + POLYNOMIAL_NEARBY_FEATURE_RANGE, baseFeatureCount) && idx < polyCount; j++) {
                polyFeatures[idx++] = baseFeatures[i] * baseFeatures[j];
            }
        }
        // Second pass: sample more distant features (global interactions)
        if (idx < polyCount) {
            int step = Math.max(1, baseFeatureCount / 20);
            for (int i = 0; i < baseFeatureCount && idx < polyCount; i += step) {
                for (int j = i + step; j < baseFeatureCount && idx < polyCount; j += step) {
                    polyFeatures[idx++] = baseFeatures[i] * baseFeatures[j];
                }
            }
        }
        
        return polyFeatures;
    }
    
    private double[] buildBaseFeatures(double[] originalFeatures) {
        if (!useSpatialAugmentation) {
            return originalFeatures.clone();
        }
        
        double[] base = new double[baseFeatureCount];
        System.arraycopy(originalFeatures, 0, base, 0, rawFeatureCount);
        int offset = rawFeatureCount;
        
        for (int row = 0; row < gridSize; row++) {
            double sum = 0.0;
            for (int col = 0; col < gridSize; col++) {
                sum += originalFeatures[row * gridSize + col];
            }
            base[offset + row] = sum / gridSize;
        }
        offset += gridSize;
        
        for (int col = 0; col < gridSize; col++) {
            double sum = 0.0;
            for (int row = 0; row < gridSize; row++) {
                sum += originalFeatures[row * gridSize + col];
            }
            base[offset + col] = sum / gridSize;
        }
        
        return base;
    }
    
    private void initializeRandomFourierFeatures() {
        if (!useRandomFourierFeatures || randomFourierFeatureCount == 0) {
            randomFourierWeights = null;
            randomFourierBiases = null;
            return;
        }
        
        randomFourierWeights = new double[randomFourierFeatureCount][rawFeatureCount];
        randomFourierBiases = new double[randomFourierFeatureCount];
        Random random = new Random(RANDOM_FOURIER_SEED);
        double stdDev = Math.sqrt(2.0 * randomFourierGamma);
        
        for (int featureIndex = 0; featureIndex < randomFourierFeatureCount; featureIndex++) {
            for (int dimension = 0; dimension < rawFeatureCount; dimension++) {
                randomFourierWeights[featureIndex][dimension] = random.nextGaussian() * stdDev;
            }
            randomFourierBiases[featureIndex] = random.nextDouble() * 2.0 * Math.PI;
        }
    }
    
    private double[] computeRandomFourierFeatures(double[] originalFeatures) {
        double[] projected = new double[randomFourierFeatureCount];
        double scale = Math.sqrt(2.0 / randomFourierFeatureCount);
        
        for (int featureIndex = 0; featureIndex < randomFourierFeatureCount; featureIndex++) {
            double dotProduct = randomFourierBiases[featureIndex];
            double[] weights = randomFourierWeights[featureIndex];
            
            for (int dimension = 0; dimension < rawFeatureCount; dimension++) {
                dotProduct += weights[dimension] * originalFeatures[dimension];
            }
            projected[featureIndex] = scale * Math.cos(dotProduct);
        }
        
        return projected;
    }
    
    private void normalizeVectorInPlace(double[] vector) {
        double norm = 0.0;
        for (double value : vector) {
            norm += value * value;
        }
        norm = Math.sqrt(norm);
        if (norm > NORMALIZATION_EPSILON) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }
}

/**
 * Simple ensemble wrapper that combines multiple independently trained Linear SVMs
 * by confidence-weighted voting.
 */
class LinearSVMEnsemble implements Classifier {
    private static final int NUM_CLASSES = 10;
    private final List<LinearSVM> models;
    
    public LinearSVMEnsemble(List<LinearSVM> models) {
        if (models == null || models.isEmpty()) {
            throw new IllegalArgumentException("Ensemble must contain at least one model");
        }
        this.models = new ArrayList<>(models);
    }
    
    @Override
    public void train(List<DigitSample> trainingData) {
        throw new UnsupportedOperationException("LinearSVMEnsemble expects pre-trained models");
    }
    
    @Override
    public ClassificationResult classify(DigitSample testSample) {
        double[] classVotes = new double[NUM_CLASSES];
        double totalVote = 0.0;
        
        for (LinearSVM model : models) {
            ClassificationResult result = model.classify(testSample);
            double confidence = Math.max(1e-6, result.getConfidence());
            classVotes[result.getPredictedLabel()] += confidence;
            totalVote += confidence;
        }
        
        int predictedLabel = 0;
        double bestVote = classVotes[0];
        for (int classIndex = 1; classIndex < NUM_CLASSES; classIndex++) {
            if (classVotes[classIndex] > bestVote) {
                bestVote = classVotes[classIndex];
                predictedLabel = classIndex;
            }
        }
        
        double confidence = totalVote > 0 ? bestVote / totalVote : 0.0;
        return new ClassificationResult(predictedLabel, confidence);
    }
    
    @Override
    public String getAlgorithmName() {
        return String.format("Linear SVM Ensemble (%d models)", models.size());
    }
}

// ============================================================================
// UTILITY CLASSES
// ============================================================================

/**
 * Utility class for loading and managing datasets.
 * Handles CSV file parsing and dataset operations.
 */
class DatasetLoader {
    private static final int NUM_FEATURES = 64; // 8x8 pixel values
    private static final int NUM_DIGIT_CLASSES = 10; // Digit classes 0-9
    
    /**
     * Private constructor to prevent instantiation of utility class.
     */
    private DatasetLoader() {
        throw new UnsupportedOperationException("Utility class cannot be instantiated");
    }
    
    /**
     * Loads a dataset from a CSV file.
     * 
     * @param filename Path to the CSV file (relative to resources or absolute)
     * @return List of DigitSample objects
     * @throws IOException if file cannot be read
     */
    public static List<DigitSample> loadDataset(String filename) throws IOException {
        List<DigitSample> samples = new ArrayList<>();
        
        // Try to load from resources first, then as absolute path
        InputStream inputStream = DatasetLoader.class.getClassLoader().getResourceAsStream(filename);
        if (inputStream == null) {
            // Try as file path
            inputStream = new FileInputStream(filename);
        }
        
        try (BufferedReader br = new BufferedReader(new InputStreamReader(inputStream))) {
            String line;
            int lineNumber = 0;
            
            while ((line = br.readLine()) != null) {
                lineNumber++;
                try {
                    DigitSample sample = parseLine(line);
                    samples.add(sample);
                } catch (Exception exception) {
                    throw new IOException("Error parsing line " + lineNumber + ": " + exception.getMessage(), exception);
                }
            }
        }
        
        return samples;
    }
    
    /**
     * Parses a single line from the CSV file.
     * 
     * @param line CSV line to parse
     * @return DigitSample object
     * @throws IllegalArgumentException if line format is invalid
     */
    private static DigitSample parseLine(String line) {
        String[] parts = line.trim().split(",");
        
        if (parts.length != NUM_FEATURES + 1) {
            throw new IllegalArgumentException(
                String.format("Expected %d values (64 features + 1 label), found %d", 
                             NUM_FEATURES + 1, parts.length));
        }
        
        // Extract features (first 64 values)
        double[] features = new double[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; i++) {
            try {
                features[i] = Double.parseDouble(parts[i]);
            } catch (NumberFormatException exception) {
                throw new IllegalArgumentException("Invalid feature value at position " + i + ": " + parts[i]);
            }
        }
        
        // Extract label (last value)
        int label;
        try {
            label = Integer.parseInt(parts[NUM_FEATURES]);
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException("Invalid label value: " + parts[NUM_FEATURES]);
        }
        
        if (label < 0 || label > 9) {
            throw new IllegalArgumentException("Label must be between 0 and 9, found: " + label);
        }
        
        return new DigitSample(features, label);
    }
    
    /**
     * Analyzes and returns the class distribution of a dataset.
     * 
     * @param dataset The dataset to analyze
     * @return Array where index i contains the count of samples with label i
     */
    public static int[] getClassDistribution(List<DigitSample> dataset) {
        int[] classCounts = new int[NUM_DIGIT_CLASSES];
        
        for (DigitSample sample : dataset) {
            classCounts[sample.getLabel()]++;
        }
        
        return classCounts;
    }
    
    /**
     * Prints a formatted analysis of the dataset distribution.
     * 
     * @param dataset The dataset to analyze
     * @param datasetName Name for display purposes
     */
    public static void printDatasetAnalysis(List<DigitSample> dataset, String datasetName) {
        int[] classCounts = getClassDistribution(dataset);
        
        System.out.println("\n" + datasetName + " Analysis:");
        System.out.println("Total samples: " + dataset.size());
        System.out.println("Class distribution:");
        
        for (int digitClass = 0; digitClass < NUM_DIGIT_CLASSES; digitClass++) {
            double percentage = (classCounts[digitClass] * 100.0) / dataset.size();
            System.out.printf("  Digit %d: %4d samples (%.1f%%)\n", digitClass, classCounts[digitClass], percentage);
        }
        System.out.println();
    }
}

/**
 * Utility class for calculating distances between feature vectors.
 * Provides various distance metrics commonly used in machine learning.
 */
class DistanceCalculator {
    
    /**
     * Private constructor to prevent instantiation of utility class.
     */
    private DistanceCalculator() {
        throw new UnsupportedOperationException("Utility class cannot be instantiated");
    }
    
    /**
     * Calculates the Euclidean distance between two feature vectors.
     * 
     * @param features1 First feature vector
     * @param features2 Second feature vector
     * @return Euclidean distance between the vectors
     * @throws IllegalArgumentException if vectors have different lengths
     */
    public static double euclideanDistance(double[] features1, double[] features2) {
        if (features1.length != features2.length) {
            throw new IllegalArgumentException("Feature vectors must have the same length");
        }
        
        double sum = 0.0;
        for (int i = 0; i < features1.length; i++) {
            double diff = features1[i] - features2[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
    
    /**
     * Calculates the Manhattan distance between two feature vectors.
     * 
     * @param features1 First feature vector
     * @param features2 Second feature vector
     * @return Manhattan distance between the vectors
     * @throws IllegalArgumentException if vectors have different lengths
     */
    public static double manhattanDistance(double[] features1, double[] features2) {
        if (features1.length != features2.length) {
            throw new IllegalArgumentException("Feature vectors must have the same length");
        }
        
        double sum = 0.0;
        for (int i = 0; i < features1.length; i++) {
            sum += Math.abs(features1[i] - features2[i]);
        }
        return sum;
    }
    
    /**
     * Calculates the Minkowski distance between two feature vectors.
     * 
     * @param features1 First feature vector
     * @param features2 Second feature vector
     * @param p The order of the Minkowski distance (p=1 for Manhattan, p=2 for Euclidean)
     * @return Minkowski distance between the vectors
     * @throws IllegalArgumentException if vectors have different lengths or p <= 0
     */
    public static double minkowskiDistance(double[] features1, double[] features2, double p) {
        if (features1.length != features2.length) {
            throw new IllegalArgumentException("Feature vectors must have the same length");
        }
        if (p <= 0) {
            throw new IllegalArgumentException("p must be positive");
        }
        
        double sum = 0.0;
        for (int i = 0; i < features1.length; i++) {
            sum += Math.pow(Math.abs(features1[i] - features2[i]), p);
        }
        return Math.pow(sum, 1.0 / p);
    }
}

/**
 * Utility class for calculating and displaying evaluation metrics.
 * Provides confusion matrix generation, accuracy calculation, and performance reporting.
 */
class EvaluationMetrics {
    private static final int NUM_CLASSES = 10;
    
    /**
     * Private constructor to prevent instantiation of utility class.
     */
    private EvaluationMetrics() {
        throw new UnsupportedOperationException("Utility class cannot be instantiated");
    }
    
    /**
     * Generates a confusion matrix for a classifier on test data.
     * 
     * @param classifier The trained classifier
     * @param testData The test dataset
     * @return 10x10 confusion matrix where [i][j] is the count of samples 
     *         with true label i predicted as label j
     */
    public static int[][] generateConfusionMatrix(Classifier classifier, List<DigitSample> testData) {
        int[][] confusionMatrix = new int[NUM_CLASSES][NUM_CLASSES];
        
        for (DigitSample testSample : testData) {
            ClassificationResult result = classifier.classify(testSample);
            confusionMatrix[testSample.getLabel()][result.getPredictedLabel()]++;
        }
        
        return confusionMatrix;
    }
    
    /**
     * Prints a formatted confusion matrix with statistics.
     * 
     * @param confusionMatrix The confusion matrix to print
     * @param algorithmName Name of the algorithm for display
     */
    public static void printConfusionMatrix(int[][] confusionMatrix, String algorithmName) {
        System.out.println("\nConfusion Matrix for " + algorithmName + ":");
        System.out.println("True\\Pred   0   1   2   3   4   5   6   7   8   9");
        
        for (int i = 0; i < NUM_CLASSES; i++) {
            System.out.printf("   %d     ", i);
            for (int j = 0; j < NUM_CLASSES; j++) {
                System.out.printf("%4d", confusionMatrix[i][j]);
            }
            System.out.println();
        }
    }
    
    /**
     * Calculates and prints per-class performance metrics.
     * 
     * @param confusionMatrix The confusion matrix
     * @param algorithmName Name of the algorithm for display
     */
    public static void printClassificationReport(int[][] confusionMatrix, String algorithmName) {
        System.out.println("\nPer-class Performance Metrics for " + algorithmName + ":");
        System.out.println("Class | Precision | Recall   | F1-Score | Support");
        System.out.println("------|-----------|--------- |----------|--------");
        
        double totalPrecision = 0, totalRecall = 0, totalF1 = 0;
        int totalSupport = 0;
        
        for (int i = 0; i < NUM_CLASSES; i++) {
            int truePositives = confusionMatrix[i][i];
            int falsePositives = 0;
            int falseNegatives = 0;
            int support = 0;
            
            // Calculate FP, FN, and support
            for (int j = 0; j < NUM_CLASSES; j++) {
                if (i != j) {
                    falsePositives += confusionMatrix[j][i];
                    falseNegatives += confusionMatrix[i][j];
                }
                support += confusionMatrix[i][j];
            }
            
            double precision = (truePositives + falsePositives > 0) ? 
                (double) truePositives / (truePositives + falsePositives) : 0.0;
            double recall = (support > 0) ? (double) truePositives / support : 0.0;
            double f1Score = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
            
            System.out.printf("  %d   |   %.3f   |  %.3f   |  %.3f   |  %4d\n", 
                i, precision, recall, f1Score, support);
            
            totalPrecision += precision;
            totalRecall += recall;
            totalF1 += f1Score;
            totalSupport += support;
        }
        
        // Macro averages
        System.out.println("------|-----------|----------|----------|--------");
        System.out.printf(" Avg  |   %.3f   |  %.3f   |  %.3f   |  %4d\n", 
            totalPrecision / NUM_CLASSES, totalRecall / NUM_CLASSES, totalF1 / NUM_CLASSES, totalSupport);
    }
    
    /**
     * Calculates the overall accuracy from a confusion matrix.
     * 
     * @param confusionMatrix The confusion matrix
     * @return Accuracy percentage (0.0 to 100.0)
     */
    public static double calculateAccuracy(int[][] confusionMatrix) {
        int correct = 0;
        int total = 0;
        
        for (int i = 0; i < NUM_CLASSES; i++) {
            for (int j = 0; j < NUM_CLASSES; j++) {
                if (i == j) {
                    correct += confusionMatrix[i][j];
                }
                total += confusionMatrix[i][j];
            }
        }
        
        return total > 0 ? (correct * 100.0) / total : 0.0;
    }
    
    /**
     * Performs comprehensive evaluation of a classifier.
     * 
     * @param classifier The trained classifier
     * @param testData The test dataset
     * @return Accuracy percentage
     */
    public static double evaluateClassifier(Classifier classifier, List<DigitSample> testData) {
        int[][] confusionMatrix = generateConfusionMatrix(classifier, testData);
        double accuracy = calculateAccuracy(confusionMatrix);
        
        System.out.printf("\n%s Accuracy: %.2f%%\n", classifier.getAlgorithmName(), accuracy);
        printConfusionMatrix(confusionMatrix, classifier.getAlgorithmName());
        printClassificationReport(confusionMatrix, classifier.getAlgorithmName());
        
        return accuracy;
    }
}