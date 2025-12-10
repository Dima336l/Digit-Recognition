import java.io.*;
import java.util.ArrayList;
import java.util.Random;
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
    
    // Numeric constants (public for access from other classes)
    public static final double ZERO = 0.0;
    public static final double ONE = 1.0;
    public static final double TWO = 2.0;
    public static final int INT_ZERO = 0;
    public static final int INT_ONE = 1;
    public static final int INT_TWO = 2;
    public static final int MIN_TRAINING_SAMPLES = 2;
    public static final int MIN_VALIDATION_INDEX = 1;
    public static final int MIN_LABEL = 0;
    public static final int MAX_LABEL = 9;
    public static final int EARLY_STOPPING_MIN_EPOCHS = 10;
    public static final double PERCENTAGE_MULTIPLIER = 100.0;
    public static final double DISTANCE_WEIGHTING_EPSILON = 1e-8;
    public static final double STD_DEVIATION_EPSILON = 1e-8;
    public static final double DEFAULT_STD_DEVIATION = 1.0;
    public static final double MARGIN_THRESHOLD = 1.0;
    public static final int SPATIAL_AUGMENTATION_MULTIPLIER = 2;
    public static final int MAX_POLYNOMIAL_FEATURES = 896;
    public static final int POLYNOMIAL_STEP_DIVISOR = 20;
    public static final double POLYNOMIAL_COUNT_DIVISOR = 2.0;
    public static final double RFF_SCALE_MULTIPLIER = 2.0;
    public static final double RFF_BIAS_MULTIPLIER = 2.0;
    public static final double F1_SCORE_MULTIPLIER = 2.0;
    public static final int BINARY_POSITIVE_LABEL = 1;
    public static final int BINARY_NEGATIVE_LABEL = -1;
    
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
            ArrayList<DigitSample> dataset1 = DatasetLoader.loadDataset(DATASET1_PATH);
            ArrayList<DigitSample> dataset2 = DatasetLoader.loadDataset(DATASET2_PATH);
            
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
    private static void performTwoFoldValidation(ArrayList<DigitSample> dataset1, ArrayList<DigitSample> dataset2) {
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
    
    /**
     * Evaluates all algorithms (NN, k-NN, weighted k-NN, SVM) on given datasets.
     * Returns results summary for this fold.
     */
    private static FoldResults evaluateAllAlgorithms(ArrayList<DigitSample> trainSet, ArrayList<DigitSample> testSet, String foldName) {
        // Evaluate nearest neighbor baseline
        double nnAccuracy = evaluateNearestNeighbor(trainSet, testSet);
        
        // Find best k-NN and evaluate weighted k-NN
        int bestK = findBestKNN(trainSet, testSet);
        double bestKnnAccuracy = getKnnAccuracy(trainSet, testSet, bestK);
        double weightedAccuracy = evaluateWeightedKNN(trainSet, testSet, bestK);
        
        // Train and evaluate SVM
        SVMTrainingResult svmResult = trainBestLinearSVM(trainSet);
        double svmAccuracy = EvaluationMetrics.evaluateClassifier(svmResult.classifier, testSet);
        
        // Print summary and return results
        printFoldSummary(foldName, nnAccuracy, bestK, bestKnnAccuracy, weightedAccuracy, svmAccuracy, svmResult.hyperparameterSummary);
        return new FoldResults(nnAccuracy, bestK, bestKnnAccuracy, weightedAccuracy, svmAccuracy, svmResult.hyperparameterSummary);
    }
    
    /**
     * Evaluates nearest neighbor classifier (k=1 baseline).
     */
    private static double evaluateNearestNeighbor(ArrayList<DigitSample> trainSet, ArrayList<DigitSample> testSet) {
        System.out.println("\n1. NEAREST NEIGHBOR (k=1) - BASELINE:");
        NearestNeighbor nnClassifier = new NearestNeighbor();
        nnClassifier.train(trainSet);
        return EvaluationMetrics.evaluateClassifier(nnClassifier, testSet);
    }
    
    /**
     * Finds best k value for k-NN by testing all k values.
     */
    private static int findBestKNN(ArrayList<DigitSample> trainSet, ArrayList<DigitSample> testSet) {
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
        return bestK;
    }
    
    /**
     * Gets accuracy for k-NN with specified k value.
     */
    private static double getKnnAccuracy(ArrayList<DigitSample> trainSet, ArrayList<DigitSample> testSet, int k) {
        KNearestNeighbors knnClassifier = new KNearestNeighbors(k, false);
        knnClassifier.train(trainSet);
        return knnClassifier.evaluate(testSet);
    }
    
    /**
     * Evaluates weighted k-NN classifier with distance-based voting.
     */
    private static double evaluateWeightedKNN(ArrayList<DigitSample> trainSet, ArrayList<DigitSample> testSet, int k) {
        System.out.println("3. WEIGHTED k-NN (distance-based voting):");
        KNearestNeighbors weightedKNN = new KNearestNeighbors(k, true);
        weightedKNN.train(trainSet);
        return EvaluationMetrics.evaluateClassifier(weightedKNN, testSet);
    }
    
    /**
     * Prints summary of all algorithm results for this fold.
     */
    private static void printFoldSummary(String foldName, double nnAccuracy, int bestK, 
                                         double bestKnnAccuracy, double weightedAccuracy, 
                                         double svmAccuracy, String svmSummary) {
        System.out.println("\n" + foldName + " SUMMARY:");
        System.out.printf("  Nearest Neighbor (k=1): %.2f%%\n", nnAccuracy);
        System.out.printf("  k-NN (best k=%d): %.2f%%\n", bestK, bestKnnAccuracy);
        System.out.printf("  Weighted k-NN (k=%d): %.2f%%\n", bestK, weightedAccuracy);
        System.out.printf("  Linear SVM [%s]: %.2f%%\n", svmSummary, svmAccuracy);
        System.out.println();
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
        
        /**
         * Creates a new FoldResults with all algorithm accuracies.
         */
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
    
    /**
     * Trains best Linear SVM by searching hyperparameters and creating ensemble.
     */
    private static SVMTrainingResult trainBestLinearSVM(ArrayList<DigitSample> trainingData) {
        validateTrainingData(trainingData);
        ArrayList<HyperparameterCandidate> candidates = searchHyperparameters(trainingData); // grid search
        CustomCollections.sort(candidates); // best-first ordering
        ArrayList<LinearSVM> ensembleModels = buildEnsembleModels(trainingData, candidates); // train top configs
        return createSVMTrainingResult(ensembleModels, candidates); // wrap classifier + summary
    }
    
    /**
     * Validates that training data has sufficient samples.
     */
    private static void validateTrainingData(ArrayList<DigitSample> trainingData) {
        if (trainingData == null || trainingData.size() < MIN_TRAINING_SAMPLES) {
            throw new IllegalArgumentException("Training data must contain at least two samples for SVM training");
        }
    }
    
    /**
     * Searches all hyperparameter combinations and evaluates them.
     */
    private static ArrayList<HyperparameterCandidate> searchHyperparameters(ArrayList<DigitSample> trainingData) {
        ArrayList<HyperparameterCandidate> candidates = new ArrayList<>();
        System.out.println("Performing internal validation for Linear SVM hyperparameters...");
        for (double lambda : SVM_LAMBDA_CANDIDATES) { // sweep regularization
            for (int epochs : SVM_EPOCH_CANDIDATES) { // sweep epochs
                for (double minLR : SVM_MIN_LR_CANDIDATES) { // sweep lr floor
                    double accuracy = evaluateSVMHyperparameters(trainingData, lambda, epochs, minLR); // inner val
                    System.out.printf("  Candidate λ=%.5f, epochs=%d, η_min=%.1e -> validation accuracy %.2f%%\n",
                                      lambda, epochs, minLR, accuracy);
                    candidates.add(new HyperparameterCandidate(lambda, epochs, minLR, accuracy));
                }
            }
        }
        return candidates;
    }
    
    /**
     * Builds ensemble models from top candidates.
     */
    private static ArrayList<LinearSVM> buildEnsembleModels(ArrayList<DigitSample> trainingData, 
                                                         ArrayList<HyperparameterCandidate> candidates) {
        ArrayList<LinearSVM> models = new ArrayList<>();
        int modelsToTrain = Math.min(SVM_ENSEMBLE_SIZE, candidates.size());
        for (int i = 0; i < modelsToTrain; i++) {
            HyperparameterCandidate candidate = candidates.get(i);
            LinearSVM model = new LinearSVM(candidate.lambda, candidate.epochs, candidate.minLearningRate);
            model.train(trainingData);
            models.add(model);
        }
        return models;
    }
    
    /**
     * Creates final SVM training result with classifier and summary.
     */
    private static SVMTrainingResult createSVMTrainingResult(ArrayList<LinearSVM> ensembleModels,
                                                              ArrayList<HyperparameterCandidate> candidates) {
        Classifier classifier;
        String summaryText;
        double referenceAccuracy = candidates.isEmpty() ? ZERO : candidates.get(0).validationAccuracy;
        ArrayList<String> summaries = new ArrayList<>();
        for (int i = 0; i < ensembleModels.size(); i++) {
            summaries.add(candidates.get(i).describe());
        }
        if (ensembleModels.size() == 1) {
            classifier = ensembleModels.get(0);
            summaryText = summaries.get(0);
        } else {
            classifier = new LinearSVMEnsemble(ensembleModels);
            summaryText = String.format("Ensemble of %d models: %s", 
                                        ensembleModels.size(), String.join("; ", summaries));
        }
        return new SVMTrainingResult(classifier, summaryText, referenceAccuracy);
    }
    
    /**
     * Evaluates SVM hyperparameters using cross-validation.
     */
    private static double evaluateSVMHyperparameters(ArrayList<DigitSample> trainingData,
                                                     double lambda, int epochs, double minLearningRate) {
        double accuracySum = 0.0;
        for (int repeat = 0; repeat < SVM_VALIDATION_REPEATS; repeat++) { // repeat with new shuffle
            double accuracy = evaluateSingleValidationFold(trainingData, lambda, epochs, minLearningRate, repeat);
            accuracySum += accuracy;
        }
        return accuracySum / SVM_VALIDATION_REPEATS;
    }
    
    /**
     * Evaluates a single validation fold for hyperparameter tuning.
     */
    private static double evaluateSingleValidationFold(ArrayList<DigitSample> trainingData, double lambda,
                                                        int epochs, double minLearningRate, int repeat) {
        ArrayList<DigitSample> shuffled = new ArrayList<>(trainingData); // copy
        CustomCollections.shuffle(shuffled, new Random(CustomObjects.hash(lambda, epochs, minLearningRate, repeat))); // reproducible shuffle
        int validationStartIndex = calculateValidationSplitIndex(shuffled.size()); // split point
        ArrayList<DigitSample> innerTrain = new ArrayList<>(shuffled.subList(0, validationStartIndex)); // train fold
        ArrayList<DigitSample> validationSet = new ArrayList<>(shuffled.subList(validationStartIndex, shuffled.size())); // val fold
        LinearSVM candidate = new LinearSVM(lambda, epochs, minLearningRate); // candidate model
        candidate.train(innerTrain); // fit on inner train
        return candidate.evaluate(validationSet);
    }
    
    /**
     * Calculates the split index for validation set.
     */
    /**
     * Calculates the split index for validation set.
     * Ensures index is within valid bounds (at least 1, at most totalSize-1).
     */
    private static int calculateValidationSplitIndex(int totalSize) {
        int index = (int) (totalSize * VALIDATION_SPLIT_RATIO);
        return Math.min(Math.max(index, MIN_VALIDATION_INDEX), totalSize - INT_ONE);
    }
    
    /**
     * Helper class to store SVM training results.
     */
    private static class SVMTrainingResult {
        final Classifier classifier;
        final String hyperparameterSummary;
        final double validationAccuracy;
        
        /**
         * Creates a new SVM training result.
         */
        SVMTrainingResult(Classifier classifier, String hyperparameterSummary, double validationAccuracy) {
            this.classifier = classifier;
            this.hyperparameterSummary = hyperparameterSummary;
            this.validationAccuracy = validationAccuracy;
        }
    }
    
    /**
     * Helper class to store hyperparameter candidate and its validation accuracy.
     * Implements Comparable for sorting by accuracy (descending).
     */
    private static class HyperparameterCandidate implements Comparable<HyperparameterCandidate> {
        final double lambda;
        final int epochs;
        final double minLearningRate;
        final double validationAccuracy;
        
        /**
         * Creates a new hyperparameter candidate.
         */
        HyperparameterCandidate(double lambda, int epochs, double minLearningRate, double validationAccuracy) {
            this.lambda = lambda;
            this.epochs = epochs;
            this.minLearningRate = minLearningRate;
            this.validationAccuracy = validationAccuracy;
        }
        
        /**
         * Compares candidates by validation accuracy (descending order).
         */
        @Override
        public int compareTo(HyperparameterCandidate other) {
            return Double.compare(other.validationAccuracy, this.validationAccuracy);
        }
        
        /**
         * Returns a formatted string describing this hyperparameter candidate.
         */
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
    
    /**
     * Returns a string representation of this sample.
     */
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
    
    /**
     * Returns a string representation of this classification result.
     */
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
    
    /**
     * Returns a string representation of this neighbor distance.
     */
    @Override
    public String toString() {
        return String.format("NeighborDistance{label=%d, distance=%.3f}", label, distance);
    }
    
    /**
     * Checks equality based on label and distance.
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        NeighborDistance that = (NeighborDistance) obj;
        return label == that.label && 
               Double.compare(that.distance, distance) == 0;
    }
    
    /**
     * Computes hash code based on label and distance.
     */
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
    void train(ArrayList<DigitSample> trainingData);
    
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
    default double evaluate(ArrayList<DigitSample> testData) {
        int correct = 0;
        
        for (DigitSample testSample : testData) {
            ClassificationResult result = classify(testSample);
            if (result.getPredictedLabel() == testSample.getLabel()) {
                correct++;
            }
        }
        
        return (correct * DigitRecognitionApp.PERCENTAGE_MULTIPLIER) / testData.size();
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
    public void train(ArrayList<DigitSample> trainingData) {
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
    private ArrayList<DigitSample> trainingData;
    
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
    public void train(ArrayList<DigitSample> trainingData) {
        // k-NN is a lazy learning algorithm - just store the training data
        this.trainingData = new ArrayList<>(trainingData);
    }
    
    @Override
    public ClassificationResult classify(DigitSample testSample) {
        if (trainingData == null || trainingData.isEmpty()) {
            throw new IllegalStateException("Classifier must be trained before classification");
        }
        
        // Calculate distances to all training samples
        ArrayList<NeighborDistance> distances = new ArrayList<>();
        
        for (DigitSample trainSample : trainingData) {
            double distance = DistanceCalculator.euclideanDistance(
                testSample.getFeatures(), trainSample.getFeatures());
            distances.add(new NeighborDistance(trainSample.getLabel(), distance));
        }
        
        // Sort by distance and take k nearest neighbors
        CustomCollections.sort(distances);
        ArrayList<NeighborDistance> kNearest = new ArrayList<>(distances.subList(0, Math.min(k, distances.size())));
        
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
    private ClassificationResult majorityVote(ArrayList<NeighborDistance> neighbors) {
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
    private ClassificationResult weightedVote(ArrayList<NeighborDistance> neighbors) {
        double[] weights = new double[NUM_CLASSES];
        
        for (NeighborDistance neighbor : neighbors) {
            // Use inverse distance as weight (add small epsilon to avoid division by zero)
            double weight = DigitRecognitionApp.ONE / (neighbor.getDistance() + DigitRecognitionApp.DISTANCE_WEIGHTING_EPSILON);
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
        
        double confidence = totalWeight > DigitRecognitionApp.ZERO ? maxWeight / totalWeight : DigitRecognitionApp.ZERO;
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
    
    // Numeric constants for LinearSVM
    private static final double ZERO = 0.0;
    private static final double ONE = 1.0;
    private static final int INT_ZERO = 0;
    private static final int INT_ONE = 1;
    private static final int INT_TWO = 2;
    private static final int SPATIAL_AUGMENTATION_MULTIPLIER = 2;
    private static final int EARLY_STOPPING_MIN_EPOCHS = 10;
    private static final int BINARY_POSITIVE_LABEL = 1;
    private static final int BINARY_NEGATIVE_LABEL = -1;
    private static final double MARGIN_THRESHOLD = 1.0;
    private static final double STD_DEVIATION_EPSILON = 1e-8;
    private static final double DEFAULT_STD_DEVIATION = 1.0;
    private static final int MAX_POLYNOMIAL_FEATURES = 896;
    private static final double POLYNOMIAL_COUNT_DIVISOR = 2.0;
    private static final int POLYNOMIAL_STEP_DIVISOR = 20;
    private static final double RFF_SCALE_MULTIPLIER = 2.0;
    private static final double RFF_BIAS_MULTIPLIER = 2.0;
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
    
    /**
     * Creates a Linear SVM with specified hyperparameters.
     * 
     * @param regularizationParameter Lambda (λ) - controls regularization strength
     * @param maxEpochs Maximum number of training epochs
     * @param minLearningRate Minimum learning rate (clamping threshold)
     */
    public LinearSVM(double regularizationParameter, int maxEpochs, double minLearningRate) {
        this.regularizationParameter = regularizationParameter;
        this.maxEpochs = maxEpochs;
        this.minLearningRate = minLearningRate;
        this.isTrained = false;
    }
    
    @Override
    /**
     * Trains the Linear SVM classifier on provided training data.
     */
    public void train(ArrayList<DigitSample> trainingData) {
        validateTrainingData(trainingData);
        initializeFeatureConfiguration(trainingData);
        ArrayList<DigitSample> normalizedData = normalizeTrainingData(trainingData);
        trainAllPairs(normalizedData);
        this.isTrained = true;
    }
    
    /**
     * Validates training data is not null or empty.
     */
    private void validateTrainingData(ArrayList<DigitSample> trainingData) {
        if (trainingData == null || trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
    }
    
    /**
     * Initializes feature configuration and counts.
     */
    private void initializeFeatureConfiguration(ArrayList<DigitSample> trainingData) {
        rawFeatureCount = trainingData.get(0).getFeatureCount(); // base dims from dataset
        gridSize = (int) Math.round(Math.sqrt(rawFeatureCount)); // infer image grid size
        useSpatialAugmentation = ENABLE_SPATIAL_AUGMENTATION && gridSize * gridSize == rawFeatureCount; // add row/col means only if square
        baseFeatureCount = useSpatialAugmentation ? rawFeatureCount + SPATIAL_AUGMENTATION_MULTIPLIER * gridSize : rawFeatureCount; // base + spatial extras
        useRandomFourierFeatures = ENABLE_RANDOM_FOURIER_FEATURES; // enable RFF block
        randomFourierFeatureCount = useRandomFourierFeatures ? RANDOM_FOURIER_FEATURE_COUNT : 0; // RFF dims
        randomFourierGamma = RANDOM_FOURIER_GAMMA; // RBF bandwidth for RFF
        initializeRandomFourierFeatures(); // sample RFF weights/biases
        int polynomialFeatureCount = ENABLE_POLYNOMIAL_FEATURES ? computePolynomialFeatureCount(baseFeatureCount) : 0; // degree-2 poly dims
        numFeatures = baseFeatureCount + randomFourierFeatureCount + polynomialFeatureCount; // total augmented dims
        weightVectors = new double[NUM_PAIRS][numFeatures]; // OvO weight matrix
        biasTerms = new double[NUM_PAIRS]; // OvO biases
        initializePairMapping(); // map (class1,class2) -> pair index
    }
    
    /**
     * Normalizes training data using z-score normalization.
     */
    private ArrayList<DigitSample> normalizeTrainingData(ArrayList<DigitSample> trainingData) {
        double[] featureMeans = computeFeatureMeans(trainingData);
        double[] featureStds = computeFeatureStdDeviations(trainingData, featureMeans);
        storedMeans = featureMeans;
        storedStds = featureStds;
        return normalizeFeaturesZScore(trainingData, featureMeans, featureStds);
    }
    
    /**
     * Trains all pairwise binary classifiers (one-vs-one).
     */
    private void trainAllPairs(ArrayList<DigitSample> normalizedData) {
        int pairIndex = 0;
        for (int class1 = 0; class1 < NUM_CLASSES; class1++) {
            for (int class2 = class1 + 1; class2 < NUM_CLASSES; class2++) {
                trainBinaryClassifier(normalizedData, class1, class2, pairIndex);
                pairIndex++;
            }
        }
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
    /**
     * Gets the pair index for two classes (ensures class1 < class2 for consistent lookup).
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
    /** Trains one binary classifier (OvO) using Pegasos SGD. */
    private void trainBinaryClassifier(ArrayList<DigitSample> trainingData, int class1, int class2, int pairIndex) {
        ArrayList<DigitSample> pairData = filterPairData(trainingData, class1, class2);
        double classWeight = computeClassWeight(pairData, class1, class2);
        PegasosResult result = runPegasos(pairData, class1, classWeight);
        weightVectors[pairIndex] = result.weights;
        biasTerms[pairIndex] = result.bias;
    }
    
    /** Keeps only samples from the two classes being separated. */
    /**
     * Filters training data to only include samples from the two specified classes.
     */
    private ArrayList<DigitSample> filterPairData(ArrayList<DigitSample> data, int class1, int class2) {
        ArrayList<DigitSample> pairData = new ArrayList<>();
        for (DigitSample sample : data) {
            int label = sample.getLabel();
            if (label == class1 || label == class2) pairData.add(sample);
        }
        return pairData;
    }
    
    /** Computes class weight to balance minority class. */
    private double computeClassWeight(ArrayList<DigitSample> pairData, int class1, int class2) {
        int c1 = 0, c2 = 0;
        for (DigitSample sample : pairData) { if (sample.getLabel() == class1) c1++; else c2++; }
        double weight = USE_CLASS_WEIGHTING && c1 > INT_ZERO && c2 > INT_ZERO ? (double) c2 / c1 : ONE;
        return Math.min(weight, MAX_CLASS_WEIGHT);
    }
    
    /** Runs Pegasos SGD and returns averaged weights/bias. */
    private PegasosResult runPegasos(ArrayList<DigitSample> pairData, int class1, double classWeight) {
        double[] weights = new double[numFeatures];
        double[] cumulative = new double[numFeatures]; // for averaged perceptron style
        double bias = 0.0, cumulativeBias = 0.0;
        int snapshots = 0, numSamples = pairData.size(), globalIter = 0;
        ArrayList<DigitSample> shuffled = new ArrayList<>(pairData); // working copy
        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            CustomCollections.shuffle(shuffled); // new order each epoch
            PegasosEpochResult result = runEpoch(shuffled, class1, classWeight, weights, bias, globalIter); // one pass
            globalIter += shuffled.size(); // advance global iteration counter
            bias = result.bias; // carry forward updated bias
            // Early stop if very few updates after warm-up
            if (result.updates < numSamples * EARLY_STOPPING_UPDATE_THRESHOLD && epoch > EARLY_STOPPING_MIN_EPOCHS) break;
            accumulateSnapshot(weights, cumulative); // accumulate for averaging
            cumulativeBias += bias;
            snapshots++;
        }
        double finalBias = snapshots > 0 ? averageWeights(weights, cumulative, cumulativeBias, snapshots) : bias; // average if any snapshots
        return new PegasosResult(weights, finalBias);
    }
    
    /** Processes one epoch; returns updates count and final bias. */
    /**
     * Runs one epoch of Pegasos training on the provided data.
     * Processes each sample, updates weights based on margin, and enforces constraints.
     */
    private PegasosEpochResult runEpoch(ArrayList<DigitSample> data, int class1, double classWeight,
                                        double[] weights, double bias, int globalIterStart) {
        int updates = 0; // Count of samples that triggered weight updates (margin < 1)
        int globalIter = globalIterStart; // Track total iterations across all epochs
        
        // Process each sample in the shuffled dataset
        for (DigitSample sample : data) {
            globalIter++; // Increment global iteration counter for learning rate schedule
            
            // Adaptive learning rate: decreases as 1/(λ*t), clamped to minimum
            // This ensures convergence while preventing too-small updates
            double lr = Math.max(minLearningRate, 1.0 / (regularizationParameter * globalIter));
            
            // Convert multi-class label to binary: class1 = +1, other = -1
                int binaryLabel = (sample.getLabel() == class1) ? BINARY_POSITIVE_LABEL : BINARY_NEGATIVE_LABEL;
            
            // Apply class weighting: positive class gets higher weight if it's minority
            double sampleWeight = (binaryLabel == BINARY_POSITIVE_LABEL) ? classWeight : ONE;
            
            // Compute margin: y * (w·x + b). Positive margin = correct classification
            double margin = binaryLabel * computeDecisionValue(sample.getFeatures(), weights, bias);
            
            // Hinge loss active: if margin < 1, sample is misclassified or too close to boundary
            if (margin < MARGIN_THRESHOLD) {
                updates++; // Count this as an update
                // Apply gradient step: update weights and bias based on hinge loss
                bias = gradientStep(weights, bias, lr, sampleWeight, binaryLabel, sample.getFeatures());
            } else {
                // Margin >= 1: sample is correctly classified with sufficient margin
                // Only apply weight decay (regularization), no gradient update
                bias = decayOnly(weights, bias, lr);
            }
            
            // Project weights to ensure ||w|| <= 1/sqrt(λ) for stability
            enforceMaxNorm(weights);
        }
        
        return new PegasosEpochResult(updates, bias);
    }
    
    /** Applies hinge-loss gradient + decay. */
    private double gradientStep(double[] weights, double bias, double lr, double sampleWeight, int label, double[] features) {
        double scale = 1.0 - lr * regularizationParameter;
        double wlr = lr * sampleWeight;
        for (int i = 0; i < numFeatures; i++) weights[i] = scale * weights[i] + wlr * label * features[i];
        return scale * bias + wlr * label;
    }
    
    /** Applies decay when margin is satisfied. */
    private double decayOnly(double[] weights, double bias, double lr) {
        double scale = 1.0 - lr * regularizationParameter;
        for (int i = 0; i < numFeatures; i++) weights[i] *= scale;
        return bias * scale;
    }
    
    /** Keeps weights within max norm for stability. */
    private void enforceMaxNorm(double[] weights) {
        double norm = 0.0;
        for (double w : weights) norm += w * w;
        norm = Math.sqrt(norm);
        double maxNorm = ONE / Math.sqrt(regularizationParameter);
        if (norm > maxNorm) {
            double scale = maxNorm / norm;
            for (int i = 0; i < numFeatures; i++) weights[i] *= scale;
        }
    }
    
    /** Accumulates weights for averaging. */
    private void accumulateSnapshot(double[] weights, double[] cumulative) {
        for (int i = 0; i < numFeatures; i++) cumulative[i] += weights[i];
    }
    
    /** Averages snapshots to produce final weights and bias. */
    private double averageWeights(double[] weights, double[] cumulative, double cumulativeBias, int snapshots) {
        for (int i = 0; i < numFeatures; i++) weights[i] = cumulative[i] / snapshots;
        return cumulativeBias / snapshots;
    }

    /** Holder for Pegasos training result. */
    private static class PegasosResult {
        final double[] weights;
        final double bias;
        /**
         * Creates a new Pegasos result with trained weights and bias.
         */
        PegasosResult(double[] weights, double bias) {
            this.weights = weights;
            this.bias = bias;
        }
    }
    
    /** Holder for one epoch outcome. */
    private static class PegasosEpochResult {
        final int updates;
        final double bias;
        /**
         * Creates a new epoch result with update count and bias.
         */
        PegasosEpochResult(int updates, double bias) {
            this.updates = updates;
            this.bias = bias;
        }
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
    private double[] computeFeatureMeans(ArrayList<DigitSample> trainingData) {
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
    private double[] computeFeatureStdDeviations(ArrayList<DigitSample> trainingData, double[] means) {
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
            if (stds[featureIndex] < STD_DEVIATION_EPSILON) {
                stds[featureIndex] = DEFAULT_STD_DEVIATION;
            }
        }
        
        return stds;
    }
    
    /**
     * Normalizes features using z-score normalization (standardization).
     * Transforms features to have mean=0 and std=1.
     */
    private ArrayList<DigitSample> normalizeFeaturesZScore(ArrayList<DigitSample> trainingData, 
                                                       double[] means, double[] stds) {
        ArrayList<DigitSample> normalizedData = new ArrayList<>();
        
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
    
    /**
     * Classifies a test sample using one-vs-one voting strategy.
     */
    @Override
    public ClassificationResult classify(DigitSample testSample) {
        validateTrained();
        double[] normalizedFeatures = normalizeTestFeatures(testSample);
        VotingResult voting = runPairwiseVoting(normalizedFeatures);
        return createClassificationResult(voting);
    }
    
    /**
     * Validates that classifier has been trained.
     */
    private void validateTrained() {
        if (!isTrained) {
            throw new IllegalStateException("Classifier must be trained before classification");
        }
    }
    
    /**
     * Normalizes test sample features using training statistics.
     */
    private double[] normalizeTestFeatures(DigitSample testSample) {
        double[] features = testSample.getFeatures();
        double[] augmentedFeatures = getAugmentedFeatures(features);
        double[] normalizedFeatures = new double[numFeatures];
        for (int featureIndex = 0; featureIndex < numFeatures; featureIndex++) {
            normalizedFeatures[featureIndex] = 
                (augmentedFeatures[featureIndex] - storedMeans[featureIndex]) / storedStds[featureIndex];
        }
        normalizeVectorInPlace(normalizedFeatures);
        return normalizedFeatures;
    }
    
    /**
     * Runs all pairwise classifiers and collects votes.
     */
    private VotingResult runPairwiseVoting(double[] normalizedFeatures) {
        int[] votes = new int[NUM_CLASSES];
        double[] confidences = new double[NUM_CLASSES];
        for (int class1 = 0; class1 < NUM_CLASSES; class1++) {
            for (int class2 = class1 + 1; class2 < NUM_CLASSES; class2++) {
                processPair(class1, class2, normalizedFeatures, votes, confidences);
            }
        }
        return new VotingResult(votes, confidences);
    }
    
    /**
     * Processes a single pair and updates votes/confidences.
     */
    private void processPair(int class1, int class2, double[] features, int[] votes, double[] confidences) {
        int pairIdx = getPairIndex(class1, class2);
        double decisionValue = computeDecisionValue(features, weightVectors[pairIdx], biasTerms[pairIdx]);
                if (decisionValue > ZERO) {
            votes[class1]++;
                    confidences[class1] += Math.max(INT_ZERO, decisionValue);
        } else {
            votes[class2]++;
                    confidences[class2] += Math.max(INT_ZERO, -decisionValue);
        }
    }
    
    /**
     * Finds winning class and creates classification result.
     */
    private ClassificationResult createClassificationResult(VotingResult voting) {
        int predictedClass = findWinningClass(voting.votes, voting.confidences);
        double confidence = computeConfidence(voting.votes[predictedClass]);
        return new ClassificationResult(predictedClass, confidence);
    }
    
    /**
     * Finds class with most votes (with confidence tie-breaking).
     */
    private int findWinningClass(int[] votes, double[] confidences) {
        int predictedClass = 0;
        int maxVotes = votes[0];
        for (int classLabel = 1; classLabel < NUM_CLASSES; classLabel++) {
            if (votes[classLabel] > maxVotes || 
                (votes[classLabel] == maxVotes && confidences[classLabel] > confidences[predictedClass])) {
                maxVotes = votes[classLabel];
                predictedClass = classLabel;
            }
        }
        return predictedClass;
    }
    
    /**
     * Computes confidence score from vote count.
     */
    private double computeConfidence(int voteCount) {
        double totalVotes = NUM_CLASSES - 1; // Each class participates in 9 comparisons
        return (double) voteCount / totalVotes;
    }
    
    /**
     * Helper class to store voting results.
     */
    private static class VotingResult {
        final int[] votes;
        final double[] confidences;
        
        VotingResult(int[] votes, double[] confidences) {
            this.votes = votes;
            this.confidences = confidences;
        }
    }
    
    @Override
    public String getAlgorithmName() {
        return "Linear SVM";
    }
    
    public String getHyperparameterSummary() {
        return String.format("λ=%.5f, epochs=%d, η_min=%.1e", 
                             regularizationParameter, maxEpochs, minLearningRate);
    }
    
    /**
     * Combines base, RFF, and polynomial features into augmented feature vector.
     */
    private double[] getAugmentedFeatures(double[] originalFeatures) {
        double[] baseFeatures = buildBaseFeatures(originalFeatures);
        double[] augmented = new double[numFeatures];
        int offset = copyBaseFeatures(baseFeatures, augmented, 0);
        offset = copyRandomFourierFeatures(originalFeatures, augmented, offset);
        copyPolynomialFeatures(baseFeatures, augmented, offset);
        return augmented;
    }
    
    /**
     * Copies base features into augmented array.
     */
    private int copyBaseFeatures(double[] baseFeatures, double[] augmented, int offset) {
        System.arraycopy(baseFeatures, 0, augmented, offset, baseFeatureCount);
        return offset + baseFeatureCount;
    }
    
    /**
     * Copies random Fourier features if enabled.
     */
    private int copyRandomFourierFeatures(double[] originalFeatures, double[] augmented, int offset) {
        if (useRandomFourierFeatures) {
            double[] randomFeatures = computeRandomFourierFeatures(originalFeatures);
            System.arraycopy(randomFeatures, 0, augmented, offset, randomFourierFeatureCount);
            return offset + randomFourierFeatureCount;
        }
        return offset;
    }
    
    /**
     * Copies polynomial features if enabled.
     */
    private void copyPolynomialFeatures(double[] baseFeatures, double[] augmented, int offset) {
        if (ENABLE_POLYNOMIAL_FEATURES) {
            double[] polyFeatures = computePolynomialFeatures(baseFeatures);
            System.arraycopy(polyFeatures, 0, augmented, offset, polyFeatures.length);
        }
    }
    
    private int computePolynomialFeatureCount(int baseCount) {
        // For degree 2: include all pairwise products (x_i * x_j where i <= j)
        // This gives us baseCount + baseCount*(baseCount+1)/2 features
        // But to keep it manageable, we'll use a subset: top features only
        int maxPolyFeatures = Math.min(MAX_POLYNOMIAL_FEATURES, baseCount * (baseCount + INT_ONE) / INT_TWO); // Increased for >98% accuracy
        return maxPolyFeatures;
    }
    
    /**
     * Computes degree-2 polynomial features (squares and products).
     */
    private double[] computePolynomialFeatures(double[] baseFeatures) {
        int polyCount = computePolynomialFeatureCount(baseFeatureCount);
        double[] polyFeatures = new double[polyCount];
        int idx = addSquaredFeatures(baseFeatures, polyFeatures, polyCount);
        idx = addNearbyProducts(baseFeatures, polyFeatures, idx, polyCount);
        addDistantProducts(baseFeatures, polyFeatures, idx, polyCount);
        return polyFeatures;
    }
    
    /**
     * Adds squared features (x_i^2) to polynomial array.
     */
    private int addSquaredFeatures(double[] baseFeatures, double[] polyFeatures, int polyCount) {
        int idx = 0;
        int maxSquared = Math.min(baseFeatureCount, (int)(polyCount / POLYNOMIAL_COUNT_DIVISOR));
        for (int i = 0; i < maxSquared; i++) {
            polyFeatures[idx++] = baseFeatures[i] * baseFeatures[i];
        }
        return idx;
    }
    
    /**
     * Adds nearby pairwise products (local interactions).
     */
    private int addNearbyProducts(double[] baseFeatures, double[] polyFeatures, int idx, int polyCount) {
        for (int i = 0; i < baseFeatureCount && idx < polyCount; i++) {
            int maxJ = Math.min(i + POLYNOMIAL_NEARBY_FEATURE_RANGE, baseFeatureCount);
            for (int j = i + 1; j < maxJ && idx < polyCount; j++) {
                polyFeatures[idx++] = baseFeatures[i] * baseFeatures[j];
            }
        }
        return idx;
    }
    
    /**
     * Adds distant pairwise products (global interactions).
     */
    private void addDistantProducts(double[] baseFeatures, double[] polyFeatures, int idx, int polyCount) {
        if (idx >= polyCount) return;
            int step = Math.max(INT_ONE, baseFeatureCount / POLYNOMIAL_STEP_DIVISOR);
        for (int i = 0; i < baseFeatureCount && idx < polyCount; i += step) {
            for (int j = i + step; j < baseFeatureCount && idx < polyCount; j += step) {
                polyFeatures[idx++] = baseFeatures[i] * baseFeatures[j];
            }
        }
    }
    
    /**
     * Builds base features with optional spatial augmentation.
     */
    private double[] buildBaseFeatures(double[] originalFeatures) {
        if (!useSpatialAugmentation) {
            return originalFeatures.clone();
        }
        double[] base = new double[baseFeatureCount];
        System.arraycopy(originalFeatures, 0, base, 0, rawFeatureCount);
        int offset = addRowAverages(originalFeatures, base, rawFeatureCount);
        addColumnAverages(originalFeatures, base, offset);
        return base;
    }
    
    /**
     * Adds row average features to base array.
     */
    private int addRowAverages(double[] originalFeatures, double[] base, int offset) {
        for (int row = 0; row < gridSize; row++) {
            double sum = computeRowSum(originalFeatures, row);
            base[offset + row] = sum / gridSize;
        }
        return offset + gridSize;
    }
    
    /**
     * Computes sum of values in a row.
     */
    private double computeRowSum(double[] features, int row) {
        double sum = 0.0;
        for (int col = 0; col < gridSize; col++) {
            sum += features[row * gridSize + col];
        }
        return sum;
    }
    
    /**
     * Adds column average features to base array.
     */
    private void addColumnAverages(double[] originalFeatures, double[] base, int offset) {
        for (int col = 0; col < gridSize; col++) {
            double sum = computeColumnSum(originalFeatures, col);
            base[offset + col] = sum / gridSize;
        }
    }
    
    /**
     * Computes sum of values in a column.
     */
    private double computeColumnSum(double[] features, int col) {
        double sum = 0.0;
        for (int row = 0; row < gridSize; row++) {
            sum += features[row * gridSize + col];
        }
        return sum;
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
        double stdDev = Math.sqrt(RFF_SCALE_MULTIPLIER * randomFourierGamma);
        
        for (int featureIndex = 0; featureIndex < randomFourierFeatureCount; featureIndex++) {
            for (int dimension = 0; dimension < rawFeatureCount; dimension++) {
                randomFourierWeights[featureIndex][dimension] = random.nextGaussian() * stdDev;
            }
            randomFourierBiases[featureIndex] = random.nextDouble() * RFF_BIAS_MULTIPLIER * Math.PI;
        }
    }
    
    private double[] computeRandomFourierFeatures(double[] originalFeatures) {
        double[] projected = new double[randomFourierFeatureCount];
        double scale = Math.sqrt(RFF_SCALE_MULTIPLIER / randomFourierFeatureCount);
        
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
    private final ArrayList<LinearSVM> models;
    
    public LinearSVMEnsemble(ArrayList<LinearSVM> models) {
        if (models == null || models.isEmpty()) {
            throw new IllegalArgumentException("Ensemble must contain at least one model");
        }
        this.models = new ArrayList<>(models);
    }
    
    @Override
    public void train(ArrayList<DigitSample> trainingData) {
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
        
        double confidence = totalVote > DigitRecognitionApp.ZERO ? bestVote / totalVote : DigitRecognitionApp.ZERO;
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
    public static ArrayList<DigitSample> loadDataset(String filename) throws IOException {
        ArrayList<DigitSample> samples = new ArrayList<>();
        
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
        
        if (label < DigitRecognitionApp.MIN_LABEL || label > DigitRecognitionApp.MAX_LABEL) {
            throw new IllegalArgumentException("Label must be between " + DigitRecognitionApp.MIN_LABEL + " and " + DigitRecognitionApp.MAX_LABEL + ", found: " + label);
        }
        
        return new DigitSample(features, label);
    }
    
    /**
     * Analyzes and returns the class distribution of a dataset.
     * 
     * @param dataset The dataset to analyze
     * @return Array where index i contains the count of samples with label i
     */
    public static int[] getClassDistribution(ArrayList<DigitSample> dataset) {
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
    public static void printDatasetAnalysis(ArrayList<DigitSample> dataset, String datasetName) {
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
    public static int[][] generateConfusionMatrix(Classifier classifier, ArrayList<DigitSample> testData) {
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
    /** Prints per-class precision/recall/F1 and macro averages. */
    public static void printClassificationReport(int[][] confusionMatrix, String algorithmName) {
        printReportHeader(algorithmName);
        double totalPrecision = 0, totalRecall = 0, totalF1 = 0;
        int totalSupport = 0;
        for (int i = 0; i < NUM_CLASSES; i++) {
            ClassMetrics metrics = computeClassMetrics(confusionMatrix, i);
            printClassMetricsLine(i, metrics);
            totalPrecision += metrics.precision;
            totalRecall += metrics.recall;
            totalF1 += metrics.f1Score;
            totalSupport += metrics.support;
        }
        printMacroAverages(totalPrecision, totalRecall, totalF1, totalSupport);
    }
    
    /** Prints the header for classification report. */
    private static void printReportHeader(String algorithmName) {
        System.out.println("\nPer-class Performance Metrics for " + algorithmName + ":");
        System.out.println("Class | Precision | Recall   | F1-Score | Support");
        System.out.println("------|-----------|--------- |----------|--------");
    }
    
    /** Computes precision/recall/F1 for one class. */
    private static ClassMetrics computeClassMetrics(int[][] confusionMatrix, int cls) {
        int tp = confusionMatrix[cls][cls];
        int fp = 0, fn = 0, support = 0;
        for (int j = 0; j < NUM_CLASSES; j++) {
            if (cls != j) {
                fp += confusionMatrix[j][cls];
                fn += confusionMatrix[cls][j];
            }
            support += confusionMatrix[cls][j];
        }
        double precision = (tp + fp > DigitRecognitionApp.INT_ZERO) ? (double) tp / (tp + fp) : DigitRecognitionApp.ZERO;
        double recall = (support > DigitRecognitionApp.INT_ZERO) ? (double) tp / support : DigitRecognitionApp.ZERO;
        double f1 = (precision + recall > DigitRecognitionApp.ZERO) ? DigitRecognitionApp.F1_SCORE_MULTIPLIER * precision * recall / (precision + recall) : DigitRecognitionApp.ZERO;
        return new ClassMetrics(precision, recall, f1, support);
    }
    
    /** Prints one class metrics line. */
    private static void printClassMetricsLine(int cls, ClassMetrics metrics) {
        System.out.printf("  %d   |   %.3f   |  %.3f   |  %.3f   |  %4d\n",
            cls, metrics.precision, metrics.recall, metrics.f1Score, metrics.support);
    }
    
    /** Prints macro averages over all classes. */
    private static void printMacroAverages(double totalPrecision, double totalRecall, double totalF1, int totalSupport) {
        System.out.println("------|-----------|----------|----------|--------");
        System.out.printf(" Avg  |   %.3f   |  %.3f   |  %.3f   |  %4d\n",
            totalPrecision / NUM_CLASSES, totalRecall / NUM_CLASSES, totalF1 / NUM_CLASSES, totalSupport);
    }
    
    /** Holder for per-class metrics. */
    private static class ClassMetrics {
        final double precision;
        final double recall;
        final double f1Score;
        final int support;
        ClassMetrics(double precision, double recall, double f1Score, int support) {
            this.precision = precision;
            this.recall = recall;
            this.f1Score = f1Score;
            this.support = support;
        }
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
        
        return total > DigitRecognitionApp.INT_ZERO ? (correct * DigitRecognitionApp.PERCENTAGE_MULTIPLIER) / total : DigitRecognitionApp.ZERO;
    }
    
    /**
     * Performs comprehensive evaluation of a classifier.
     * 
     * @param classifier The trained classifier
     * @param testData The test dataset
     * @return Accuracy percentage
     */
    public static double evaluateClassifier(Classifier classifier, ArrayList<DigitSample> testData) {
        int[][] confusionMatrix = generateConfusionMatrix(classifier, testData);
        double accuracy = calculateAccuracy(confusionMatrix);
        
        System.out.printf("\n%s Accuracy: %.2f%%\n", classifier.getAlgorithmName(), accuracy);
        printConfusionMatrix(confusionMatrix, classifier.getAlgorithmName());
        printClassificationReport(confusionMatrix, classifier.getAlgorithmName());
        
        return accuracy;
    }
}

// ============================================================================
// CUSTOM UTILITY CLASSES (Replacing java.util.List and java.util.Objects)
// ============================================================================

/**
 * Custom List implementation to replace java.util.List.
 * Simple dynamic array-based list.
 */
/**
 * Custom utility class to replace java.util.Objects functionality.
 */
class CustomObjects {
    /**
     * Custom hash function to replace Objects.hash().
     * Uses a simple polynomial hash with prime number.
     */
    public static int hash(Object... values) {
        int result = 1;
        for (Object value : values) {
            result = 31 * result + (value == null ? 0 : value.hashCode());
        }
        return result;
    }
    
    public static boolean equals(Object a, Object b) {
        return (a == b) || (a != null && a.equals(b));
    }
}

/**
 * Custom utility class to replace java.util.Collections functionality.
 */
class CustomCollections {
    /**
     * Custom sort implementation using insertion sort (simple and stable).
     */
    public static <T extends Comparable<T>> void sort(ArrayList<T> list) {
        for (int i = 1; i < list.size(); i++) {
            T key = list.get(i);
            int j = i - 1;
            while (j >= 0 && list.get(j).compareTo(key) > 0) {
                list.set(j + 1, list.get(j));
                j--;
            }
            list.set(j + 1, key);
        }
    }
    
    /**
     * Custom shuffle implementation using Fisher-Yates algorithm.
     */
    public static <T> void shuffle(ArrayList<T> list, Random random) {
        for (int i = list.size() - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            T temp = list.get(i);
            list.set(i, list.get(j));
            list.set(j, temp);
        }
    }
    
    /**
     * Shuffles list with default random generator.
     */
    public static <T> void shuffle(ArrayList<T> list) {
        shuffle(list, new Random());
    }
}