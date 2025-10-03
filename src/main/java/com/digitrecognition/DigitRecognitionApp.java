package com.digitrecognition;

import java.util.List;
import com.digitrecognition.algorithms.*;
import com.digitrecognition.models.DigitSample;
import com.digitrecognition.utils.*;

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
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
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
    private static FoldResults evaluateAllAlgorithms(List<DigitSample> trainSet, List<DigitSample> testSet, String foldName) {
        
        // 1. Nearest Neighbor (baseline)
        System.out.println("\n1. NEAREST NEIGHBOR (k=1) - BASELINE:");
        NearestNeighbor nnClassifier = new NearestNeighbor();
        nnClassifier.train(trainSet);
        double nnAccuracy = EvaluationMetrics.evaluateClassifier(nnClassifier, testSet);
        
        // 2. k-NN with different k values (hyperparameter tuning)
        System.out.println("\n2. k-NEAREST NEIGHBORS HYPERPARAMETER ANALYSIS:");
        double bestAccuracy = 0;
        int bestK = 1;
        
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
        
        // 3. Weighted k-NN with best k
        System.out.println("3. WEIGHTED k-NN (distance-based voting):");
        KNearestNeighbors weightedKNN = new KNearestNeighbors(bestK, true);
        weightedKNN.train(trainSet);
        double weightedAccuracy = EvaluationMetrics.evaluateClassifier(weightedKNN, testSet);
        
        // Summary for this fold
        System.out.println("\n" + foldName + " SUMMARY:");
        System.out.printf("  Nearest Neighbor (baseline): %.2f%%\n", nnAccuracy);
        System.out.printf("  Best k-NN (k=%d): %.2f%%\n", bestK, bestAccuracy);
        System.out.printf("  Weighted k-NN (k=%d): %.2f%%\n", bestK, weightedAccuracy);
        System.out.println();
        
        return new FoldResults(nnAccuracy, bestAccuracy, bestK, weightedAccuracy);
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
        double avgKNNAccuracy = (fold1Results.bestKNNAccuracy + fold2Results.bestKNNAccuracy) / 2.0;
        double avgWeightedAccuracy = (fold1Results.weightedAccuracy + fold2Results.weightedAccuracy) / 2.0;
        
        System.out.println("Average Performance Across Both Folds:");
        System.out.printf("  Nearest Neighbor (baseline): %.2f%%\n", avgNNAccuracy);
        System.out.printf("  k-NN (best k): %.2f%%\n", avgKNNAccuracy);
        System.out.printf("  Weighted k-NN: %.2f%%\n", avgWeightedAccuracy);
        
        System.out.println("\nBest k values found:");
        System.out.printf("  Fold 1: k=%d\n", fold1Results.bestK);
        System.out.printf("  Fold 2: k=%d\n", fold2Results.bestK);
        
        // Determine best overall algorithm
        String bestAlgorithm;
        double bestOverallAccuracy;
        
        if (avgWeightedAccuracy >= avgKNNAccuracy && avgWeightedAccuracy >= avgNNAccuracy) {
            bestAlgorithm = "Weighted k-NN";
            bestOverallAccuracy = avgWeightedAccuracy;
        } else if (avgKNNAccuracy >= avgNNAccuracy) {
            bestAlgorithm = "k-NN";
            bestOverallAccuracy = avgKNNAccuracy;
        } else {
            bestAlgorithm = "Nearest Neighbor";
            bestOverallAccuracy = avgNNAccuracy;
        }
        
        System.out.printf("\nBest performing algorithm: %s (%.2f%% average accuracy)\n", 
                         bestAlgorithm, bestOverallAccuracy);
        
        System.out.println("\n=== Analysis Complete ===");
    }
    
    /**
     * Helper class to store results from a single fold evaluation.
     */
    private static class FoldResults {
        final double nnAccuracy;
        final double bestKNNAccuracy;
        final int bestK;
        final double weightedAccuracy;
        
        FoldResults(double nnAccuracy, double bestKNNAccuracy, int bestK, double weightedAccuracy) {
            this.nnAccuracy = nnAccuracy;
            this.bestKNNAccuracy = bestKNNAccuracy;
            this.bestK = bestK;
            this.weightedAccuracy = weightedAccuracy;
        }
    }
}
