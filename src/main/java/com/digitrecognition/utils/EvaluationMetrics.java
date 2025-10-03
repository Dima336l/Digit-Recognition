package com.digitrecognition.utils;

import java.util.List;
import com.digitrecognition.models.DigitSample;
import com.digitrecognition.models.ClassificationResult;
import com.digitrecognition.algorithms.Classifier;

/**
 * Utility class for calculating and displaying evaluation metrics.
 * Provides confusion matrix generation, accuracy calculation, and performance reporting.
 * 
 * @author Dumitru Nirca
 */
public class EvaluationMetrics {
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
