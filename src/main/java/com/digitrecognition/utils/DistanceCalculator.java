package com.digitrecognition.utils;

/**
 * Utility class for calculating distances between feature vectors.
 * Provides various distance metrics commonly used in machine learning.
 * 
 * @author Dumitru Nirca
 */
public class DistanceCalculator {
    
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
