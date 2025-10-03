package com.digitrecognition.algorithms;

import java.util.*;
import com.digitrecognition.models.DigitSample;
import com.digitrecognition.models.ClassificationResult;
import com.digitrecognition.models.NeighborDistance;
import com.digitrecognition.utils.DistanceCalculator;

/**
 * k-Nearest Neighbors classifier implementation.
 * Supports both majority voting and distance-weighted voting.
 * 
 * @author Dumitru Nirca
 */
public class KNearestNeighbors implements Classifier {
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
