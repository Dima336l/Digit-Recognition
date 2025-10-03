package com.digitrecognition.models;

/**
 * Data structure to hold the results of a classification operation.
 * Contains the predicted label, confidence score, and optional additional metrics.
 * 
 * @author Dumitru Nirca
 */
public class ClassificationResult {
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
