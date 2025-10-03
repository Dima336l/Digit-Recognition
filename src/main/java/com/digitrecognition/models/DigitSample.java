package com.digitrecognition.models;

/**
 * Data structure to represent a single digit sample with features and label.
 * Each sample contains 64 features (8x8 pixel values) and a digit label (0-9).
 * 
 * @author Dumitru Nirca
 */
public class DigitSample {
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
