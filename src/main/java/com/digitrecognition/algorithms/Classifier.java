package com.digitrecognition.algorithms;

import java.util.List;
import com.digitrecognition.models.DigitSample;
import com.digitrecognition.models.ClassificationResult;

/**
 * Interface for digit classification algorithms.
 * Defines the contract that all classification algorithms must implement.
 * 
 * @author Dumitru Nirca
 */
public interface Classifier {
    
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
