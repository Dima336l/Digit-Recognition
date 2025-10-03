package com.digitrecognition.algorithms;

import java.util.List;
import com.digitrecognition.models.DigitSample;
import com.digitrecognition.models.ClassificationResult;

/**
 * Nearest Neighbor classifier (k=1) implementation.
 * This is a specialized version of k-NN for the baseline case.
 * 
 * @author Dumitru Nirca
 */
public class NearestNeighbor implements Classifier {
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
