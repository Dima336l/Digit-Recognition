package com.digitrecognition.utils;

import java.io.*;
import java.util.*;
import com.digitrecognition.models.DigitSample;

/**
 * Utility class for loading and managing datasets.
 * Handles CSV file parsing and dataset operations.
 * 
 * @author Dumitru Nirca
 */
public class DatasetLoader {
    private static final int NUM_FEATURES = 64; // 8x8 pixel values
    
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
                } catch (Exception e) {
                    throw new IOException("Error parsing line " + lineNumber + ": " + e.getMessage(), e);
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
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid feature value at position " + i + ": " + parts[i]);
            }
        }
        
        // Extract label (last value)
        int label;
        try {
            label = Integer.parseInt(parts[NUM_FEATURES]);
        } catch (NumberFormatException e) {
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
        int[] classCounts = new int[10]; // 10 digit classes (0-9)
        
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
        
        for (int i = 0; i < 10; i++) {
            double percentage = (classCounts[i] * 100.0) / dataset.size();
            System.out.printf("  Digit %d: %4d samples (%.1f%%)\n", i, classCounts[i], percentage);
        }
        System.out.println();
    }
}
