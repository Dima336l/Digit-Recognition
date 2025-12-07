# Digit Recognition System

**CST 3170 Machine Learning Coursework**  
**Author:** Dumitru Nirca  
**Date:** December 2025

---

## Overview

A comprehensive machine learning system for handwritten digit recognition (0-9) using the UCI Machine Learning Repository dataset. The system implements multiple classification algorithms and performs rigorous evaluation using two-fold cross-validation.

## Features

- **Multiple ML Algorithms**: Implements 4 different classification algorithms
- **Hyperparameter Tuning**: Automatic optimization of k for k-NN and λ/epochs/η for Linear SVM
- **Comprehensive Evaluation**: Two-fold cross-validation with detailed metrics
- **Performance Metrics**: Accuracy, confusion matrices, precision, recall, F1-scores
- **Single-File Architecture**: All code consolidated in one Java file for easy deployment

## Algorithms Implemented

1. **Nearest Neighbor (Baseline)**
   - 1-Nearest Neighbor with Euclidean distance
   - Serves as baseline for comparison
   - Lazy learning approach

2. **k-Nearest Neighbors (k-NN)**
   - Majority voting among k nearest neighbors
   - Hyperparameter tuning: Tests k values {1, 3, 5, 7, 9, 11}
   - Automatically selects optimal k based on validation performance

3. **Weighted k-Nearest Neighbors**
   - Distance-weighted voting mechanism
   - Closer neighbors have more influence: `weight = 1 / (distance + ε)`
   - Uses the optimal k value found during hyperparameter tuning

4. **Linear Support Vector Machine (SVM)**
   - One-vs-one multiclass classification (pairwise, 45 binary classifiers) with Pegasos-style SGD
   - Feature engineering: spatial augmentation, Random Fourier Features (512), polynomial features
   - Ensemble of 5 models with voting for final predictions
   - Z-score normalization plus runtime-selected regularization, epoch count, and minimum learning rate
   - Soft-margin SVM with L2 regularization and confidence scores derived from decision values

## Dataset

- **Source**: UCI Machine Learning Repository
- **Format**: CSV files with 64 features + 1 label
- **Features**: 8×8 pixel grayscale values (64 features per sample)
- **Classes**: 10 digit classes (0-9)
- **Files**: `dataSet1.csv` and `dataSet2.csv`

## Project Structure

```
Digit-Recognition-Eclipse-backup/
├── src/
│   └── DigitRecognitionApp.java    # Main application (all classes in one file)
├── bin/                             # Compiled class files
├── dataSet1.csv                     # Training/test dataset 1
├── dataSet2.csv                     # Training/test dataset 2
├── README.md                        # This file
└── Digit_Recognition_Report.md      # Detailed coursework report
```

## Requirements

- **Java**: JDK 8 or higher
- **IDE**: Eclipse (recommended) or any Java IDE
- **Data Files**: `dataSet1.csv` and `dataSet2.csv` in the project root

## How to Run

### Option 1: Using Eclipse

1. **Import Project**:
   - Open Eclipse
   - File → Import → Existing Projects into Workspace
   - Select the project directory
   - Click Finish

2. **Run the Application**:
   - Right-click on `DigitRecognitionApp.java`
   - Run As → Java Application
   - Or use the main method's Run button

### Option 2: Command Line

1. **Compile**:
   ```bash
   javac -d bin -sourcepath src src/DigitRecognitionApp.java
   ```

2. **Run**:
   ```bash
   java -cp bin DigitRecognitionApp
   ```

**Note**: Ensure `dataSet1.csv` and `dataSet2.csv` are in the project root directory.

## Evaluation Methodology

### Two-Fold Cross-Validation

The system performs two-fold cross-validation:
- **Fold 1**: Train on Dataset 1, Test on Dataset 2
- **Fold 2**: Train on Dataset 2, Test on Dataset 1

This ensures robust evaluation and better utilization of available data.

### Performance Metrics

For each algorithm, the system reports:
- **Overall Accuracy**: Percentage of correctly classified samples
- **Confusion Matrix**: 10×10 matrix showing classification errors
- **Per-Class Metrics**: Precision, Recall, F1-Score, and Support for each digit
- **Macro Averages**: Average performance across all classes

## Expected Output

The program generates comprehensive output including:

1. Dataset analysis (class distribution)
2. Algorithm evaluation for each fold:
   - Nearest Neighbor results
   - k-NN hyperparameter analysis
   - Weighted k-NN results
   - Linear SVM results
   - Confusion matrices
   - Per-class performance metrics
3. Overall summary with average performance across both folds
4. Best performing algorithm identification

## Code Architecture

The application is organized into several logical sections within a single file:

### Model Classes
- `DigitSample`: Represents a single digit sample with features and label
- `ClassificationResult`: Stores prediction results with confidence scores
- `NeighborDistance`: Helper class for k-NN algorithms (implements `Comparable`)

### Algorithm Classes
- `Classifier`: Interface defining the contract for all algorithms
- `NearestNeighbor`: Baseline 1-NN implementation
- `KNearestNeighbors`: k-NN with majority and weighted voting
- `LinearSVM`: Linear Support Vector Machine with Pegasos optimization

### Utility Classes
- `DatasetLoader`: Handles CSV file parsing and dataset operations
- `DistanceCalculator`: Provides distance metrics (Euclidean, Manhattan, Minkowski)
- `EvaluationMetrics`: Generates confusion matrices and performance reports

## Key Features

### Hyperparameter Tuning
The system automatically tests multiple k values and selects the optimal one:
```java
K_VALUES = {1, 3, 5, 7, 9, 11}
```
- Linear SVM also runs a grid search per fold across:
  ```java
  SVM_LAMBDA_CANDIDATES = {0.0003, 0.00035, 0.0004, 0.00045, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.0010, 0.0012}
  SVM_EPOCH_CANDIDATES = {100, 120, 140, 160, 180, 200, 220, 240}
  SVM_MIN_LR_CANDIDATES = {5e-8, 7.5e-8, 1e-7, 1.25e-7, 1.5e-7}
  SVM_ENSEMBLE_SIZE = 5
  SVM_VALIDATION_REPEATS = 6
  ```
  with repeated train/validation splits (6 repeats per candidate) to average out noise before picking the best 5 models for ensemble voting.

### Distance Metrics
- **Euclidean Distance**: Primary metric for all algorithms
- **Manhattan Distance**: Available for alternative implementations
- **Minkowski Distance**: Generalized distance metric

### Normalization
- Linear SVM standardizes each feature (z-score) and then scales vectors to unit length, improving SGD stability.
- Normalization parameters are computed on the training fold and reused when evaluating the test fold.

## Performance

Latest two-fold averages (January 2026 run):
- **Nearest Neighbor (k=1)**: 98.26 %
- **k-NN (best k per fold)**: 98.27 %
- **Weighted k-NN**: 98.24 %
- **Linear SVM (tuned Pegasos)**: 97.30 %

*Results are reproducible via `java -cp bin DigitRecognitionApp` with the provided datasets.*

## Code Quality

- **Object-Oriented Design**: Clear interfaces and modular architecture
- **Comprehensive Documentation**: JavaDoc comments for all classes and methods
- **Error Handling**: Input validation and exception management
- **Best Practices**: Defensive copying, meaningful variable names, named constants
- **Extensibility**: Easy to add new algorithms via the `Classifier` interface

## Extending the System

To add a new algorithm:

1. Implement the `Classifier` interface
2. Implement required methods: `train()`, `classify()`, `getAlgorithmName()`
3. Add algorithm evaluation in `evaluateAllAlgorithms()` method
4. Update `FoldResults` class if needed
5. Update summary methods to include new algorithm

## Troubleshooting

### "Selection does not contain a main type"
- Ensure `DigitRecognitionApp.java` is in the `src/` folder
- Check that the main method is properly defined
- Clean and rebuild the project in Eclipse

### File Not Found Errors
- Verify `dataSet1.csv` and `dataSet2.csv` are in the project root
- Check file paths in the code match your directory structure

### Compilation Errors
- Ensure Java JDK 8+ is installed
- Check that all required imports are present
- Clean and rebuild the project

## License

This project is part of CST 3170 Machine Learning coursework.

## References

1. UCI Machine Learning Repository - Optical Recognition of Handwritten Digits Dataset
2. Duda, R.O., Hart, P.E., & Stork, D.G. (2001). Pattern Classification. Wiley-Interscience.
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.

---

**For detailed analysis and results, see `Digit_Recognition_Report.md`**

