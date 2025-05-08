# k-NN Classifier from Scratch (Python/NumPy)

## Project Overview

This project demonstrates the fundamentals of the k-NN algorithm by:

*   Implementing Euclidean distance calculation between data points.
*   Developing a k-NN classifier function to predict class labels based on the majority class among the 'k' nearest neighbors in the training data.
*   Evaluating classifier performance using classification accuracy.
*   Analyzing the impact of the 'k' parameter on model performance, including identifying potential overfitting and underfitting by comparing training set and test set accuracies across different 'k' values.
*   Applying the classifier to various datasets (Synthetic, Wine Quality, Phoneme) loaded from HDF5 files.

## Key Features & Implementations

*   The core classification logic uses basic NumPy operations without relying on high-level ML libraries like scikit-learn for the k-NN part itself.
*   Uses `h5py` to load datasets.
*   Employs `matplotlib` to plot accuracy vs. 'k' graphs for analysis.
*   Includes a comparison with Linear Regression using `sklearn.linear_model.LinearRegression` on the Wine dataset, evaluating Mean Squared Error (MSE) and R² score.
*   Uses `sklearn.preprocessing.MinMaxScaler` for feature normalization on real-world datasets.
*   Includes correlation matrix visualization using `pandas` and `seaborn` for feature analysis in Exercises 2 and 3.

This project serves as a practical exercise in understanding and implementing a fundamental classification algorithm and evaluating its characteristics.
