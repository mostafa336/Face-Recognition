import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

Dataset = "Dataset"
alpha = 0.9


def createDataMatrixLabelVector(directory):
    images = []
    labels = []
    for subject_id in os.listdir(directory):
        subject_path = os.path.join(directory, subject_id)
        if os.path.isdir(subject_path):
            for filename in os.listdir(subject_path):
                img_path = os.path.join(subject_path, filename)
                img = Image.open(img_path).convert('L')
                img_array = np.array(img)
                img_vector = img_array.flatten()
                images.append(img_vector)
                labels.append(int(subject_id[1:]))
    return np.array(images), np.array(labels)


def split_dataset(dataMatrix, labelVector):
    dataMatrix_train = dataMatrix[::2]  # (odd rows)
    labelVector_train = labelVector[::2]
    dataMatrix_test = dataMatrix[1::2]  # (even rows)
    labelVector_test = labelVector[1::2]
    return dataMatrix_train, labelVector_train, dataMatrix_test, labelVector_test


def centralizeData(dataMatrix):
    mean = np.mean(dataMatrix, axis=0)
    return dataMatrix - mean


def computeEigen(cov):
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors


def chooseDimensionality(eigenvalues, eigenvectors, alpha):
    cumulative_sum = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    num_components = 200
    for i in range(len(cumulative_sum)):
        if cumulative_sum[i] >= alpha:
            num_components = i + 1
            break
    return eigenvectors[:, :num_components]


def normalizeEigenvectors(eigenvectors):
    for i in range(eigenvectors.shape[1]):
        eigenvector = eigenvectors[:, i]
        magnitude = np.linalg.norm(eigenvector)
        eigenvectors[:, i] /= magnitude


def PCA(D, alpha):
    Z = centralizeData(D)
    cov = Z @ Z.T  # produce a 200x200 matrix instead of the 10304x10304 produced by Z.T @ Z
    eigenvalues, eigenvectors = computeEigen(
        cov)  # same eigenvalues of the 10304x10304 but the eigenvectors need some computations
    eigenvectors = chooseDimensionality(eigenvalues, eigenvectors, alpha)
    eigenvectors = Z.T @ eigenvectors  # eigenvectors converted to match the 10304x10304 eigenvectors
    normalizeEigenvectors(eigenvectors)
    projectedData = Z @ eigenvectors
    return projectedData, eigenvectors


def test(projectedData_train, projectedData_test, labelVector_train, labelVector_test):
    classifier = KNeighborsClassifier(n_neighbors=1)
    # Train the classifier using the projected training set
    classifier.fit(projectedData_train, labelVector_train)
    # Predict labels for the projected test set
    predicted_labels = classifier.predict(projectedData_test)
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == labelVector_test)
    return accuracy


# Read Data
dataMatrix, labelVector = createDataMatrixLabelVector(Dataset)
# Split Data
dataMatrix_train, labelVector_train, dataMatrix_test, labelVector_test = split_dataset(dataMatrix, labelVector)
# Run PCA
projectedData_train, projectionMatrix = PCA(dataMatrix_train, alpha)
# Project test Data
projectedData_test = centralizeData(dataMatrix_test) @ projectionMatrix
# Test data and Calculate Accuracy
accuracy = test(projectedData_train, projectedData_test, labelVector_train, labelVector_test)

print("Data Matrix shape:", dataMatrix.shape)
print("Label vector shape:", labelVector.shape)
print("Data Matrix test shape:", dataMatrix_test.shape)
print("Label vector test shape:", labelVector_test.shape)
print("Data Matrix train shape:", dataMatrix_train.shape)
print("Label vector train shape:", labelVector_train.shape)
print("Projection Matrix shape:", projectionMatrix.shape)
print("Projected train data shape ", projectedData_train.shape)
print("Projected test data shape ", projectedData_test.shape)
print("Accuracy :", accuracy, " Alpha : ", alpha)
