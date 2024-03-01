import time

import numpy as np
import os
from sklearn.decomposition import PCA as RandomizedPCA
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

Dataset = "Dataset"
alpha_values = [0.8, 0.85, 0.9, 0.95]
k_values = [1, 3, 5, 7]


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


def split_dataset_50_50(dataMatrix, labelVector):
    dataMatrix_train = dataMatrix[::2]  # (odd rows)
    labelVector_train = labelVector[::2]
    dataMatrix_test = dataMatrix[1::2]  # (even rows)
    labelVector_test = labelVector[1::2]
    return dataMatrix_train, labelVector_train, dataMatrix_test, labelVector_test


def split_dataset_70_30(dataMatrix, labelVector):
    # Assuming dataMatrix and labelVector are both numpy arrays
    subjects = np.unique(labelVector)
    dataMatrix_train = []
    labelVector_train = []
    dataMatrix_test = []
    labelVector_test = []

    for subject in subjects:
        # Extract indices of instances belonging to the current subject
        indices = np.where(labelVector == subject)[0]

        # Split indices into training and test sets
        train_indices = indices[:7]  # 7 instances for training
        test_indices = indices[7:10]  # 3 instances for testing

        # Add training data and labels
        dataMatrix_train.append(dataMatrix[train_indices])
        labelVector_train.extend(labelVector[train_indices])

        # Add test data and labels
        dataMatrix_test.append(dataMatrix[test_indices])
        labelVector_test.extend(labelVector[test_indices])

    # Concatenate lists into numpy arrays
    dataMatrix_train = np.concatenate(dataMatrix_train)
    dataMatrix_test = np.concatenate(dataMatrix_test)
    labelVector_train = np.array(labelVector_train)
    labelVector_test = np.array(labelVector_test)

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
    # Iterate over each value of k
    for k in k_values:
        classifier = KNeighborsClassifier(n_neighbors=k, weights='distance')
        # Train the classifier using the training set
        classifier.fit(projectedData_train, labelVector_train)

        # Predict labels for the test set
        predicted_labels = classifier.predict(projectedData_test)

        # Calculate accuracy
        accuracy = np.mean(predicted_labels == labelVector_test)
        print("Accuracy for k =", k, ":", accuracy)


# Read Data
dataMatrix, labelVector = createDataMatrixLabelVector(Dataset)
# Split Data
dataMatrix_train, labelVector_train, dataMatrix_test, labelVector_test = split_dataset_50_50(dataMatrix, labelVector)
dataMatrix_train73, labelVector_train73, dataMatrix_test73, labelVector_test73 = split_dataset_70_30(dataMatrix,
                                                                                                     labelVector)

# print("Data Matrix shape:", dataMatrix.shape)
# print("Label vector shape:", labelVector.shape)
# print("50% 50% Data Matrix test shape:", dataMatrix_test.shape)
# print("50% 50% Label vector test shape:", labelVector_test.shape)
# print("50% 50% Data Matrix train shape:", dataMatrix_train.shape)
# print("50% 50% Label vector train shape:", labelVector_train.shape)
# print("70% 30% Data Matrix test shape:", dataMatrix_test73.shape)
# print("70% 30% Label vector test shape:", labelVector_test73.shape)
# print("70% 30% Data Matrix train shape:", dataMatrix_train73.shape)
# print("70% 30% Label vector train shape:", labelVector_train73.shape)
print("-------------------------------------------------")
print("50% 50% split :")
print("-------------------------------------------------")
for alpha in alpha_values:
    print("For alpha : ", alpha)
    # Run PCA
    timePCA_start = time.time()
    projectedData_train, projectionMatrix = PCA(dataMatrix_train, alpha)
    timePCA_end = time.time()
    # Project test Data
    projectedData_test = centralizeData(dataMatrix_test) @ projectionMatrix
    # Test data and Calculate Accuracy
    print("-------------------------------------------------")
    print("PCA :    time : ", (timePCA_end - timePCA_start))
    test(projectedData_train, projectedData_test, labelVector_train, labelVector_test)
    timeRPCA_start = time.time()
    rpca = RandomizedPCA(n_components=projectionMatrix.shape[1], svd_solver='randomized', random_state=42)
    projectedData_train = rpca.fit_transform(dataMatrix_train)
    timeRPCA_end = time.time()
    print("-------------------------------------------------")
    print("Randomized PCA :    time : ", (timeRPCA_end - timeRPCA_start))
    projectedData_test = rpca.transform(dataMatrix_test)
    test(projectedData_train, projectedData_test, labelVector_train, labelVector_test)
    print("---------------------------------------------------------------------------")

print("70% 30% split :")
print("-------------------------------------------------")
for alpha in alpha_values:
    print("For alpha : ", alpha)
    # Run PCA
    projectedData_train, projectionMatrix = PCA(dataMatrix_train73, alpha)
    # Project test  Data
    projectedData_test = centralizeData(dataMatrix_test73) @ projectionMatrix
    # Test data and Calculate Accuracy
    test(projectedData_train, projectedData_test, labelVector_train73, labelVector_test73)
    print("---------------------------------------------------------------------------")