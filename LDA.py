import numpy as np
# import cupy as cp
import os
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

import time

Dataset = "Dataset"


def create_data_matrix_label_vector(directory):
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


def split_dataset(data_matrix, label_vector):
    data_matrix_train = data_matrix[::2]  # (odd rows)
    label_vector_train = label_vector[::2]
    data_matrix_test = data_matrix[1::2]  # (even rows)
    label_vector_test = label_vector[1::2]
    return data_matrix_train, label_vector_train, data_matrix_test, label_vector_test


def calculate_class_means(data_matrix, label_vector):
    unique_classes = np.unique(label_vector)  # get unique classes names
    class_means = {}  # create dictionary (key: class_label and value: mean of each class)
    for class_label in unique_classes:
        class_indices = np.where(label_vector == class_label)[0]
        class_data = data_matrix[class_indices]
        class_mean = np.mean(class_data, axis=0)
        class_means[class_label] = class_mean

    return class_means


def calculate_within_and_between_class_scatter_matrices(data_matrix, label_vector, class_means,
                                          overall_mean, class_sizes):
    col = data_matrix.shape[1]  # number of features
    S_W = np.zeros((col, col))  # initialize S_W square matrix with zeros
    S_B = np.zeros((col, col))  # initialize S_لآ square matrix with zeros

    for class_label, mean_vector in class_means.items():
        # calculate S_W
        class_indices = np.where(label_vector == class_label)[0]
        z = np.subtract(data_matrix[class_indices], mean_vector)
        S_W += np.dot(np.transpose(z), z)

        # calculate S_W
        nk = class_sizes[class_label]
        diff = np.array([mean_vector - overall_mean])
        S_B += nk * diff.T @ diff

    return S_W, S_B


def computeEigen(cov):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors


def chooseDimensionality(eigenvalues, eigenvectors, num_components):
    # Choose the top 'num_components' eigenvalues and eigenvectors
    selected_eigenvalues = eigenvalues[:num_components]
    selected_eigenvectors = eigenvectors[:, :num_components]
    return selected_eigenvalues, selected_eigenvectors


def normalizeEigenvectors(eigenvectors):
    for i in range(eigenvectors.shape[1]):
        eigenvector = eigenvectors[:, i]
        magnitude = np.linalg.norm(eigenvector)
        eigenvectors[:, i] /= magnitude


def LDA(data_matrix_train, label_vector_train, class_sizes):
    # calculate each class mean and overall mean
    class_means = calculate_class_means(data_matrix_train, label_vector_train)
    overall_mean = np.mean(data_matrix_train, axis=0)

    # calculate within and between class scatter matrices
    S_W, S_B = calculate_within_and_between_class_scatter_matrices(data_matrix_train, label_vector_train,
                                                                   class_means, overall_mean, class_sizes)

    # Get the inverse of S_W
    S_W_inv = np.linalg.inv(S_W)
    result = S_W_inv @ S_B

    eigenvalues, eigenvectors = computeEigen(result)

    # Choose the top 39 eigenvalues and eigenvectors
    num_components = 39
    selected_eigenvalues, selected_eigenvectors = chooseDimensionality(eigenvalues,
                                                                       eigenvectors, num_components)

    # eigenvectors = Z @ selected_eigenvectors
    # normalizeEigenvectors(eigenvectors)

    return class_means, selected_eigenvectors


def test(projectedData_train, projectedData_test, labelVector_train, labelVector_test):
    classifier = KNeighborsClassifier(n_neighbors=1)
    # Train the classifier using the projected training set
    classifier.fit(projectedData_train, labelVector_train)
    # Predict labels for the projected test set
    predicted_labels = classifier.predict(projectedData_test)
    # Calculate accuracy
    accuracy = np.mean(predicted_labels == labelVector_test)
    return accuracy


def centralize(data_matrix, label_vector, class_means):
    Z = np.zeros((data_matrix.shape[0], data_matrix.shape[1]))

    for class_label, mean_vector in class_means.items():
        class_indices = np.where(label_vector == class_label)[0]
        z = np.subtract(data_matrix[class_indices], mean_vector)
        Z[class_indices, :] = z

    return Z


start = time.time()
# Read Data
data_matrix, label_vector = create_data_matrix_label_vector(Dataset)
# Split Data
data_matrix_train, labelVector_train, dataMatrix_test, labelVector_test = split_dataset(data_matrix,
                                                                                        label_vector)
# Calculate number of samples in each class
class_sizes = {class_label: np.sum(labelVector_train == class_label)
               for class_label in np.unique(labelVector_train)}
# Run LDA algorithm
# LDA(data_matrix_train, label_vector_train, class_sizes)
class_means, projectionMatrix = LDA(data_matrix_train, labelVector_train, class_sizes)

Z_train = centralize(data_matrix_train,labelVector_train, class_means)
projectedData_train = Z_train @ projectionMatrix

Z_test = centralize(dataMatrix_test, labelVector_test, class_means)
projectedData_test = Z_test @ projectionMatrix

accuracy = test(projectedData_train, projectedData_test, labelVector_train, labelVector_test)

print(f"Projected data : {projectedData_train.shape}")
print(f"Projection matrix : {projectionMatrix.shape}")
print("accuracy: ", accuracy)

end = time.time()

print("Time in seconds ", (end - start))
