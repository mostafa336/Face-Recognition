import numpy as np
import os
import torch
import time
import random
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA2

Dataset = "Dataset"
k_values = [1, 3, 5, 7]

num_nonFaces_values = [25, 50, 100 , 150, 200, 250, 300, 350]
colors = ['red', 'blue', 'green', 'orange']
accuracies = []
num_comp=[]
times=[]
predicted_labels = []
idCouldDetectFace = []
idCouldNotDetectFace = []
idCouldDetectNonFace = []
idCouldNotDetectNonFace = []
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


def createDataMatrixLabelVector_FNF():
    Faces = "Dataset"
    NonFaces = "nonfaces_dataset"
    images = []
    labels = []
    for subject_id in os.listdir(Faces):
        subject_path = os.path.join(Faces, subject_id)
        if os.path.isdir(subject_path):
            for filename in os.listdir(subject_path):
                img_path = os.path.join(subject_path, filename)
                img = Image.open(img_path).convert('L')
                img_array = np.array(img)
                img_vector = img_array.flatten()
                images.append(img_vector)
                labels.append(1)
    for subject_id in os.listdir(NonFaces):
        subject_path = os.path.join(NonFaces, subject_id)
        if os.path.isdir(subject_path):
            for filename in os.listdir(subject_path):
                img_path = os.path.join(subject_path, filename)
                img = Image.open(img_path).convert('L')
                img = img.resize((92, 112))
                img_array = np.array(img)
                img_vector = img_array.flatten()
                images.append(img_vector)
                labels.append(0)
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


def split_data_FNF(data_matrix, label_vector, num_nonFaces):
    # Define the number of samples for training
    training_start = 200
    training_end = 400 + num_nonFaces

    # Split the data and labels
    X_train = data_matrix[training_start:training_end]
    X_test = data_matrix[:200]
    X_test = np.concatenate((X_test, data_matrix[-200:]), axis=0)

    y_train = label_vector[training_start:training_end]
    y_test = label_vector[:200]
    y_test = np.concatenate((y_test, label_vector[-200:]), axis=0)

    return X_train, y_train, X_test, y_test

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
    S_B = np.zeros((col, col))  # initialize S_B square matrix with zeros

    for class_label, mean_vector in class_means.items():
        # calculate S_W
        class_indices = np.where(label_vector == class_label)[0]
        z = np.subtract(data_matrix[class_indices], mean_vector)
        S_W += np.dot(np.transpose(z), z)

        # calculate S_B
        nk = class_sizes[class_label]
        diff = np.array([mean_vector - overall_mean])
        S_B += nk * diff.T @ diff

    return S_W, S_B


def computeEigen(cov):
    cov_tensor = torch.tensor(cov, dtype=torch.float32)
    eigenvalues, eigenvectors = torch.linalg.eig(cov_tensor)
    real_eigenvalues = np.real(eigenvalues)
    real_eigenvectors = np.real(eigenvectors)
    sorted_indices = torch.argsort(real_eigenvalues, descending=True)
    sorted_eigenvalues = real_eigenvalues[sorted_indices]
    sorted_eigenvectors = real_eigenvectors[:, sorted_indices]

    sorted_eigenvalues = sorted_eigenvalues.numpy()
    sorted_eigenvectors = sorted_eigenvectors.numpy()

    return sorted_eigenvalues, sorted_eigenvectors


def chooseDimensionality(eigenvalues, eigenvectors, num_components):
    # Choose the top 'num_components' eigenvalues and eigenvectors
    selected_eigenvalues = eigenvalues[:num_components]
    selected_eigenvectors = eigenvectors[:, :num_components]
    return selected_eigenvalues, selected_eigenvectors



def LDA(data_matrix_train, label_vector_train, class_sizes, num_components):
    # calculate each class mean and overall mean
    class_means = calculate_class_means(data_matrix_train, label_vector_train)
    overall_mean = np.mean(data_matrix_train, axis=0)

    # calculate within and between class scatter matrices
    S_W, S_B = calculate_within_and_between_class_scatter_matrices(data_matrix_train, label_vector_train,
                                                                   class_means, overall_mean, class_sizes)

    S_W_torch = torch.tensor(S_W)
    S_W_inv = torch.pinverse(S_W_torch).numpy()
    result = S_W_inv @ S_B

    eigenvalues, eigenvectors = computeEigen(result)

    selected_eigenvalues, selected_eigenvectors = chooseDimensionality(eigenvalues,
                                                                       eigenvectors, num_components)

    return selected_eigenvectors


def test(projectedData_train, projectedData_test, labelVector_train, labelVector_test,ExtractID):
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
        if ExtractID:
            ExtractIDs(predicted_labels, labelVector_test)

def lda_variation(data_matrix_train, label_vector_train, dataMatrix_test, labelVector_test):
    lda2 = LDA2(solver= 'svd')

    projectedData_train = lda2.fit_transform(data_matrix_train, label_vector_train)
    projectedData_test = lda2.transform(dataMatrix_test)

    knn = KNeighborsClassifier(n_neighbors=1, weights='distance')
    knn.fit(projectedData_train, label_vector_train)
    predicted_labels = knn.predict(projectedData_test)

    accuracy = np.mean(predicted_labels == labelVector_test)
    return accuracy


def runLDA_FNF(ExtractId):
    start = time.time()
    dataMatrix, labelVector = createDataMatrixLabelVector_FNF()
    for num_nonFaces in num_nonFaces_values:
        dataMatrix_train, labelVector_train, dataMatrix_test, labelVector_test = split_data_FNF(dataMatrix, labelVector,
                                                                                                num_nonFaces)
        class_sizes = {class_label: np.sum(labelVector_train == class_label)
                       for class_label in np.unique(labelVector_train)}
        projectionMatrix = LDA(dataMatrix_train, labelVector_train,class_sizes, 50)
        overall_mean = np.mean(dataMatrix_train, axis=0)

        Z_train = dataMatrix_train - overall_mean
        projectedData_train = Z_train @ projectionMatrix

        Z_test = dataMatrix_test - overall_mean
        projectedData_test = Z_test @ projectionMatrix
        # Test data and Calculate Accuracy
        test(projectedData_train, projectedData_test, labelVector_train, labelVector_test,ExtractId)
        end = time.time()
        print("LDA time:", end - start,'--'*30)
        if ExtractId:
           return dataMatrix_test


def runLDA50_50():
    print("50%,50%")
    start = time.time()
    data_matrix, label_vector = create_data_matrix_label_vector(Dataset)
    data_matrix_train, labelVector_train, dataMatrix_test, labelVector_test = split_dataset_50_50(data_matrix,
                                                                                                  label_vector)
    class_sizes = {class_label: np.sum(labelVector_train == class_label)
                   for class_label in np.unique(labelVector_train)}
    projectionMatrix = LDA(data_matrix_train, labelVector_train, class_sizes, 39)
    overall_mean = np.mean(data_matrix_train, axis=0)

    Z_train = data_matrix_train - overall_mean
    projectedData_train = Z_train @ projectionMatrix

    Z_test = dataMatrix_test - overall_mean
    projectedData_test = Z_test @ projectionMatrix

    test(projectedData_train, projectedData_test, labelVector_train, labelVector_test,False)
    end = time.time()
    print("LDA time:", end - start)

def runLDAVariation():
    start = time.time()
    data_matrix, label_vector = create_data_matrix_label_vector(Dataset)

    data_matrix_train, labelVector_train, dataMatrix_test, labelVector_test = split_dataset_50_50(data_matrix,
                                                                                                  label_vector)

    accuracy2 = lda_variation(data_matrix_train, labelVector_train, dataMatrix_test, labelVector_test)
    print("LDA variation accuracy at k = 1:", accuracy2)
    end = time.time()
    print("LDA variation Time:", end - start)


def runLDA70_30():
    print("70, 30")
    start = time.time()
    data_matrix, label_vector = create_data_matrix_label_vector(Dataset)
    data_matrix_train73, labelVector_train73, dataMatrix_test73, labelVector_test73 = split_dataset_70_30(data_matrix,
                                                                                                          label_vector)
    class_sizes = {class_label: np.sum(labelVector_train73 == class_label)
                   for class_label in np.unique(labelVector_train73)}

    projectionMatrix73 = LDA(data_matrix_train73, labelVector_train73, class_sizes, 39)

    overall_mean73 = np.mean(data_matrix_train73, axis=0)

    Z_train73 = data_matrix_train73 - overall_mean73
    projectedData_train73 = Z_train73 @ projectionMatrix73

    Z_test73 = dataMatrix_test73 - overall_mean73
    projectedData_test73 = Z_test73 @ projectionMatrix73

    test(projectedData_train73, projectedData_test73, labelVector_train73, labelVector_test73,False)

    end = time.time()

    print("Time in seconds ", (end - start))

def ExtractIDs(predictedLabel, label_test):
    for i in range(len(predictedLabel)):
        if predictedLabel[i] == label_test[i] and label_test[i] == 1:
            idCouldDetectFace.append(i)
        elif predictedLabel[i] == label_test[i] and label_test[i] == 0:
            idCouldDetectNonFace.append(i)
        if predictedLabel[i] != label_test[i] and label_test[i] == 1:
            idCouldNotDetectFace.append(i)
        elif predictedLabel[i] != label_test[i] and label_test[i] == 0:
            idCouldNotDetectNonFace.append(i)
def plotSuccessFailureCases():
    idCouldDetectFace.clear()
    idCouldNotDetectFace.clear()
    idCouldNotDetectNonFace.clear()
    idCouldDetectNonFace.clear()
    dataMatrix = runLDA_FNF(True)
    fig, axes = plt.subplots(4, 5, figsize=(25, 25))
    axes = axes.flatten()
    axeId = 0
    Ids = random.sample(idCouldDetectFace, 5)
    for id in Ids:
        image = dataMatrix[id].reshape((112, 92))
        axes[axeId].imshow(image, cmap='gray')
        axes[axeId].axis('off')
        axes[axeId].set_title(f"Face Successfully Detected", fontweight='bold', fontsize=20)

        axeId = axeId + 1
    Ids = random.sample(idCouldDetectNonFace, 5)
    for id in Ids:
        image = dataMatrix[id].reshape((112, 92))
        axes[axeId].imshow(image, cmap='gray')
        axes[axeId].axis('off')
        axes[axeId].set_title(f"Non Face Successfully Detected", fontweight='bold', fontsize=18)
        axeId = axeId + 1
    Ids = random.sample(idCouldNotDetectFace, 1)
    for id in Ids:
        image = dataMatrix[id].reshape((112, 92))
        axes[axeId].imshow(image, cmap='gray')
        axes[axeId].axis('off')
        axes[axeId].set_title(f"Failed To Detect Face", fontweight='bold', fontsize=20)
        axeId = axeId + 1
    Ids = random.sample(idCouldNotDetectNonFace, 9)
    for id in Ids:
        image = dataMatrix[id].reshape((112, 92))
        axes[axeId].imshow(image, cmap='gray')
        axes[axeId].axis('off')
        axes[axeId].set_title(f"Failed To Detect Non Face", fontweight='bold', fontsize=20)
        axeId = axeId + 1
    plt.tight_layout()
    plt.show()
    accuracies.clear()

# Run LDA algorithm
# runLDA50_50()
# runLDA70_30()
# runLDAVariation()
# runLDA_FNF(False)
plotSuccessFailureCases()