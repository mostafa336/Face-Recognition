import time
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA as RandomizedPCA
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import random

Dataset = "Dataset"
alpha_values = [0.8, 0.85, 0.9, 0.95]
k_values = [1, 3, 5, 7]
num_nonFaces = [25, 50, 100, 150, 200, 250, 300, 350]
colors = ['red', 'blue', 'green', 'orange']
accuracies = []
num_comp = []
times = []
idCouldDetectFace = []
idCouldNotDetectFace = []
idCouldDetectNonFace = []
idCouldNotDetectNonFace = []
predicted_labels = []


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
    print(np.array(images).shape)
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
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
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


def test(projectedData_train, projectedData_test, labelVector_train, labelVector_test, k_values, ExtractID):
    # Iterate over each value of k
    for k in k_values:
        classifier = KNeighborsClassifier(n_neighbors=k, weights='distance')
        # Train the classifier using the training set
        classifier.fit(projectedData_train, labelVector_train)

        # Predict labels for the test set
        predicted_labels = classifier.predict(projectedData_test)

        # Calculate accuracy
        accuracy = np.mean(predicted_labels == labelVector_test)
        accuracies.append(accuracy)
        print("Accuracy for k =", k, ":", accuracy)
        if ExtractID:
            ExtractIDs(predicted_labels, labelVector_test)


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
def runPCA50_50(k_values, alpha_values, RPCA):
    # # Read Data
    dataMatrix, labelVector = createDataMatrixLabelVector(Dataset)
    # Split Data
    dataMatrix_train, labelVector_train, dataMatrix_test, labelVector_test = split_dataset_50_50(dataMatrix,
                                                                                                 labelVector)
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
        mean = np.mean(dataMatrix_train, axis=0)
        centralizedDataMatrix_test = dataMatrix_test - mean
        projectedData_test = centralizedDataMatrix_test @ projectionMatrix
        # Test data and Calculate Accuracy
        print("-------------------------------------------------")
        print("PCA :    time : ", (timePCA_end - timePCA_start))
        times.append(timePCA_end - timePCA_start)
        test(projectedData_train, projectedData_test, labelVector_train, labelVector_test, k_values,False)
        if RPCA:
            num_comp.append(projectionMatrix.shape[1])
            timeRPCA_start = time.time()
            rpca = RandomizedPCA(n_components=projectionMatrix.shape[1], svd_solver='randomized', random_state=42)
            projectedData_train = rpca.fit_transform(dataMatrix_train)
            timeRPCA_end = time.time()
            print("-------------------------------------------------")
            print("Randomized PCA :    time : ", (timeRPCA_end - timeRPCA_start))
            times.append(timeRPCA_end - timeRPCA_start)
            projectedData_test = rpca.transform(dataMatrix_test)
            test(projectedData_train, projectedData_test, labelVector_train, labelVector_test, k_values,False)
            print("---------------------------------------------------------------------------")


def runPCA70_30(k_values, alpha_values):
    # # Read Data
    dataMatrix, labelVector = createDataMatrixLabelVector(Dataset)
    dataMatrix_train73, labelVector_train73, dataMatrix_test73, labelVector_test73 = split_dataset_70_30(dataMatrix,
                                                                                                         labelVector)
    print("70% 30% split :")
    print("-------------------------------------------------")
    for alpha in alpha_values:
        print("For alpha : ", alpha)
        # Run PCA
        projectedData_train, projectionMatrix = PCA(dataMatrix_train73, alpha)
        mean = np.mean(dataMatrix_train73, axis=0)
        centralizedDataMatrix_test = dataMatrix_test73 - mean
        projectedData_test = centralizedDataMatrix_test @ projectionMatrix
        # Test data and Calculate Accuracy
        test(projectedData_train, projectedData_test, labelVector_train73, labelVector_test73, k_values,False)
        print("---------------------------------------------------------------------------")


def runPCA_FNF(k_values, alpha_values, num_nonFaces_values, ExtractId):
    dataMatrix, labelVector = createDataMatrixLabelVector_FNF()
    for num_nonFaces in num_nonFaces_values:
        dataMatrix_train, labelVector_train, dataMatrix_test, labelVector_test = split_data_FNF(dataMatrix, labelVector,
                                                                                                num_nonFaces)
        for alpha in alpha_values:
            print("-------------------------------------------------")
            print("For alpha : ", alpha)
            projectedData_train, projectionMatrix = PCA(dataMatrix_train, alpha)
            # Project test Data
            mean = np.mean(dataMatrix_train, axis=0)
            centralizedDataMatrix_test = dataMatrix_test - mean
            projectedData_test = centralizedDataMatrix_test @ projectionMatrix
            # Test data and Calculate Accuracy
            test(projectedData_train, projectedData_test, labelVector_train, labelVector_test, k_values, ExtractId)
            if ExtractId:
                return dataMatrix_test


def plot(PlotName, Xname, x, Yname, y):
    # Plotting the data
    plt.plot(x, y)

    # Adding labels and title
    plt.xlabel(Xname)
    plt.ylabel(Yname)
    plt.title(PlotName)

    # Displaying the plot
    plt.show()


def chooseColor(alpha):
    if alpha == 0.8:
        return "green"
    if alpha == 0.9:
        return "red"
    if alpha == 0.85:
        return "blue"
    if alpha == 0.95:
        return "orange"
    else:
        return "gray"


def plotAlphaVsAccuracy():
    # alpha vs accuracy
    runPCA50_50([1], [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1], False)
    plot('ALPHA vs Accuracy', "Alpha", [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1], "Accuracy", accuracies)
    accuracies.clear()


def plotAccuracyVsK():
    # K vs accuracy
    for alpha in alpha_values:
        runPCA50_50(k_values, [alpha], False)
        plt.plot(k_values, accuracies, color=chooseColor(alpha), label='alpha = {}'.format(alpha))
        accuracies.clear()
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs K for different alpha')
    plt.legend()
    plt.show()
    accuracies.clear()


# Faces vs Non Faces
def plotAccuracyVsNonFaces():
    for alpha in alpha_values:
        runPCA_FNF([1], [alpha], num_nonFaces, False)
        plt.plot(num_nonFaces, accuracies, color=chooseColor(alpha), label='alpha = {}'.format(alpha))
        accuracies.clear()
    plt.xlabel('Number of Non Face Images in Training Set')
    plt.ylabel('Accuracy')
    plt.title(' Training Set : 200 Faces + X NonFace Images \nTest Set : 200 Faces + 200 NonFace Images ')
    plt.legend()
    plt.show()
    accuracies.clear()


def plot50_50Vs70_30_alpha():
    runPCA50_50([1], [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1], False)
    plt.plot([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1], accuracies, color='red', label='50% 50% split')
    accuracies.clear()
    runPCA70_30([1], [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
    plt.plot([0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1], accuracies, color='blue', label='70% 30% split')
    accuracies.clear()
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.title(' 50% 50% split Accuracy vs 70% 30% split Accuracy\n For different alpha and K= 1 ')
    plt.legend()
    plt.show()
    accuracies.clear()


def plot50_50Vs70_30_k():
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for alpha in alpha_values:
        runPCA50_50(k_values, [alpha], False)
        ax1.plot(k_values, accuracies, color=chooseColor(alpha), label='alpha = {}'.format(alpha))
        accuracies.clear()
    for alpha in alpha_values:
        runPCA70_30(k_values, [alpha])
        ax2.plot(k_values, accuracies, color=chooseColor(alpha), label='alpha = {}'.format(alpha))
        accuracies.clear()
    ax1.set_xlabel('K')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('50% 50% Accuracy vs K')
    ax2.set_xlabel('K')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('70% 30% Accuracy vs K')
    plt.tight_layout()
    plt.legend()
    plt.show()
    accuracies.clear()


def plot50_50Vs70_30_k_alpha():
    plot50_50Vs70_30_alpha()
    plot50_50Vs70_30_k()


def PCAVsRPCA_accuracy():
    runPCA50_50([1], [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1], True)
    plt.plot(num_comp, accuracies[::2], color='red', label='PCA')
    plt.plot(num_comp, accuracies[1::2], color='blue', label='Randomized PCA')
    accuracies.clear()
    plt.xlabel('Number of components')
    plt.ylabel('Accuracy')
    plt.title(' PCA vs Randomized PCA based on Accuracy vs Number of components ')
    plt.legend()
    plt.show()
    accuracies.clear()
    times.clear()
    num_comp.clear()


def PCAVsRPCA_time():
    runPCA50_50([1], [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1], True)
    print(len(num_comp))
    print(len(times))
    plt.plot(num_comp, times[::2], color='red', label='PCA')
    plt.plot(num_comp, times[1::2], color='blue', label='Randomized PCA')
    accuracies.clear()
    plt.xlabel('Number of components')
    plt.ylabel('Time')
    plt.title(' PCA vs Randomized PCA based on Time vs Number of components ')
    plt.legend()
    plt.show()
    num_comp.clear()
    times.clear()


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
    dataMatrix = runPCA_FNF([1], [0.9], [300], True)
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
        axes[axeId].set_title(f"Non Face Successfully Detected", fontweight='bold', fontsize=20)
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


plotAlphaVsAccuracy()
plotAccuracyVsK()
plotAccuracyVsNonFaces()
plotSuccessFailureCases()
plot50_50Vs70_30_k_alpha()
PCAVsRPCA_accuracy()
PCAVsRPCA_time()

