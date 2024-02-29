from PIL import Image
import os
import numpy as np
import re

def read_and_flatten(folder_path):
    trainingSample = []
    testSample = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        img = Image.open(file_path).convert('L')  # Convert to grayscale using 'L' mode
        flattened_img = np.array(img).flatten()
        file_number = re.search(r'\d+', file_name)
        file_number = int(file_number.group())
        if file_number % 2 != 0:  # odd training
            trainingSample.append(flattened_img)
        else:
            testSample.append(flattened_img)

    return np.array(trainingSample), np.array(testSample)

main_folder_path = r"./face recongnition"
num_classes = 40

trainingSet = []
testSet = []

for i in range(1, num_classes + 1):
    class_folder_path = os.path.join(main_folder_path, f"s{i}")
    training, test = read_and_flatten(class_folder_path)
    trainingSet.append(training)
    testSet.extend(test)

trainingSet = np.array(trainingSet)
testSet = np.array(testSet)

mean_training_per_class = np.mean(trainingSet, axis=1)
overall_mean = np.mean((mean_training_per_class), axis=0)


# calculate Sb
# Sb = []
# for i in range(0, 40):
#     tmp = (mean_training_per_class[i] - overall_mean)
#     Sb += np.outer(tmp, tmp)
#
# print(Sb)
# print(overall_mean.shape)
# print(overall_mean)

# print("-"*30,mean_training_per_class)

# Print the shape of mean_training_per_class
# print(f"Shape of mean_training_per_class: {mean_training_per_class.shape}")

# # Print the mean for each class
# for i, mean_class in enumerate(mean_training_per_class, start=1):
#     print(f"Mean for Class {i}:\n{mean_class}")
#     print("-" * 40)
