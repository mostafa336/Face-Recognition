from PIL import Image
import os
import numpy as np
import re
# Folder path containing subfolders with .pgm images
main_folder_path = "./face recongnition"

# Initialize an array to store flattened images
flattened_images = []
labels = []
label_cnt = 0
# Loop through subfolders
for subfolder_name in os.listdir(main_folder_path):
    subfolder_path = os.path.join(main_folder_path, subfolder_name)

    # Ensure it's a directory
    if os.path.isdir(subfolder_path):
        # Loop through images in each subfolder (assuming there are 10 images in each subfolder)
        match = re.search(r'\d+', subfolder_name)
        label_cnt = int(match.group()) if match else None
        for i in range(1, 11):
            image_name = f"{i}.pgm"
            image_path = os.path.join(subfolder_path, image_name)

            # Read the image using PIL
            try:
                img = Image.open(image_path)
                # Flatten the image and append it to the list
                flattened_img = np.array(img).flatten()
                flattened_images.append(flattened_img)
                labels.append(label_cnt)
                print(f"subfolder : {subfolder_path} : {i}")
                print(f"labels : {label_cnt}")
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")

flattened_images = np.array(flattened_images)
labels = np.array(labels)

print(f"Shape of flattened_images: {flattened_images.shape}")
print(f"Shape of labels: {labels.shape}")


