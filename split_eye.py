import pandas as pd
import shutil
import os

# Load the dataset
file_path = '/media/xusunhan/ZhiTai/AI_fundus/EyeQ-master/data/Label_EyeQ_train.csv'  # Update this to the correct path of your CSV file
data = pd.read_csv(file_path)

# Filter the dataset where quality is 2
quality_2_data = data[data['quality'] == 2]

# Directory where images are stored
source_dir = '/media/xusunhan/ZhiTai/AI_fundus/EyeQ-master/EyeQ_preprocess/original_img/train/'  # Update this to the correct path of your images folder
# Directory where images with quality 2 will be moved
destination_dir = '/media/xusunhan/ZhiTai/AI_fundus/dataset/EyeQ/train/reject/'  # Update this to the desired path for quality 2 images

# Create the destination directory if it does not exist
os.makedirs(destination_dir, exist_ok=True)

# Move the images to the destination directory
for index, row in quality_2_data.iterrows():
    image_name = row['image']
    source_path = os.path.join(source_dir, image_name)
    destination_path = os.path.join(destination_dir, image_name)
    
    # Check if the source image exists
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
        print(f"Moved: {image_name}")
    else:
        print(f"File not found: {image_name}")

print("All quality 2 images have been moved.")
