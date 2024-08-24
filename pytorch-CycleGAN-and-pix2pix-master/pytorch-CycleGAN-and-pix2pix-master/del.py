import os
from PIL import Image

# 设置图片文件夹路径
folder_path = './datasets/large_real/testB'

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # 检查文件是否为图片
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            
            # 检查宽度或高度是否为3215
            if width == 1957 or height == 1957:
                print(f"Deleting image: {filename} with size {width}x{height}")
                os.remove(file_path)
    except IOError:
        print(f"File {filename} is not a valid image file.")
