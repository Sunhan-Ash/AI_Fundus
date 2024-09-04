from PIL import Image
def drop_pixels(image, downscale_factor):
    original_image = image.copy()
    width, height = original_image.size
    new_width = width // downscale_factor
    new_height = height // downscale_factor
    new_image = Image.new('RGB', (new_width, new_height))
    
    pixels = original_image.load()
    new_pixels = new_image.load()
    # 通过创建一个新的图像，然后把原图的像素按照比例复制进去，再用resize函数恢复
    for i in range(new_width):
        for j in range(new_height):
            new_pixels[i, j] = pixels[i * downscale_factor, j * downscale_factor]
    resized_image = new_image.resize((width, height), Image.BILINEAR)
    
    return resized_image