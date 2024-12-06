import cv2
import numpy as np
import matplotlib.pyplot as plt

def retinex(image, sigma):
    image = np.float32(image) / 255.0
    log_image = np.log1p(image)
    
    # Apply Gaussian filter to the logarithm of the image
    blurred_log_image = cv2.GaussianBlur(log_image, (sigma, sigma), 0)
    
    # Subtract the blurred image from the original image in the log domain
    enhanced_log_image = log_image - blurred_log_image
    
    # Convert back to spatial domain and apply exponential function
    enhanced_image = np.expm1(enhanced_log_image)
    
    return np.clip(enhanced_image * 255, 0, 255).astype(np.uint8)

def lab_retinex(image_path, sigma=55):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    
    # Convert image from BGR to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the Lab image into L, A, B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Apply Retinex on the L channel
    enhanced_l_channel = retinex(l_channel, sigma)
    
    # Merge the enhanced L channel with the original A and B channels
    enhanced_lab_image = cv2.merge((enhanced_l_channel, a_channel, b_channel))
    
    # Convert the enhanced Lab image back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

# Example usage
if __name__ == "__main__":
    input_image_path = './input_debug/0001.jpg'
    output_image_path = './output/output_image.jpg'
    
    enhanced_image = lab_retinex(input_image_path)
    
    # Save the enhanced image
    cv2.imwrite(output_image_path, enhanced_image)
    
    # Display images using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Enhanced Image')
    axes[1].axis('off')
    
    plt.show()