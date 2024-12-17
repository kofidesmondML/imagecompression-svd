import numpy as np 
import os
import cv2
import time 
from PIL import Image 


def mse_frobenius(A, B):
    try:
        # Ensure the matrices have the same shape
        if A.shape != B.shape:
            raise ValueError("Matrices must have the same dimensions.")
        # Calculate the difference
        difference = A - B
        # Calculate MSE using the Frobenius norm
        mse = (np.linalg.norm(difference, 'fro') ** 2) / (A.shape[0] * A.shape[1])
        return mse
    except ValueError as ve:
        print("ValueError:", ve)
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None

def calculate_compression_ratio(original_image_path, compressed_image_path=None):
    try:
        if not os.path.exists(original_image_path):
            raise FileNotFoundError("The specified original image file does not exist.")
        original_size = os.path.getsize(original_image_path)
        if compressed_image_path:
            if not os.path.exists(compressed_image_path):
                raise FileNotFoundError("The specified compressed image file does not exist.")
            compressed_size = os.path.getsize(compressed_image_path)
            compression_ratio = compressed_size / original_size
            return compression_ratio
        else:
            print("Compressed image path not provided. Cannot calculate the compression ratio.")
            return None

    except FileNotFoundError as fnf_error:
        print("FileNotFoundError:", fnf_error)
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None
    
def measure_computational_time(func, *args, **kwargs):
    start_time = time.time()  # Record the start time
    result = func(*args, **kwargs)  # Call the function and capture its result
    end_time = time.time()  # Record the end time
    computational_time = end_time - start_time  # Calculate elapsed time
    return computational_time, result

def mse_frobenius_colored(image1, image2):
    mse = 0
    channels1 = cv2.split(image1)
    channels2 = cv2.split(image2)
    for c1, c2 in zip(channels1, channels2):
        mse += np.mean((c1 - c2) ** 2)
    return mse / 3
def save_compressed_image(compressed_image, k, func_name):
    folder = "image_results"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f'{func_name}_compressed_image_k={k}.jpg')
    compressed_image_scaled = np.clip(compressed_image * 255, 0, 255)
    compressed_image_rounded = np.round(compressed_image_scaled).astype(np.uint8)
    img = Image.fromarray(compressed_image_rounded)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(filename, format='JPEG')
    return filename



