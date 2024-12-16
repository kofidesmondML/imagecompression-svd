import numpy as np 
import os
import time  


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