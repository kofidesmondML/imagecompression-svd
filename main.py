from PIL import Image
import matplotlib.pyplot as plt 
import os 

image_path = 'image_results'
os.makedirs(image_path, exist_ok=True)

image = Image.open("alone.jpg")
gray_image = image.convert('L')

gray_image_path = os.path.join(image_path, 'gray_alone.jpg')
gray_image.save(gray_image_path)
color_image_path=os.path.join(image_path, 'color_alone.jpg')
image.save(color_image_path)
print(f"Grayscale image saved at: {gray_image_path}")
print(f"Colored Image saved at: {color_image_path}  ")