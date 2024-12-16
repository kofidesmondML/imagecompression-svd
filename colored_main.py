from PIL import Image
import matplotlib.pyplot as plt 
import os 
import numpy as np
import pandas as pd
import svd 
import evaluation
import cv2

image_path = 'image_results'
evaluation_path='evaluation_results'
image_bgr = cv2.imread('alone.jpg')
image=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
image=image/255
plt.imshow(image)
colored_image_path=os.path.join(image_path, 'color_alone.jpg')

singular_values=[1,5,10,50,100]#150,200,250,300,350,400,450,500,750,1000,1500,1750,2000,2250,2400,2431]
print('This is the start of the truncated svd')
tsvd_results=[]
for k in singular_values:
    print(f'Running the truncated svd for {k} singular value')
    comp_time, comp_image=evaluation.measure_computational_time(svd.trunc_svd_colored,image,k)
    compressed_image_path=os.path.join(image_path, f'tsvd_{k}_colored.jpg')
    plt.imsave(compressed_image_path,comp_image, cmap='gray')
    mse=evaluation.mse_frobenius(image, comp_image)

    compressed_ratio=evaluation.calculate_compression_ratio(colored_image_path,compressed_image_path)

    tsvd_results.append({
            'k': k,
            'time_taken': comp_time,
            'mse': mse,
            'compression_ratio': compressed_ratio
        })
    tsvd_df=pd.DataFrame(tsvd_results)
print(tsvd_df.head())