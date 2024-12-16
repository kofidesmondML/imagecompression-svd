from PIL import Image
import matplotlib.pyplot as plt 
import os 
import numpy as np
import pandas as pd
import svd 
import evaluation

image_path = 'image_results'
evaluation_path='evaluation_results'
os.makedirs(image_path, exist_ok=True)
image = Image.open("alone.jpg")
gray_image = image.convert('L')
image_matrix = np.array(gray_image)/255
print(image_matrix.shape)

gray_image_path = os.path.join(image_path, 'gray_alone.jpg')
gray_image.save(gray_image_path)
color_image_path=os.path.join(image_path, 'color_alone.jpg')
image.save(color_image_path)
print(f"Grayscale image saved at: {gray_image_path}")
print(f"Colored Image saved at: {color_image_path}")

singular_values=[1,5,10,50,100]#150,200,250,300,350,400,450,500,750,1000,1500,1750,2000,2250,2400,2431]
print('This is the start of the truncated svd')
tsvd_results=[]
for k in singular_values:
    print(f'Running the truncated svd for {k} singular value')
    comp_time, comp_image=evaluation.measure_computational_time(svd.trunc_svd,image_matrix,k)
    compressed_image_path=os.path.join(image_path, f'tsvd_{k}.jpg')
    plt.imsave(compressed_image_path,comp_image, cmap='gray')
    mse=evaluation.mse_frobenius(image_matrix, comp_image)
   
    compressed_ratio=evaluation.calculate_compression_ratio(gray_image_path,compressed_image_path)

    tsvd_results.append({
            'k': k,
            'time_taken': comp_time,
            'mse': mse,
            'compression_ratio': compressed_ratio
        })
    tsvd_df=pd.DataFrame(tsvd_results)
print(tsvd_df.head())

print(' This is the start of the compressed svd')
csvd_results=[]
for k in singular_values:
    print(f' Running the compressed svd for {k} singular value')
    comp_time, comp_image=evaluation.measure_computational_time(svd.compressed_svd,image_matrix,k)
    compressed_image_path=os.path.join(image_path, f'csvd_{k}.jpg')
    plt.imsave(compressed_image_path,comp_image, cmap='gray')
    mse=evaluation.mse_frobenius(image_matrix, comp_image)
    compressed_ratio=evaluation.calculate_compression_ratio(gray_image_path,compressed_image_path)
    csvd_results.append({
            'k': k,
            'time_taken': comp_time,
            'mse': mse,
            'compression_ratio': compressed_ratio
        })
    csvd_df=pd.DataFrame(csvd_results)
print(csvd_df.head())

print(' This is the start of the randomized svd without power iterations')
rsvd_results=[]
for k in singular_values:
    print(f' Running the randomized svd without power iterations for {k} singular value')
    comp_time, comp_image=evaluation.measure_computational_time(svd.randomized_svd,image_matrix,k)
    compressed_image_path=os.path.join(image_path, f'rsvd_0_{k}.jpg')
    plt.imsave(compressed_image_path,comp_image, cmap='gray')
    mse=evaluation.mse_frobenius(image_matrix, comp_image)
    compressed_ratio=evaluation.calculate_compression_ratio(gray_image_path,compressed_image_path)
    rsvd_results.append({
            'k': k,
            'time_taken': comp_time,
            'mse': mse,
            'compression_ratio': compressed_ratio
        })
    rsvd_df=pd.DataFrame(rsvd_results)
print(rsvd_df.head())

print('This is the start of the randomized svd with power iterations q=2')
rsvd_2_results=[]
for k in singular_values: 
    print(f'Running the randomized svd with power iterations q=2')
    comp_time, comp_image=evaluation.measure_computational_time(svd.randomized_svd,image_matrix,k,q=2)
    compressed_image_path=os.path.join(image_path, f'rsvd_q=2_{k}.jpg')
    plt.imsave(compressed_image_path,comp_image, cmap='gray')
    mse=evaluation.mse_frobenius(image_matrix, comp_image)
    compressed_ratio=evaluation.calculate_compression_ratio(gray_image_path,compressed_image_path)
    rsvd_2_results.append({
            'k': k,
            'time_taken': comp_time,
            'mse': mse,
            'compression_ratio': compressed_ratio
        })
    rsvd_2_df=pd.DataFrame(rsvd_2_results)
print(rsvd_2_df.head())

print('This is the start of the randomized svd with power iterations q=3')
rsvd_3_results=[]
for k in singular_values: 
    print(f'Running the randomized svd with power iterations q=3')
    comp_time, comp_image=evaluation.measure_computational_time(svd.randomized_svd,image_matrix,k,q=3)
    compressed_image_path=os.path.join(image_path, f'rsvd_q=3_{k}.jpg')
    plt.imsave(compressed_image_path,comp_image, cmap='gray')
    mse=evaluation.mse_frobenius(image_matrix, comp_image)
    compressed_ratio=evaluation.calculate_compression_ratio(gray_image_path,compressed_image_path)
    rsvd_3_results.append({
            'k': k,
            'time_taken': comp_time,
            'mse': mse,
            'compression_ratio': compressed_ratio
        })
    rsvd_3_df=pd.DataFrame(rsvd_3_results)
print(rsvd_3_df.head())


os.makedirs(evaluation_path, exist_ok=True)

dfs = [tsvd_df, csvd_df, rsvd_df, rsvd_2_df, rsvd_3_df]
legend_labels = ['trunc_svd', 'csvd', 'rsvd q=0', 'rsvd q=2', 'rsvd q=3']
columns = ['time_taken', 'mse', 'compression_ratio']
titles = ['Computational Time', 'MSE Error', 'Compression Ratio']

for i, col in enumerate(columns):
    plt.figure(figsize=(10, 5))
    for idx, df in enumerate(dfs):
        plt.plot(df['k'], df[col], label=legend_labels[idx])
    
    plt.title(titles[i])
    plt.xlabel('k')
    plt.ylabel(col)
    plt.legend(loc='best')
    plt.tight_layout()
    
    image_path = os.path.join(evaluation_path, f"{col}_plot.png")
    plt.savefig(image_path)
    plt.close()













 




