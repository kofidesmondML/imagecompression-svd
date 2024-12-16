import numpy as np 
import cv2 


def trunc_svd(image_array,k):
    try:
        U, S, Vt = np.linalg.svd(image_array, full_matrices=False)
        S_k = np.diag(S[:k])
        #print(S_k)
        compressed_image = np.dot(U[:, :k], np.dot(S_k, Vt[:k, :]))
        return compressed_image
    except Exception as e:
        print("Error in performing SVD:", e)
        return None

def compressed_svd(X, k, p=10):
    l = k + p
    m, n = X.shape
    # Step 2: Generate a random test matrix
    Phi = np.random.randn(l, m)
    # Step 3: Sketch the input matrix
    Y = Phi @ X
    # Step 4: Form smaller l × l matrix
    B = Y @ Y.T
    # Step 5: Ensure symmetry
    B = 0.5 * (B + B.T)
    # Step 6: Truncated eigendecomposition
    D, T = np.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors in descending order, then truncate to k
    idx = np.argsort(D)[::-1]
    D = D[idx][:k]
    T = T[:, idx][:, :k]
    # Step 7: Rescale eigenvalues
    S_tilde = np.sqrt(np.diag(D))
    # Step 8: Approximate right singular values
    V_tilde = Y.T @ T @ np.linalg.inv(S_tilde)
    # Step 9: Approximate unscaled left singular values
    U_tilde = X @ V_tilde
    # Step 10: SVD on U_tilde to update left singular vectors and values
    U, Sigma, Q_T = np.linalg.svd(U_tilde, full_matrices=False)
    S = np.diag(Sigma[:k])
    # Step 11: Update right singular vectors
    V = V_tilde @ Q_T.T
    # Calculate the product U * S * V.T
    product = U[:, :k] @ S @ V[:, :k].T
    return product


def randomized_svd(A, k, p=10, q=0):
    # Step 1: Set l for slight oversampling
    l = k + p
    # Step 2: Generate a random test matrix
    m, n = A.shape
    Phi = np.random.randn(n, l)
    # Step 3: Compute Y = A * Phi
    Y = A @ Phi
    # Power iteration to improve approximation
    for i in range(q):
        print('Power iteration', i)
        Y = A @ (A.T @ Y)
    # Step 4: Compute QR decomposition of Y
    Q, R = np.linalg.qr(Y, mode='reduced')
    # Step 5: Form the smaller matrix B
    B = Q.T @ A
    # Step 6: Perform SVD on B
    U_tilde, Sigma, Vt = np.linalg.svd(B, full_matrices=False)
    # Step 7: Adjust U with Q to get the final U
    U = Q @ U_tilde
    # Return U, S as a diagonal matrix, V, and their product
    S = np.diag(Sigma[:k])
    V = Vt[:k, :].T
    product = U[:, :k] @ S @ V.T

    return  product

def trunc_svd_colored(image_array, k):
    try:
        channels = cv2.split(image_array)
        print("Channel splitted")
        compressed_channels = []
        for idx,channel in enumerate(channels):
            print(f'processing channel',idx+1)
            U, S, Vt = np.linalg.svd(channel, full_matrices=False)
            S_k = np.diag(S[:k])
            compressed_channel = np.dot(U[:, :k], np.dot(S_k, Vt[:k, :]))
            compressed_channels.append(compressed_channel)
        compressed_image = cv2.merge(compressed_channels)
        return compressed_image
    except Exception as e:
        print("Error in performing SVD:", e)
        return None

def compressed_svd_colored(image_array, k, p=10):
    compressed_channels = []
    channels = cv2.split(image_array)
    
    for X in channels:
        l = k + p
        m, n = X.shape
        
        Phi = np.random.randn(l, m)
        Y = Phi @ X
        B = Y @ Y.T
        B = 0.5 * (B + B.T)
        
        D, T = np.linalg.eigh(B)
        idx = np.argsort(D)[::-1]
        D = D[idx][:k]
        T = T[:, idx][:, :k]
        
        S_tilde = np.sqrt(np.diag(D))
        V_tilde = Y.T @ T @ np.linalg.inv(S_tilde)
        U_tilde = X @ V_tilde
        
        U, Sigma, Q_T = np.linalg.svd(U_tilde, full_matrices=False)
        S = np.diag(Sigma[:k])
        V = V_tilde @ Q_T.T
        
        compressed_channel = U[:, :k] @ S @ V[:, :k].T
        compressed_channels.append(compressed_channel)
    
    compressed_image = cv2.merge(compressed_channels)
    return compressed_image

def randomized_svd_colored(image_array, k, p=10, q=0):
    compressed_channels = []
    channels = cv2.split(image_array)
    
    for A in channels:
        l = k + p
        m, n = A.shape
        Phi = np.random.randn(n, l)
        Y = A @ Phi
        
        for i in range(q):
            Y = A @ (A.T @ Y)
        
        Q, R = np.linalg.qr(Y, mode='reduced')
        B = Q.T @ A
        U_tilde, Sigma, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ U_tilde
        S = np.diag(Sigma[:k])
        V = Vt[:k, :].T
        compressed_channel = U[:, :k] @ S @ V.T
        
        compressed_channels.append(compressed_channel)
    compressed_image = cv2.merge(compressed_channels)
    return compressed_image

    