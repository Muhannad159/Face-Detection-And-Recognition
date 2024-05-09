import os
import cv2
import numpy as np
from sympy import symbols, eye, det, solve

def load_images():    
    root_folder = "Dataset\Training"

    # List all subfolders inside the data folder
    subfolders = [os.path.join(root_folder, folder) for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))]
    # print(subfolders)
    # List to store all image paths
    image_paths = []
    image_list = []
    # Loop through each subfolder
    for folder in subfolders:
        # List all image files inside the subfolder
        image_files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith(".pgm") or file.endswith(".jpg")]
        
        # Add image paths to the list
        image_paths.extend(image_files)

    for images in image_paths:
        loaded_image = cv2.imread(images)
        loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
        
        image_list.append(loaded_image)
    PCA(image_list)

def PCA(image_list):
    print("Performing PCA...")
    oneD_vector =[]
    for image in image_list:
        oneD_vector.append(image.flatten())
    
    covarinace_matrix = calculate_covariance_matrix(oneD_vector)
    eigenvalues, eigenvectors = get_eigenvalues_and_eigenvectors(covarinace_matrix)
    print(eigenvectors.shape)
    print("eigen values", eigenvalues)


def calculate_covariance_matrix(image_list):
    """
    Method to calculate the covariance matrix of the image list.

    Args:
        image_list: The list of images.

    Returns:
        The covariance matrix.
    """
    # Calculate the mean of the images
    mean = np.mean(image_list, axis=0)
    # Subtract the mean from the images
    mean_subtracted_images = image_list - mean
    # Calculate the covariance matrix
    # print(len(image_list))
    covariance_matrix = 1/(len(image_list)-1) * np.dot(mean_subtracted_images.T, mean_subtracted_images)
    return covariance_matrix

def get_eigenvalues_and_eigenvectors(covariance_matrix):
    """
    Method to calculate the eigenvalues and eigenvectors of the covariance matrix.

    Args:
        covariance_matrix: The covariance matrix.

    Returns:
        The eigenvalues and eigenvectors.
    """
    # Calculate the eigenvalues and eigenvectors
    n = covariance_matrix.shape[0]
    λ = symbols('λ')  # Symbol for eigenvalues
    I = eye(n)  # Identity matrix of the same size
    A = covariance_matrix.astype(float)  # Ensure float type for computation
    
    # Calculate the characteristic polynomial
    characteristic_matrix = A - λ * I
    characteristic_poly = det(characteristic_matrix)

    # Solve the characteristic polynomial to get the eigenvalues
    eigenvalues = solve(characteristic_poly, λ)

    # Calculate the eigenvectors
    eigenvectors = []
    
    for eigenvalue in eigenvalues:
        # Create the matrix (A - eigenvalue * I)
        characteristic_matrix = covariance_matrix - eigenvalue * I
        
        # Find the null space to get eigenvectors
        null_space = characteristic_matrix.nullspace()

        if not null_space:
            raise ValueError(f"No eigenvector found for eigenvalue {eigenvalue}.")
        
        # Add all vectors in the null space as eigenvectors
        for vector in null_space:
            eigenvectors.append(np.array(vector).astype(np.float64).flatten())

    return eigenvalues, eigenvectors


def main():
    print("Loading images...")
    load_images()


if __name__ == "__main__":
    main()