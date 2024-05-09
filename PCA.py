import numpy as np
import cv2
import os

def load_images(root_folder, img_width=64, img_height=64):
    image_paths = []
    image_list = []
    
    # Get all subdirectories in the root folder
    for folder in os.listdir(root_folder):
        if os.path.isdir(os.path.join(root_folder, folder)):
            subdir_path = os.path.join(root_folder, folder)
            # Collect all images in the subdirectory
            image_files = [os.path.join(subdir_path, file) for file in os.listdir(subdir_path) if file.endswith(".pgm") or file.endswith(".jpg")]
            image_paths.extend(image_files)

    # Read and resize images
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # print("image shape: ", image.shape)
        image = cv2.resize(image, (img_width, img_height))
        image_list.append(image.flatten())  # Flatten the image to a 1D array

    return np.array(image_list)

def calculate_covariance_matrix(images):
    # Compute the mean image
    mean_image = np.mean(images, axis=0)
    # Subtract the mean from each image
    centered_images = images - mean_image
    # Compute the covariance matrix
    covariance_matrix = np.cov(centered_images, rowvar=False)
    print("done computing covariance matrix")
    print("shape of covariance matrix: ", covariance_matrix.shape)
    return covariance_matrix, mean_image

def get_eigenvalues_and_eigenvectors(covariance_matrix):
    # Use numpy.linalg.eig to compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print("done computing eigenvectors")
    # print("shape of eigenvectors: ", eigenvectors.shape)

    # Normalize eigenvectors (each eigenvector is a column in the returned eigenvectors matrix)
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    
    return eigenvalues, eigenvectors

def pca_analysis(root_folder):
    images = load_images(root_folder)  # Load and preprocess images
    covariance_matrix, mean_image = calculate_covariance_matrix(images)  # Calculate covariance and mean
    eigenvalues, eigenvectors = get_eigenvalues_and_eigenvectors(covariance_matrix)  # Get eigenvalues and eigenvectors
    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    desired_variance = 0.90  # 90% variance explained
    num_components = np.argmax(cumulative_variance >= desired_variance) + 1
    principal_components = eigenvectors[:, :num_components]
    projected_data = np.dot(images, principal_components)
    print("principal_components shape: ", principal_components.shape)
    print("projected_data shape: ", projected_data.shape)









def main():
    print("Loading images...")
    pca_analysis("Dataset/Training")


if __name__ == "__main__":
    main()