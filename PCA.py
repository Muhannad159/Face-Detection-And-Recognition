import numpy as np
import cv2
import os


def load_images(root_folder, img_width=64, img_height=64):
    image_paths = []
    image_list = []
    images_not_flattened = []
    labels = []

    # Get all subdirectories in the root folder
    for folder in os.listdir(root_folder):
        if os.path.isdir(os.path.join(root_folder, folder)):
            subdir_path = os.path.join(root_folder, folder)
            # Collect all images in the subdirectory
            image_files = []
            for file in os.listdir(subdir_path):
                if file.endswith(".pgm") or file.endswith(".jpg") or file.endswith(".png"):
                    image_files.append(os.path.join(subdir_path, file))
                    labels.append(folder)
            image_paths.extend(image_files)

    # Read and resize images
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # print("image shape: ", image.shape)
        image = cv2.resize(image, (img_width, img_height))
        images_not_flattened.append(image)
        image_list.append(image.flatten())  # Flatten the image to a 1D array

    return np.array(image_list), images_not_flattened, image_paths, labels


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
    # covariance_matrix = 1/(len(image_list)-1) * np.dot(mean_subtracted_images.T, mean_subtracted_images)
    covariance_matrix = np.cov(mean_subtracted_images, rowvar=False)
    return covariance_matrix, mean

def get_eigenvalues_and_eigenvectors(covariance_matrix):
    # Use numpy.linalg.eig to compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print("done computing eigenvectors")
    # print("shape of eigenvectors: ", eigenvectors.shape)

    # Normalize eigenvectors (each eigenvector is a column in the returned eigenvectors matrix)
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

    return eigenvalues, eigenvectors


def pca_analysis(root_folder):
    images, images_not_flattened, image_paths, labels = load_images(root_folder)  # Load and preprocess images
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
    # detect_faces("Dataset/Testing/s3/2.pgm",principal_components,projected_data)
    return principal_components, projected_data, images_not_flattened, image_paths, labels


def detect_faces(image_path, recognition_threshold, principal_components, projected_data, labels, img_width=64, img_height=64):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # print("image shape: ", image.shape)
    image = cv2.resize(image, (img_width, img_height))
    image = image.flatten()
    projected_image = np.dot(image, principal_components)

    # distance = projected_data -
    minimum_distance = float('inf')
    face_id = -1
    # print("labels: ", labels)
    # print("projected_data shape: ", projected_data.shape)
    # print("labels shape: ", len(labels))
    for i in range(projected_data.shape[0]):
        eclidian_distance = np.linalg.norm(projected_data[i] - projected_image)
        if eclidian_distance < recognition_threshold and eclidian_distance < minimum_distance:
            minimum_distance = eclidian_distance
            face_id = i
    # print("minimum distance: ", minimum_distance)
    return face_id


def main():
    print("Loading images...")
    pca_analysis("Dataset/Training")


if __name__ == "__main__":
    main()
