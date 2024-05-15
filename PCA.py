import numpy as np
import cv2
import os


def load_images(root_folder, img_width=64, img_height=64):
    """
    Load images from a given root folder and resize them to the specified dimensions.

    Args:
        root_folder (str): The root directory where the images are stored.
        img_width (int, optional): The width to which images should be resized. Defaults to 64.
        img_height (int, optional): The height to which images should be resized. Defaults to 64.

    Returns:
        tuple: A tuple containing the list of flattened images, the list of non-flattened images,
               the list of image paths, and the list of labels corresponding to the images.
    """
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
    print("images matrix", np.array(image_list).shape)

    return np.array(image_list), images_not_flattened, image_paths, labels


def calculate_covariance_matrix(image_list):
    """
    Calculate the covariance matrix of a list of images.

    Args:
        image_list (list): The list of images.

    Returns:
        tuple: A tuple containing the covariance matrix and the mean image.
    """
    # Calculate the mean of the images
    mean = np.mean(image_list, axis=0)
    # Subtract the mean from the images
    mean_subtracted_images = image_list - mean
    # Calculate the covariance matrix
    # covariance_matrix = 1/(len(image_list)-1) * np.dot(mean_subtracted_images.T, mean_subtracted_images)
    covariance_matrix = np.cov(mean_subtracted_images, rowvar=False)
    print("covariance matrix shape: ", covariance_matrix.shape)
    return covariance_matrix, mean

def get_eigenvalues_and_eigenvectors(covariance_matrix):
    """
    Compute the eigenvalues and eigenvectors of a given covariance matrix.

    Args:
        covariance_matrix (numpy.ndarray): The covariance matrix.

    Returns:
        tuple: A tuple containing the eigenvalues and eigenvectors.
    """
    # Use numpy.linalg.eig to compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    print("done computing eigenvectors")
    print("shape of eigenvectors: ", eigenvectors.shape)

    # Normalize eigenvectors (each eigenvector is a column in the returned eigenvectors matrix)
    eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)

    return eigenvalues, eigenvectors


def pca_analysis(root_folder):
    """
    Perform Principal Component Analysis (PCA) on images from a given root folder.

    Args:
        root_folder (str): The root directory where the images are stored.

    Returns:
        tuple: A tuple containing the principal components, the projected data, the non-flattened images,
               the image paths, the labels, and the mean image.
    """
    images, images_not_flattened, image_paths, labels = load_images(root_folder)  # Load and preprocess images
    covariance_matrix, mean_image = calculate_covariance_matrix(images)  # Calculate covariance and mean
    eigenvalues, eigenvectors = get_eigenvalues_and_eigenvectors(covariance_matrix)  # Get eigenvalues and eigenvectors
    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    print("eigenvectors shape 2: ", eigenvectors.shape)
    cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    desired_variance = 0.90  # 90% variance explained
    num_components = np.argmax(cumulative_variance >= desired_variance) + 1
    principal_components = eigenvectors[:, :num_components]
    images = images - mean_image
    projected_data = np.dot(images, principal_components)
    print("principal_components shape: ", principal_components.shape)
    print("projected_data shape: ", projected_data.shape)
    # detect_faces("Dataset/Testing/s3/2.pgm",principal_components,projected_data)
    return principal_components, projected_data, images_not_flattened, image_paths, labels, mean_image


def detect_faces(image_path, recognition_threshold, principal_components, projected_data, labels, mean_image, img_width=64, img_height=64):
    """
    Detect faces in an image using PCA.

    Args:
        image_path (str): The path to the image.
        recognition_threshold (float): The threshold for face recognition.
        principal_components (numpy.ndarray): The principal components obtained from PCA.
        projected_data (numpy.ndarray): The projected data obtained from PCA.
        labels (list): The labels corresponding to the images.
        mean_image (numpy.ndarray): The mean image.
        img_width (int, optional): The width to which the image should be resized. Defaults to 64.
        img_height (int, optional): The height to which the image should be resized. Defaults to 64.

    Returns:
        int: The ID of the detected face. Returns -1 if no face is detected.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # print("image shape: ", image.shape)
    image = cv2.resize(image, (img_width, img_height))
    image = image.flatten()
    image = image - mean_image
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
    print("minimum distance: ", minimum_distance)
    return face_id


def main():
    """
    The main function to start the PCA analysis.
    """
    print("Loading images...")
    pca_analysis("Dataset/Training")


if __name__ == "__main__":
    main()
