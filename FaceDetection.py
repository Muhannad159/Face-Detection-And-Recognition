import cv2

def face_detection(img):
    """
    Detect faces in an image using Haar cascades.

    Args:
        img (numpy.ndarray): The input image.

    Returns:
        list: A list of rectangles where each rectangle represents a detected face. Each rectangle is represented as a tuple of 4 integers (x, y, w, h) where (x, y) is the top-left corner of the rectangle and (w, h) are its width and height respectively.
    """
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('library/haarcascade_frontalface_default.xml')
    faces = face_haar_cascade.detectMultiScale(grey_img, scaleFactor=1.1, minNeighbors=5)
    return faces

def draw_rectangle(img, faces):
    """
    Draw rectangles around detected faces in an image.

    Args:
        img (numpy.ndarray): The input image.
        faces (list): A list of rectangles where each rectangle represents a detected face. Each rectangle is represented as a tuple of 4 integers (x, y, w, h) where (x, y) is the top-left corner of the rectangle and (w, h) are its width and height respectively.

    Returns:
        numpy.ndarray: The image with rectangles drawn around detected faces.
    """
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    return img