from os import path
import sys
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtGui import QPixmap, QImage, QPainter
import cv2
from PyQt5.QtCore import Qt
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
import numpy as np
import math

import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
from os import path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.uic import loadUiType
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
import numpy as np
import math
from scipy.interpolate import RectBivariateSpline
from skimage._shared.utils import _supported_float_type
from skimage.util import img_as_float
from skimage.filters import sobel
import cv2
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw
from FaceDetection import face_detection , draw_rectangle

import time

# from skimage.filters import sobel
from scipy.interpolate import RectBivariateSpline

FORM_CLASS, _ = loadUiType(
    path.join(path.dirname(__file__), "main.ui")
)  # connects the Ui file with the Python file


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        self.handle_buttons()
        self.setWindowTitle("Face Detection")
        self.loaded_image = None
        self.image_with_faces = None

    def handle_buttons(self):
        """
        Connects buttons to their respective functions and initializes the application.
        """
        self.label.mouseDoubleClickEvent = lambda event: self.handle_mouse(event, label=self.label)
        self.pushButton.clicked.connect(self.face_detection)
        pass

    def handle_mouse(self, event, label):
        """
        Method to handle the mouse click event on the image.

        Args:
            event: The mouse click event.
        """
        if event.button() == Qt.LeftButton:
            self.load_image(label)
        # elif event.button() == Qt.RightButton:
        #     self.detect_face()

    def load_image(self, label):
        """
        Method to load the image and display it on the GUI.
        """
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.pgm)")
        self.loaded_image = cv2.imread(self.file_path)
        self.loaded_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.display_image(self.loaded_image, label)

    def display_image(self, image, label):
        """
        Method to display the image on the GUI.

        Args:
            image: The image to be displayed.
        """
        height, width, channel = image.shape
        # Resize label to fit the image
        label.resize(width, height)
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        label.setScaledContents(True)
        label.show()

    def face_detection(self):
        """
        Method to detect the face in the image.
        """
        faces = face_detection(self.loaded_image)
        self.image_with_faces = draw_rectangle(self.loaded_image, faces)
        self.display_image(self.image_with_faces, self.label_2)


def main():
    """
    Method to start the application.

    Returns:
        None
    """
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()  # infinte Loop


if __name__ == "__main__":
    main()
