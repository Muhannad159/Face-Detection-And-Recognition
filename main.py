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
from scipy.interpolate import RectBivariateSpline
from skimage._shared.utils import _supported_float_type
from skimage.util import img_as_float
from skimage.filters import sobel

import matplotlib.pyplot as plt
import scipy as sp
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
from os import path
from shape_detect import shapedetection
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
from active_contour import ActiveContour
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
import feature_extraction
from sift import siftapply
import time
from segmentation import RGB_to_LUV, kmeans_segmentation,mean_shift
import agglomerative
# from skimage.filters import sobel
from scipy.interpolate import RectBivariateSpline
from threshold import optimal_thresholding, spectral_thresholding,otsu_threshold,local_thresholding

FORM_CLASS, _ = loadUiType(
    path.join(path.dirname(__file__), "main.ui")
)  # connects the Ui file with the Python file


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        self.handle_buttons()
        self.setWindowTitle("Face Detection")

    def handle_buttons(self):
        """
        Connects buttons to their respective functions and initializes the application.
        """
        pass


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
