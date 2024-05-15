import sys
from os import path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication
from PyQt5.uic import loadUiType
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import PCA
from FaceDetection import face_detection, draw_rectangle

# Load the UI file and the Python file
FORM_CLASS, _ = loadUiType(
    path.join(path.dirname(__file__), "main.ui")
)

class MainApp(QMainWindow, FORM_CLASS):
    """
    MainApp is a class for the main application window.

    Attributes:
        parent: A reference to the parent object.
    """

    def __init__(self, parent=None):
        """
        The constructor for MainApp class.

        Parameters:
            parent (optional): A reference to the parent object.
        """
        super(MainApp, self).__init__(parent)
        self.setupUi(self)
        self.handle_buttons()
        self.setWindowTitle("Face Detection")
        self.loaded_image = None
        self.image_with_faces = None
        self.file_path = None
        (self.training_principal_components, self.training_projected_data, self.training_images_not_flattened,
         self.training_image_paths, self.training_labels, self.mean_image_training) = PCA.pca_analysis(
            "Dataset/Training"
        )
        self.testing_principal_components, self.testing_projected_data, self.testing_images_not_flattened, self.testing_image_paths, self.testing_labels, self.mean_image_testing = PCA.pca_analysis(
            "Dataset/Testing"
        )
        self.predicted_labels = []
        self.perform_pca()

    def handle_buttons(self):
        """
        Connects buttons to their respective functions and initializes the application.
        """
        self.label.mouseDoubleClickEvent = lambda event: self.handle_mouse(event, label=self.label)
        self.pushButton.clicked.connect(self.face_detection)
        self.pushButton_2.clicked.connect(self.recognize_face)
        self.recog_slider.valueChanged.connect(self.recognize_face_slider_change)
        self.recog_slider_2.valueChanged.connect(self.recognize_face_slider_change_2)

    def handle_mouse(self, event, label):
        """
        Method to handle the mouse click event on the image.

        Args:
            event: The mouse click event.
        """
        if event.button() == Qt.LeftButton:
            self.load_image(label)

    def load_image(self, label):
        """
        Method to load the image and display it on the GUI.
        """
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                        "Image Files (*.png *.jpg *.jpeg *.bmp *.pgm)")
        self.loaded_image = cv2.imread(self.file_path)
        self.loaded_image = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2RGB)
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

    def recognize_face(self):
        """
        Method to recognize the face in the image.
        """
        threshold = self.recog_slider.value()
        face_id = PCA.detect_faces(self.file_path, threshold, self.training_principal_components,
                                   self.training_projected_data, self.training_labels, self.mean_image_training)
        if face_id not in [-1, None]:
            image = cv2.imread(self.training_image_paths[face_id])
            self.display_image(image, self.label_2)

        else:
            self.label_2.setText("Face not recognized")

        # self.display_image(image, self.label_2)
        self.perform_pca()

    def perform_pca(self):
        """
        This method performs Principal Component Analysis (PCA) on the images.

        It calculates the true positive rate, false positive rate, accuracy, precision, recall, specificity,
        false positive rate value, and F1 score for the face recognition task. It also plots the ROC curve
        and calculates the AUC score.

        Attributes:
            true_positive (int): The number of true positives.
            false_positive (int): The number of false positives.
            true_negative (int): The number of true negatives.
            false_negative (int): The number of false negatives.
            false_positive_rate (list): The list of false positive rates.
            true_positive_rate (list): The list of true positive rates.
            threshold (int): The threshold value for face recognition.
            epsilon (float): A small constant to prevent division by zero.
            accuracy (float): The accuracy of face recognition.
            precision (float): The precision of face recognition.
            recall (float): The recall of face recognition.
            specificity (float): The specificity of face recognition.
            false_positive_rate_value (float): The false positive rate value.
            f1_score (float): The F1 score of face recognition.
            auc_value (float): The AUC score of the ROC curve.
        """
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        false_positive_rate = []
        true_positive_rate = []
        threshold = self.recog_slider_2.value()
        epsilon = 1e-10
        for i in range(len(self.testing_image_paths)):
            face_id = PCA.detect_faces(self.testing_image_paths[i], threshold, self.training_principal_components,
                                       self.training_projected_data, self.training_labels, self.mean_image_training)
            self.predicted_labels.append(self.testing_labels[i])
            if self.training_labels[face_id] == self.testing_labels[i] and face_id != -1:
                true_positive += 1
            elif self.training_labels[face_id] != self.testing_labels[i] and face_id != -1:
                false_positive += 1
            elif face_id == -1 and self.testing_labels[i] in self.training_labels:
                false_negative += 1
            else:
                true_negative += 1

            false_positive_rate.append(false_positive / (false_positive + true_negative + epsilon))
            true_positive_rate.append(true_positive / (true_positive + false_negative + epsilon))

        print("True Positive : ", true_positive)
        print("False Positive : ", false_positive)
        print("True Negative : ", true_negative)
        print("False Negative : ", false_negative)
        print("True Positive Rate : ", true_positive_rate)
        print("False Positive Rate : ", false_positive_rate)
        print("len(true_positive_rate) : ", len(true_positive_rate))
        print("len(false_positive_rate) : ", len(false_positive_rate))
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        if true_negative + false_positive == 0:
            specificity = 0
            false_positive_rate_value = 0
        else:
            specificity = true_negative / (true_negative + false_positive)
            false_positive_rate_value = false_positive / (false_positive + true_negative)
        f1_score = 2 * (precision * recall) / (precision + recall)

        self.accuracy_lbl.setText("Accuracy : " + str(accuracy))
        self.precision_lbl.setText("Precision : " + str(precision))
        self.recall_lbl.setText("Recall : " + str(recall))
        self.specificity_lbl.setText("Specificity : " + str(specificity))
        self.false_positive_rate_lbl.setText("False Positive Rate : " + str(false_positive_rate_value))
        self.f1_score_lbl.setText("F1 Score : " + str(f1_score))

        # ROC Curve and AUC Score
        auc_value = self.calculate_auc(false_positive_rate, true_positive_rate)
        self.plot_roc(false_positive_rate, true_positive_rate, auc_value)



    def calculate_auc(self, false_positive_rate, true_positive_rate):
        """
       Method to calculate the area under the ROC curve.

       Args:
           false_positive_rate: The false positive rate.
           true_positive_rate: The true positive rate.

       Returns:
           The area under the ROC curve.
       """
        # Sort the TPR values in ascending order
        sorted_indices = sorted(range(len(false_positive_rate)), key=lambda i: false_positive_rate[i])
        sorted_fpr = [false_positive_rate[i] for i in sorted_indices]
        sorted_tpr = [true_positive_rate[i] for i in sorted_indices]

        auc_value = 0.0
        prev_fpr = 0.0
        prev_tpr = 0.0

        # Use the trapezoidal rule to calculate the area under the curve
        for fpr, tpr in zip(sorted_fpr, sorted_tpr):
            # Calculate the area of the trapezoid formed by the current and previous points
            auc_value += 0.5 * (fpr - prev_fpr) * (tpr + prev_tpr)

            # Update the previous FPR and TPR values for the next iteration
            prev_fpr = fpr
            prev_tpr = tpr

        return auc_value

    def plot_roc(self, false_positive_rate, true_positive_rate, auc_value):
        """
        Method to plot the ROC curve.

        Args:
           false_positive_rate: The false positive rate.
           true_positive_rate: The true positive rate.
               auc_value: The area under the ROC curve.
       """
        # Sort the TPR values in ascending order
        sorted_indices = sorted(range(len(false_positive_rate)), key=lambda i: false_positive_rate[i])
        sorted_fpr = [false_positive_rate[i] for i in sorted_indices]
        sorted_tpr = [true_positive_rate[i] for i in sorted_indices]

        fig = plt.figure(figsize=(8, 6))
        plt.plot(sorted_fpr, sorted_tpr, color='blue', label='ROC Curve')
        plt.plot([0, 1], [0, 1], '--', color='gray')  # Random classifier line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve with threshold = ' + str(self.recog_slider_2.value()) + ' and AUC = ' + str(auc_value))
        plt.text(0.6, 0.2, color='red', s='AUC = ' + str(auc_value))
        plt.grid(True)
        canvas = FigureCanvas(fig)
        canvas.draw()  # Render the canvas
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(int(height), int(width), 4)

        # Convert numpy array to QImage
        qimage = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGBA8888)

        # Convert QImage to QPixmap and set it as the pixmap of roc_lbl
        pixmap = QPixmap.fromImage(qimage)
        self.roc_lbl.setPixmap(pixmap)

    def recognize_face_slider_change(self):
        """
        Method to handle the change in the recognition threshold slider.
        """
        self.recog_slider_lbl.setText("Recognition threshold : " + str(self.recog_slider.value()))

    def recognize_face_slider_change_2(self):
        """
        Method to handle the change in the recognition threshold slider.
        """
        self.recog_slider_lbl_2.setText("Recognition threshold : " + str(self.recog_slider_2.value()))
        self.perform_pca()


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