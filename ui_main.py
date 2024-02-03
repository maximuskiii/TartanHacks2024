import sys
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap
import requests
import googlemaps
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from trajectory_planning import *
import subprocess
import os

api_key = "AIzaSyAWg_bkETJdwPx8X6ENtJRp3okVKm1Oyeo"
gmaps = googlemaps.Client(api_key)
directory = os.path.dirname(os.path.realpath(__file__))
image_path = os.path.join(directory, "graph.png")


# Load the .ui files directly
MainPageUi, MainPageBase = uic.loadUiType("main_page.ui")
NavigationDisplayUi, NavigationDisplayBase = uic.loadUiType(
    "navigation_display_page.ui"
)


class ImageDisplay(NavigationDisplayBase, NavigationDisplayUi):
    def __init__(self, image_path, parent=None):
        print("I am being initialized!")
        super().__init__(parent)
        self.setupUi(self)

        # Create a QLabel to display the image
        self.image_label = QLabel(self.matplotlibWidget)
        self.image_label.setGeometry(self.matplotlibWidget.geometry())

        print("location: ", image_path)

        # Load and display the image
        self.display_image(image_path)

    def display_image(self, image_path):
        print("display image called")
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)
        # Scale the image to fit the QLabel
        self.image_label.setScaledContents(True)


class MainPage(MainPageBase, MainPageUi):
    def __init__(self, parent=None):
        super(MainPage, self).__init__(parent)
        self.setupUi(self)
        # Connect the submit button to the method that transitions to the navigation display
        self.submit_button.clicked.connect(self.gotoNavigationDisplay)

    def save_graph_image(self, image_path):
        plt.savefig(image_path)
        plt.close()  # Close the plot to free up memory

    def addr_to_coord(self, addr):
        self.geocoding = gmaps.geocode(addr)
        if self.geocoding:
            lat = self.geocoding[0]["geometry"]["location"]["lat"]
            lng = self.geocoding[0]["geometry"]["location"]["lng"]
            print(lat, lng)
        return (lat, lng)

    def process_map_image(
        self,
        address,
        api_key=api_key,
        zoom=16,
        bar_height=30,
        size="600x300",
        maptype="roadmap",
    ):
        geocoding = self.addr_to_coord(address)
        location = f"{geocoding[0]},{geocoding[1]}"
        style = "feature:all|element:labels|visibility:off"
        map_url = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom={zoom}&size={size}&maptype={maptype}&style={style}&key={api_key}"

        # Fetch the map image
        response = requests.get(map_url)
        if response.status_code == 200:
            # Convert the image to a numpy array
            image = Image.open(BytesIO(response.content))
            image_array_rgb = np.array(image)
            img = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)
        else:
            print("Error fetching the map image")
            return

        # Image processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        black_img = np.zeros_like(img)
        cv2.drawContours(black_img, contours, -1, (255, 255, 255), 4)

        # Cropping
        height, width = black_img.shape[:2]
        new_height = height - bar_height
        cropped_black_img = black_img[0:new_height, 0:width]

        return np.array(cropped_black_img)

    # Optionally, save the processed image as a numpy array
    # np.save(npy_array_path, cropped_black_img)  # You need to define npy_array_path

    def gotoNavigationDisplay(self):
        # Getting the address from the input
        self.userText = self.address_input.toPlainText()
        # Processing the map image to get the array
        self.array = self.process_map_image(self.userText)
        # Assuming you save the graph as an image in a known location
        # Modify this function to save the graph image
        # self.save_graph_image('graph.png')
        from trajectory_planning import calculate_trajectory_and_save

        calculate_trajectory_and_save(self.array)
        only_image = os.path.basename(image_path)
        self.imageDisplay = ImageDisplay(image_path=only_image, parent=self)
        self.imageDisplay.show()  # Show the ImageDisplay UI
        print("we have passed showing")
        # self.close()  # Close the MainPage UI


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainPage = MainPage()
    mainPage.show()
    sys.exit(app.exec())
