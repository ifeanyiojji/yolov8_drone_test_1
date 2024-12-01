import sys
import cv2
import requests
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from djitellopy import Tello


class TelloGUI(QWidget):
    def __init__(self, esp_ip):
        super().__init__()
        self.esp_ip = esp_ip
        self.init_ui()
        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()

        # Timer for updating video feed
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_video_feed)
        self.video_timer.start(30)  # 30ms for ~33 FPS

        # Timer for updating temperature data
        self.temp_timer = QTimer(self)
        self.temp_timer.timeout.connect(self.update_temperature)
        self.temp_timer.start(2000)  # Every 2 seconds

    def init_ui(self):
        self.setWindowTitle("Tello Drone + Temperature Data")

        # Video label
        self.video_label = QLabel(self)
        self.video_label.setText("Connecting to Tello video feed...")
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.video_label.setFixedSize(640, 480)

        # Temperature label
        self.temp_label = QLabel("Temperature: -- °C")
        self.temp_label.setStyleSheet("font-size: 20px; color: red;")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.temp_label)
        self.setLayout(layout)

    def update_video_feed(self):
        # Capture frame from Tello
        frame = self.tello.get_frame_read().frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def update_temperature(self):
        try:
            # Fetch temperature from ESP01
            response = requests.get(f"http://{self.esp_ip}/temperature")
            data = response.json()
            temperature = data["temperature"]
            self.temp_label.setText(f"Temperature: {temperature:.2f} °C")
        except Exception as e:
            print("Error fetching temperature:", e)
            self.temp_label.setText("Temperature: -- °C")

    def closeEvent(self, event):
        # Stop Tello video stream on close
        self.tello.streamoff()
        self.tello.end()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    esp_ip = "ESP01_IP_ADDRESS"  # Replace with the ESP01's IP address
    gui = TelloGUI(esp_ip)
    gui.show()
    sys.exit(app.exec_())
