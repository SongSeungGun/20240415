import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer
from keras.models import load_model
import numpy as np

class PoseClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_model()
        self.cap = cv2.VideoCapture(0)  # 기본 카메라(인덱스 0)를 엽니다.

    def initUI(self):
        self.setWindowTitle('캠으로 현재 자세 확인하기')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.result_label = QLabel()
        self.result_label.setFont(QFont("Arial", 14))
        layout.addWidget(self.result_label)
        self.cap = cv2.VideoCapture(1)  # 기본 카메라(인덱스 0)를 엽니다.

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

    def load_model(self):
        # 분류 모델을 로드합니다.
        self.model = load_model("./model/keras_Model.h5", compile=False)
        self.class_names = open("./model/labels.txt", "r", encoding="utf-8").readlines()

    def update_frame(self):
        ret, frame = self.cap.read()  # 카메라로부터 프레임을 읽어옵니다.

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환합니다.
            frame = cv2.flip(frame, 1)  # 올바른 방향을 위해 좌우 반전합니다.

            # 분류 전에 프레임을 처리합니다. (크기 조정, 정규화 등)
            processed_frame = self.process_frame(frame)

            # 처리된 프레임을 분류합니다.
            prediction = self.model.predict(processed_frame)
            index = np.argmax(prediction)
            class_name = self.class_names[index]
            confidence_score = prediction[0][index]

            # 분류 결과를 포함하여 프레임을 표시합니다.
            self.display_frame(frame)
            self.display_result(f'자세 : {class_name[2:]} Synchronization : {confidence_score:.2f}')

    def process_frame(self, frame):
        # 분류 전에 프레임을 처리합니다. (크기 조정, 정규화 등)
        # 처리된 프레임을 numpy 배열로 반환합니다.
        target_size = (224, 224)
        processed_frame = cv2.resize(frame, target_size)
        processed_frame = (processed_frame.astype(np.float32) / 127.5) - 1
        processed_frame = np.expand_dims(processed_frame, axis=0)  # 배치 차원 추가
        return processed_frame

    def display_frame(self, frame):
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qimage = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        pixmap = pixmap.scaledToWidth(800)  # 창에 맞게 너비 조정
        self.image_label.setPixmap(pixmap)

    def display_result(self, result):
        self.result_label.setText(result)

    def closeEvent(self, event):
        self.cap.release()  # 애플리케이션을 닫을 때 카메라를 해제합니다.

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PoseClassifierApp()
    window.show()
    sys.exit(app.exec_())
