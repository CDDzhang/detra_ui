from PyQt5.uic import loadUi
from PyQt5.Qt import *
import sys
import cv2

class select(QDialog):
    def __init__(self):
        super().__init__()
        self.resize(300,300)
        self.ReadLabel = False


        loadUi("UI/select.ui",self)
        self.showbutton.clicked.connect(self.startframe)
        self.stopbutton.clicked.connect(self.stopframe)

        self.timer = QTimer()
        self.timer.start(24)
        self.timer.timeout.connect(self.webcam)

    def webcam(self):
        if (self.ReadLabel):
            ret, frame = self.cap.read()
            print(ret)
            img_height, img_width, img_depth = frame.shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QImage(frame.data, img_width, img_height, img_width * img_depth, QImage.Format_RGB888)
            self.imglabel.setPixmap(QPixmap.fromImage(frame))


    def startframe(self):
        self.ReadLabel = True
        self.cap = cv2.VideoCapture(0)

    def stopframe(self):
        self.ReadLabel = False
        self.cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = select()
    w.show()
    sys.exit(app.exec_())
