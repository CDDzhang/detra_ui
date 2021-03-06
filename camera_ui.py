from PyQt5.uic import loadUi
from PyQt5.Qt import *
import sys
import detect
import torch.backends.cudnn as cudnn
from yolov5.utils.datasets import *
from yolov5.utils.utils import *
import cv2
import torch
import operator
import ui_tracker


class select(QDialog):
    def __init__(self):
        super().__init__()
        self.resize(300,300)
        self.object_list = {}
        self.ReadLabel = False
        weights_path = '/home/zhangcheng/PycharmProjects/Detra_ui/yolov5/weights/yolov5s.pt'
        self.model,self.half,self.device = detect.load_weights(weights_path,device='')
        self.Detector = False
        self.Tracker = False
        self.track_model,self.tracker = ui_tracker.tracker_loadweight()
        self.xywh = []
        self.tracker_init_label = False
        self.frame_num = 0


        loadUi("UI/select.ui",self)
        self.showbutton.clicked.connect(self.startframe)
        self.stopbutton.clicked.connect(self.stopframe)
        self.Detectorbutton.clicked.connect(self.startDetector)
        self.Trackerbutton.clicked.connect(self.startTracker)
        self.Trackerstopbutton.clicked.connect(self.stopTracker)

        self.timer = QTimer()
        self.timer.start(24)
        self.timer.timeout.connect(self.webcam)

    def webcam(self):
        if (self.ReadLabel):
            self.frame_num += 1
            ret, frame = self.cap.read()
            img_height, img_width, img_depth = frame.shape
            if self.Detector == True and self.Tracker == False:
                frame,self.object_list = self.YOLO5_detect(frame)
                self.list_view()
            if self.Tracker == True and len(self.xywh) != 0 :
                if self.tracker_init_label == False:
                    self.state,self.tracker_init_label = ui_tracker.tracker_init(frame,self.xywh,self.tracker_init_label,self.tracker,self.track_model)
                else:
                    frame = ui_tracker.tracker_run(frame,self.tracker_init_label,self.state,self.tracker)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QImage(frame.data, img_width, img_height, img_width * img_depth, QImage.Format_RGB888)
            self.imglabel.setPixmap(QPixmap.fromImage(frame))


    def stopTracker(self):
        self.Tracker = False
        self.Detector = True
        self.tracker_init_label = False

    def startframe(self):
        self.ReadLabel = True
        self.cap = cv2.VideoCapture(0)
        self.object_list = {}

    def stopframe(self):
        self.ReadLabel = False
        self.Detector = False
        self.cap.release()

    def startDetector(self):
        self.Detector = True

    def startTracker(self):
        self.Tracker = True
        self.Detector = False
        self.xywh = self.getxywh()
        print(self.xywh)

    def YOLO5_detect(self,cap_img,imgsz=640,save_img=False,view_img=True):
        imgsz = check_img_size(imgsz, s=self.model.stride.max())
        if self.half:
            self.model.half()

        view_img = True
        cudnn.benchmark = True

        # set names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # init img
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None
        # read cap to get the cap_img
        assert cap_img is not None, 'Image Not Found '
        img = letterbox(cap_img, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]
        # NMS
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)
        s = ''
        object_list = {}
        gn = torch.tensor(cap_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for i, det in enumerate(pred):
            if pred is not None and det is not None:
                s += '%gx%g ' % img.shape[2:]  # print string
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    if view_img:  # Add bbox to image
                        list_i = 1
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1,4))/gn).view(-1).tolist()
                        label = '%s %.2f' % (names[int(cls)], conf)
                        class_name = label.split(' ')[0]
                        while(object_list.get(class_name+str(list_i)) is not None ):
                            list_i = list_i + 1
                        object_list[class_name+str(list_i)] = xywh
                        label = '%s %.2f' % (class_name+str(list_i), conf)
                        plot_one_box(xyxy, cap_img, label=label, color=colors[int(cls)], line_thickness=3)
            else:
                cap_img = cap_img
        return cap_img,object_list

    def list_view(self):
        widgetres = []
        count = self.objectlistview.count()
        for i in range(count):
            widgetres.append( self.objectlistview.item(i).text())
        if self.object_list is not None and self.Detector == True:
            if operator.eq(widgetres,list(self.object_list.keys())) == False:
                self.objectlistview.clear()
                self.objectlistview.addItems(list(self.object_list.keys()))


    def getxywh(self):
        item = self.objectlistview.currentItem().text()
        current_xywh = self.object_list[item]
        if current_xywh is not None:
            return current_xywh
        else:
            print("please check clicked either")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = select()
    w.show()
    sys.exit(app.exec_())
