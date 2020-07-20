import torch.backends.cudnn as cudnn
from yolov5.models.experimental import *
from yolov5.utils.datasets import *
from yolov5.utils.utils import *
import cv2
import torch

def Attempt_Load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    model.append(torch.load(weights, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble

def load_weights(weights,device):
    device = torch_utils.select_device(device)
    half = device.type != 'cpu'

    model = Attempt_Load(weights,device)
    return model,half,device

def detect(cap_img,imgsz=640,save_img=False,view_img=True):
    weights_path = '/home/zhangcheng/PycharmProjects/Detra_ui/yolov5/weights/yolov5s.pt'
    model,half,device = load_weights(weights=weights_path,device='')
    imgsz = check_img_size(imgsz,s=model.stride.max())
    if half:
        model.half()

    vid_path,vid_writer = None,None
    view_img = True
    cudnn.benchmark = True

    # set names and colors
    names = model.module.names if hasattr(model,'module') else model.names
    colors = [[random.randint(0,255) for _ in range(3)] for _ in range(len(names))]

    # init img
    img = torch.zeros((1,3,imgsz,imgsz),device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    # read cap to get the cap_img
    assert cap_img is not None, 'Image Not Found '
    img = letterbox(cap_img, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    # NMS
    pred = non_max_suppression(pred,conf_thres=0.4,iou_thres=0.5)
    s = ''
    gn = torch.tensor(cap_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for i,det in enumerate(pred):
        if pred is not None and len(det):

            s += '%gx%g ' % img.shape[2:]  # print string
            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in det:
                if view_img:  # Add bbox to image
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, cap_img, label=label, color=colors[int(cls)], line_thickness=3)

    return cap_img


if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    ret,frame = capture.read()
    frame = detect(frame,imgsz=640)
    cv2.imshow("test", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

