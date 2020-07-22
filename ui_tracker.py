import os
import cv2
import numpy as np
import SiamDW.lib.models.models as models
from os.path import exists,join
from torch.autograd import Variable
from SiamDW.lib.tracker.siamrpn import SiamRPN
from easydict import EasyDict as edict
from SiamDW.lib.utils.utils import load_pretrain,cxy_wh_2_rect,get_axis_aligned_bbox,load_dataset,poly_iou


def tracker_loadweight():
    # load weight from path
    info = edict()
    info.arch = "SiamRPNRes22"
    info.dataset = 'NOUSE'
    info.epoch_test = True
    info.cls_type = 'thinner'

    net = models.__dict__[info.arch](anchors_nums=5, cls_type='thinner')
    tracker = SiamRPN(info)

    net = load_pretrain(net,'SiamDW/snapshot/CIResNet22_RPN.pth')
    net.eval()
    net = net.cuda()

    return net,tracker

def tracker_init(cap_img,initBB,init_already,tracker,model):
    if init_already == False:
        gw = cap_img.shape[0]
        gh = cap_img.shape[1]
        lx,ly,w,h = initBB[0]*gw,initBB[1]*gh,initBB[2]*gw,initBB[3]*gh
        target_pos = np.array([lx + w / 2, ly + h / 2])
        target_sz = np.array([w, h])
        state = tracker.init(cap_img, target_pos, target_sz, model)  # init tracker
        init_already = True
    return state,init_already

def tracker_run(cap_img,init_already,state,tracker):
    if init_already == True:
        state = tracker.track(state, cap_img)
        location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), int(location[1] + location[3])
        cv2.rectangle(cap_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return cap_img


