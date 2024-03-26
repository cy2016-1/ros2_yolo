import cv2
import torch
import random
from yolop import get_net
import time
import numpy as np
from utils import letterbox_for_img
from utils import non_max_suppression,scale_coords
from utils import plot_one_box
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
half = device.type != 'cpu'

# origin shaoe 1080x1920
h, w = 1080, 1920
# scale down to 1080/3 and 1920 /3
height, width = 384,640
#
pad_h, pad_w = 12,0
ratio = 0.3333333333333333

def load_net():
    model = get_net()
    checkpoint = torch.load('/home/ljb/temp/End-to-end.pth', map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    # model = model.cuda()
    model.eval()
    half = device.type != 'cpu'
    if half:
        model.half()  # to FP16
    return model

def predict_img(img,model):
    img_tsr, new_win = img
    det_out, da_seg_out, ll_seg_out = model(img_tsr.to(device))
    # print(len(det_out),da_seg_out.size(),ll_seg_out.size())
    da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
    # print(da_predict.size())
    da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1 / ratio), mode='bilinear')
    _, da_seg_mask = torch.max(da_seg_mask, 1)
    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
    # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)
    # print(da_seg_mask.shape)
    ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
    # print(ll_predict.size())
    ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1 / ratio), mode='bilinear')
    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
    # print(ll_seg_mask.shape)
    return (det_out,da_seg_mask,ll_seg_mask,new_win)

def run(half):
    inf_video = '/home/ljb/dataset/yolop/gta.mp4'
    conf_thres = 0.25
    iou_thres = 0.45
    model = load_net()
    model.to(device)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    cap = cv2.VideoCapture(inf_video)
    
    model.eval()
    ret,frame = cap.read()
    with torch.no_grad():
        while ret:
            h,w = frame.shape[:2]
            print(w,h)
            t = time.time()
            img, ratio, pad = letterbox_for_img(frame, new_shape=(640, 640))
            img = np.ascontiguousarray(img)
            
            img_np = np.array(img).astype(np.float32) / 255.0
            normalized_img = (img_np - MEAN) / STD
            normalized_img = normalized_img.transpose((2, 0, 1))

            img_tsr = torch.from_numpy(normalized_img)
    
            img_tsr = img_tsr.half() if half else img_tsr.float()  # uint8 to fp16/32
            if img_tsr.ndimension() == 3:
                img_tsr = img_tsr.unsqueeze(0)

            det,da,ll,re = predict_img((img_tsr,frame),model)
            # print(re.shape)
            # print(da.shape)
            print('inf:',time.time()-t)
            inf_out, _ = det
            # print(len(inf_out))
            print(img.shape)
            det_pred = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, classes=None,
                                        agnostic=False)
            det = det_pred[0]
            if len(det):
                det[:, :4] = scale_coords(img.shape[:2], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label_det_pred, color=colors[int(cls)], line_thickness=2)

            color_segmentation = np.zeros(re.shape, dtype=np.uint8)
            color_segmentation[da == 1] = [255, 0, 0]  # 自定义第一种分割区域的颜色为蓝色
            color_segmentation[ll == 1] = [0, 255, 0]  # 自定义第二种分割区域的颜色为绿色
            # 可以根据实际情况继续定义更多分割区域的颜色

            # 将彩色分割图像叠加到原始图像上
            result_image = cv2.addWeighted(re, 0.7, color_segmentation, 0.3, 0)
            cv2.imshow('win', result_image[::2,::2])
            cv2.waitKey(15)

            ret,frame = cap.read()

if __name__=='__main__':
    # load model
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('begin')
    run(half)