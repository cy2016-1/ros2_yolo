import os
import cv2
from openvino.inference_engine import IECore
import torch
import time
import yaml
import numpy as np
import logging
import sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
LOGGING_NAME = "yolov5"
load_log = logging.getLogger('load_log')
inf_log = logging.getLogger('inf_log')
LOGGER = logging.getLogger(LOGGING_NAME)
import threading
import queue

def yaml_load(file="data.yaml"):
    """Safely loads and returns the contents of a YAML file specified by `file` argument."""
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)

def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    # print(type(x))
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    # print(device)
    mps = "mps" in device.type  # Apple MPS
    # print(mps)
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    # redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    # merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        # replace torchvision
        boxes, scores = boxes.numpy(),scores.numpy()
        # print(boxes.shape,type(scores))
        i = cv2.dnn.NMSBoxes(boxes, scores,score_threshold=conf_thres,nms_threshold=iou_thres)
        i = i[:max_det]  # limit detections
        # if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
        #     # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def clip_boxes(boxes, shape):
    """Clips bounding box coordinates (xyxy) to fit within the specified image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """Rescales (xyxy) bounding boxes from img1_shape to img0_shape, optionally using provided `ratio_pad`."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resizes and pads image to new_shape with stride-multiple constraints, returns resized image, ratio, padding."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # print(new_unpad)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # print(dw,dh)
    return im, ratio, (dw, dh)

def from_numpy(x):
    device = torch.device('cpu')
    """Converts a NumPy array to a torch tensor, maintaining device compatibility."""
    return torch.from_numpy(x).to(device) if isinstance(x, np.ndarray) else x

def box_label(box, im, label="", color=(128, 128, 128), txt_color=(255, 255, 255)):
    """Add one xyxy box to image with label."""
    if isinstance(box, torch.Tensor):
        box = box.tolist()
    lw = 3 or max(round(sum(im.shape) / 2 * 0.003), 2)
    tf = max(lw - 1, 1)  # font thickness
    sf = lw / 3
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    im = cv2.rectangle(im, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)
    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im,label,(p1[0], p1[1] - 2 if outside else p1[1] + h + 2),0,sf,txt_color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return im

class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))

def read_preprocessing_img(img,new_shape):
    # print(img.shape)
    im = letterbox(img, new_shape=new_shape, auto=False)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im)
    im = im.float()
    im = im / 255
    im = im.cpu().numpy()
    im = im[None]
    # print(im.shape)
    return im,img

def do_NMS(y,conf_thres=0.45,iou_thres=0.25):
    if isinstance(y, (list, tuple)):
        pred = from_numpy(y[0]) if len(y) == 1 else [from_numpy(x) for x in y]
    else:
        pred = from_numpy(y)

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=1000)
    # print(pred[0].shape)
    return pred

def load_data(data_dir,q,LOGGER):
    IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
    VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "webm","wmv"  # include inf_dir suffixes

    file_list = os.listdir(data_dir)
    nums_dict = {'imgs': 0, 'videos': 0}
    videos_list = []
    images_list = []
    for f_n in file_list:
        if f_n.split('.')[-1] in IMG_FORMATS:
            nums_dict['imgs']+=1
            images_list.append(f_n)
        elif f_n.split('.')[-1] in VID_FORMATS:
            nums_dict['videos']+=1
            videos_list.append(f_n)
        else:
            LOGGER.warning(f'file type {f_n} unrecognized')

    LOGGER.info('processing videos')
    if len(videos_list) == 0:
        LOGGER.info(f'No videos in dir {data_dir}')
    else:
        LOGGER.info('get {} videos'.format(nums_dict['videos']))
    for v in videos_list:
        LOGGER.info('reading video {}'.format(v))
        video_path = data_dir+v
        cap = cv2.VideoCapture(video_path)
        num_frames = 0
        ref,frame = cap.read()
        while ref:
            if num_frames %8 ==0:
                q.put(frame)
            num_frames+=1
            cv2.imshow('show',frame[::2,::2])
            cv2.waitKey(50)
            ref,frame = cap.read()
        # cv2.destroyAllWindows()
        cap.release()
        LOGGER.info(f'red {num_frames} frames in vided {v}')

    LOGGER.info('processing images')
    if len(images_list) == 0:
        LOGGER.info(f'No images in dir {data_dir}')
    else:
        LOGGER.info('get {} images'.format(nums_dict['images']))
    for m in images_list:
        img = cv2.imread(data_dir+m)
        cv2.imshow('show', img[::2,::2])
        cv2.waitKey(50)
        q.put(img)
    # cv2.destroyAllWindows()

def run():
    iou_thres = 0.25  # TF.js NMS: IoU threshold
    conf_thres = 0.45
    new_shape = (640,640)
    # gen color
    colors = Colors()
    # load names
    data = "coco128.yaml"
    names = yaml_load(data)["names"]

    # load dnn model
    model = cv2.dnn.readNetFromONNX('yolov5s.onnx')

    # read and preprocess img
    img = cv2.imread('/home/ljb/dataset/yolov5/inf_dir/zidane.jpg')
    im, img = read_preprocessing_img(img=img,new_shape=new_shape)

    # inference
    t = time.time()
    model.setInput(im)
    y = model.forward()
    print(time.time()-t)
    # NMS
    pred = do_NMS(y,conf_thres=conf_thres,iou_thres=iou_thres)
    for i, det in enumerate(pred):
        #print(len(det))
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            # print(c,conf)
            box_label(xyxy, im=img, color=colors(c, True), label=names[c])
        cv2.imshow('result', img)
        cv2.waitKey(20)
        cv2.destroyAllWindows()
        print(img.shape)

def inf(q,model,iou_thres,conf_thres,names,colors,log):
    while True:
        # read and preprocess img
        img = q.get()
        im, img = read_preprocessing_img(img,new_shape=new_shape)
        # inference
        t = time.time()
        model.setInput(im)
        y = model.forward()
        log.info('inference:{}'.format(time.time()-t))

        # NMS
        pred = do_NMS(y,conf_thres=conf_thres,iou_thres=iou_thres)
        for i, det in enumerate(pred):
            # print(len(det))
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                # print(c,conf)
                box_label(xyxy, im=img, color=colors(c, True), label=names[c])
            cv2.imshow('result', img[::2,::2])
            cv2.waitKey(10)
            # cv2.destroyAllWindows()
            # print(img.shape)
'''
def onnx2vino():


    # 将 ONNX 模型转换为 OpenVINO 模型
    ov_model = ov.convert_model('yolov5s.onnx', framework='onnx')
    ov.serialize(ov_model,'yolov5s.xml')
    ov.save_weigths(ov_model,'yolovs.bin')
    # 加载转换后的 OpenVINO 模型

    model = cv2.dnn.readNetFromModelOptimizer('yolov5s.xml', 'yolov5s.bin')

    # 读取和预处理图像
    img = cv2.imread('data/images/zidane.jpg')
    im, img = read_preprocessing_img(img=img, new_shape=new_shape)
    t= time.time()
    # 进行推理
    model.setInput(im)
    y = model.forward()
    print(time.time()-t)
'''
if __name__=='__main__':
    run()
    # onnx2vino()
    # data_dir = '/home/ljb/dataset/yolov5/inf_dir/'
    # imgs_queue = queue.Queue()

    # iou_thres = 0.25
    # conf_thres = 0.45
    # new_shape = (640,640)
    # # gen color
    # colors = Colors()
    # # load names
    # data = "coco128.yaml"
    # names = yaml_load(data)["names"]
    # # load dnn model
    # model = cv2.dnn.readNetFromONNX('yolov5s.onnx')


    # read_thread = threading.Thread(target=load_data,args=(data_dir,imgs_queue,load_log,))
    # read_thread.start()
    # # time.sleep(1)
    # inf_thread = threading.Thread(target=inf,args=(imgs_queue,model,iou_thres,conf_thres,names,colors,inf_log,))
    # inf_thread.start()

    # cv2.destroyAllWindows()