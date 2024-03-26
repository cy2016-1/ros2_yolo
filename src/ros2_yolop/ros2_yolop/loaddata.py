import os

import cv2
import numpy as np
class VideoReader:
    def __init__(self,path,img_size=640):
        self.img_size = img_size
        files = os.listdir(path)
        self.nf = len(files)
        self.paths = [path+file for file in files]
        path = self.paths[0]
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame = 0

        pass

    def __iter__(self):
        self.count = 0
        return self
    def __len__(self):
        return self.nf
    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.paths[self.count]

        if '.mp4' in path:
            # Read video
            ret_val, img0 = self.cap.read()

            # end video change count
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.paths[self.count]
                    self.cap = cv2.VideoCapture(path)
                    self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.frame = 0
                    ret_val, img0 = self.cap.read()
            h0, w0 = img0.shape[:2]
            self.frame += 1
            img, ratio, pad = letterbox_for_img(img0, new_shape=self.img_size)
            h, w = img.shape[:2]
            shapes = (h0, w0), ((h / h0, w / w0), pad)

            img = np.ascontiguousarray(img)
            # cv2.imshow('pic',img)
            # cv2.waitKey(0)
            # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
            return path, img, img0, self.cap, shapes

# resize img and put padding
def letterbox_for_img(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

