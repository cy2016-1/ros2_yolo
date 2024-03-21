import os
import logging
import sys
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
LOGGER = logging.getLogger(' yolv5 data loader')
class Yolov5_inf_dataset:
    def __init__(self):
        self.inf_dir = '/home/ljb/dataset/yolov5/inf_dir/'
        IMG_FORMATS = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
        VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "webm","wmv"  # include inf_dir suffixes
        self.file_list = os.listdir(self.inf_dir)
        self.nums_dict = {'imgs': 0, 'videos': 0}
        self.videos_list = []
        self.images_list = []
        for f_n in self.file_list:
            if f_n.split('.')[-1] in IMG_FORMATS:
                self.nums_dict['imgs']+=1
                self.images_list.append(self.inf_dir+f_n)
            elif f_n.split('.')[-1] in VID_FORMATS:
                self.nums_dict['videos']+=1
                self.videos_list.append(self.inf_dir+f_n)
            else:
                LOGGER.warning(f'file type {f_n} unrecognized')
