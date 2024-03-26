import os
import logging
import sys
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout)
LOGGER = logging.getLogger(' yolv5 data loader')
class Yolop_inf_dataset:
    def __init__(self):
        self.inf_dir = '/home/ljb/dataset/yolop/'
        VID_FORMATS = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "webm","wmv"  # include inf_dir suffixes
        self.file_list = os.listdir(self.inf_dir)
        self.nums_dict = {'videos': 0}
        self.videos_list = []
        self.images_list = []
        for f_n in self.file_list:
            if f_n.split('.')[-1] in VID_FORMATS:
                self.nums_dict['videos']+=1
                self.videos_list.append(self.inf_dir+f_n)
            else:
                LOGGER.warning(f'file type {f_n} unrecognized')
