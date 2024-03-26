import cv2
import rclpy
import numpy as np
from rclpy.node import Node
import random
# if run this python script should remove .
# it is correct to add . when run in ros2
from .utils import preprocessing_img,predict_img,do_NMS
from .yolop import load_net
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
import time
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
half = device.type != 'cpu'

class Inference_node(Node):

    def __init__(self,node_name,model,device,half):
        super().__init__(node_name)
        self.MEAN = np.array([0.485, 0.456, 0.406])
        self.STD = np.array([0.229, 0.224, 0.225])
        self.conf_thres = 0.25
        self.iou_thres = 0.45
    
        self.model = model
        self.device = device
        self.half = half
        self.iou_thres = 0.25  # TF.js NMS: IoU threshold
        self.conf_thres = 0.45
        self.new_shape = (640,640)

        # origin shaoe 1080x1920
        self.h, self.w = 1080, 1920
        # scale down to 1080/3 and 1920 /3
        self.height, self.width = 384,640
        #
        self.pad_h, self.pad_w = 12,0
        self.ratio = 0.3333333333333333

        # Get names and colors
        self.names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

        self.cv_bridge = CvBridge()
        self.get_logger().info(f"start reader:{node_name}")
        self.result_pub = self.create_publisher(Image,'result',10)
        self.img_sub = self.create_subscription(Image,'image',self._callback,10)

    def _callback(self,msg):
        with torch.no_grad():
            cv_img = self.cv_bridge.imgmsg_to_cv2(msg,'bgr8')
            input,frame,img = preprocessing_img(cv_img,self.MEAN,self.STD,half=self.half,new_shape=self.new_shape)
            print()
            t = time.time()
            det,da,ll = predict_img(input,self.model,self.device,self.pad_h,self.pad_w,self.height,self.width,self.ratio)
            self.get_logger().info(f'inference time:{time.time()-t} s')
            
            re = do_NMS(det,frame,img.shape,conf_thres=self.conf_thres,iou_thres=self.iou_thres)
            print(re.shape)
            print(da.shape)
            color_segmentation = np.zeros(re.shape, dtype=np.uint8)
            color_segmentation[da == 1] = [255, 0, 0]  # 自定义第一种分割区域的颜色为蓝色
            color_segmentation[ll == 1] = [0, 255, 0]  # 自定义第二种分割区域的颜色为绿色
            # 可以根据实际情况继续定义更多分割区域的颜色

            # 将彩色分割图像叠加到原始图像上
            result_image = cv2.addWeighted(re, 0.7, color_segmentation, 0.3, 0)
            # cv2.imshow('win', result_image[::2,::2])
            # cv2.waitKey(15)
            result = self.cv_bridge.cv2_to_imgmsg(result_image,'bgr8')
            self.result_pub.publish(result)

def main(args=None): 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    half = device.type != 'cpu'
    path = '/home/ljb/temp/End-to-end.pth'
    model = load_net(device,path)
    rclpy.init(args=args)
    node = Inference_node(node_name="yolop_inf_node",model = model,device=device,half = half)  
    rclpy.spin(node) 
    rclpy.shutdown() 

if __name__ == '__main__':
    main()
