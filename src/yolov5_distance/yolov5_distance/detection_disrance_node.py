import cv2
import rclpy
import numpy as np
from rclpy.node import Node
# if run this python script should remove .
# it is correct to add . when run in ros2
from .utils import Colors, yaml_load,preprocessing_img,do_NMS,scale_boxes,box_label
from sensor_msgs.msg import Image
from std_msgs.msg import String,Int32MultiArray,Float32MultiArray
from cv_bridge import CvBridge 
import time
class Inference_node(Node):
    def __init__(self,node_name,model):
        super().__init__(node_name)
        self.model = model
        self.iou_thres = 0.25  # TF.js NMS: IoU threshold
        self.conf_thres = 0.45
        self.new_shape = (640,640)
        # gen color
        self.colors = Colors()
        # load names
        data = "/home/ljb/temp/coco128.yaml"
        self.names = yaml_load(data)["names"]
        self.cv_bridge = CvBridge()
        self.get_logger().info(f"start node:{node_name}")
        
        self.bridge = CvBridge()
        self.img_sub, self.depth_shape_sub, self.depth_ori_sub = None,None,None
        self.dt = None
        self.clr = None
        self.depth = None
        self.dep_shape = None
        self.k = None
        self.dist_coeffs = None
        self.factor = 5000
        self.result_pub = self.create_publisher(Image,'result',10)
        
        self.k_sub = self.create_subscription(Float32MultiArray,'K',self.K_call,1)
        self.img_sub = self.create_subscription(Image,'rgbd_color_topic',self.clr_call,10)
        self.depth_ori_sub = self.create_subscription(Int32MultiArray,'rgbd_depth_ori_topic',self.dep_call,10)
        self.depth_shape_sub = self.create_subscription(Int32MultiArray,'rgbd_depth_shape_topic',self.dep_spe_call,10)
    
    def K_call(self,k):
        np_array = np.array(k.data, dtype=np.float32)
        self.k = np.reshape(np_array,(3,3))

    def clr_call(self,clr):
        cv_image = self.bridge.imgmsg_to_cv2(clr, desired_encoding="bgr8")
        self.clr = cv_image
        self.process_messages()

    def dep_call(self,depth):
        np_array = np.array(depth.data, dtype=np.int32)
        self.depth = np_array
        self.process_messages()   
    
    def dep_spe_call(self,shape):
        np_array = np.array(shape.data, dtype=np.int32)
        self.dep_shape = np_array
        self.process_messages()

    def process_messages(self):
        if self.clr is not None and self.depth is not None and self.dep_shape is not None and self.k is not None:            
            depth_img = np.reshape(self.depth,self.dep_shape)
            depth_img = depth_img.astype(np.uint16)
        
            cv_img = self.clr
            input = preprocessing_img(cv_img,new_shape=self.new_shape)
            t = time.time()
            self.model.setInput(input)
            y = self.model.forward()
            self.get_logger().info(f'inference time:{time.time()-t} s')
            
            pred = do_NMS(y,conf_thres=self.conf_thres,iou_thres=self.iou_thres)
            for i, det in enumerate(pred):
                #print(len(det))
                det[:, :4] = scale_boxes(input.shape[2:], det[:, :4], cv_img.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    # print(c,conf)
                    # z = depth[v,u]/factor*1000
                    x1,y1,x2,y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                    # mm
                    z = np.mean(depth_img[x1:x2,y1:y2])/self.factor*1000
                    # to m
                    z = z/1000
                    box_label(xyxy, im=cv_img, color=self.colors(c, True), label=self.names[c]+f' {round(z,2)}m')
                # cv2.imshow('result', cv_img)
                # cv2.waitKey(20)
                # cv2.destroyAllWindows()
                # print(pred.shape)
            result = self.cv_bridge.cv2_to_imgmsg(cv_img,'bgr8')
            self.result_pub.publish(result)
            self.clr = None
            self.depth = None
            self.dep_shape = None

def main(args=None): 
    model = cv2.dnn.readNetFromONNX('/home/ljb/temp/yolov5s.onnx')

    rclpy.init(args=args)
    node = Inference_node(node_name="detection_distance",model = model)  
    rclpy.spin(node) 
    rclpy.shutdown() 

if __name__ == '__main__':
    main()
