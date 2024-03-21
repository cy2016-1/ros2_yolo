import cv2
import rclpy
from rclpy.node import Node
# if run this python script should remove .
# it is correct to add . when run in ros2
from .utils import Colors, yaml_load,preprocessing_img,do_NMS,scale_boxes,box_label
from sensor_msgs.msg import Image
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
        self.get_logger().info(f"start reader:{node_name}")
        self.result_pub = self.create_publisher(Image,'result',10)
        self.img_sub = self.create_subscription(Image,'image',self._callback,10)

    def _callback(self,msg):
        cv_img = self.cv_bridge.imgmsg_to_cv2(msg,'bgr8')
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
                box_label(xyxy, im=cv_img, color=self.colors(c, True), label=self.names[c])
            # cv2.imshow('result', cv_img)
            # cv2.waitKey(20)
            # cv2.destroyAllWindows()
            # print(pred.shape)
        result = self.cv_bridge.cv2_to_imgmsg(cv_img,'bgr8')
        self.result_pub.publish(result)
def main(args=None): 
    model = cv2.dnn.readNetFromONNX('/home/ljb/temp/yolov5s.onnx')

    rclpy.init(args=args)
    node = Inference_node(node_name="yolov5_inf_node",model = model)  
    rclpy.spin(node) 
    rclpy.shutdown() 

if __name__ == '__main__':
    main()
