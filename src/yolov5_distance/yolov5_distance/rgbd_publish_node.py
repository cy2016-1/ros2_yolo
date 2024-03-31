import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray,Float32MultiArray,String
from cv_bridge import CvBridge 
# if run this python script should remove .
# it is correct to add . when run in ros2
from .TUM_datasets import TUM_RGBD_dataset


# publish RGBD info,include colors depth and camera config
class RGBD_publisher(Node):
    def __init__(self,node_name,dataset):
        super().__init__(node_name)
        self.get_logger().info(f"start rgbd_pub:{node_name}")
        # load dataset 
        self.dataset = dataset
        self.imgs_num = len(self.dataset.rgb_list)
        self.get_logger().info(f'read {len(self.dataset.rgb_list)} color imgs')


        # create camera msg
        self.camera_k_pub = self.create_publisher(Float32MultiArray, "K", 1)
        # self.camera_dist_coeffs_pub = self.create_publisher(Float32MultiArray, "dist_coeffs", 1)
        
        
        # create ori depth msg
        self.depth_ori_publisher = self.create_publisher(Int32MultiArray, "rgbd_depth_ori_topic", 10)
        self.depth_shape_publisher = self.create_publisher(Int32MultiArray, "rgbd_depth_shape_topic", 10)
        
        # create color publisher ,queue length = 10
        self.clr_publisher = self.create_publisher(Image, "rgbd_color_topic", 10)
        
        # create color publisher ,queue length = 10
        self.depth_publisher = self.create_publisher(Image, "rgbd_depth_topic", 10)
        
        # save read index to publish
        self.count  = 0
        
        # create timer ,delay = 0.5s
        self.timer = self.create_timer(0.5,self.timer_callback)

        # cvbridge is to change format between cv2.Mat and sensor_msgs.Image
        self.bridge = CvBridge()

    def depth2color(self,depth):
        dmax,dmin = np.max(depth),np.min(depth)
        depth = (depth-dmin)/(dmax-dmin) * 255.0
        depth = depth.astype(np.uint16)
        return cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR).astype(np.uint8)


    def timer_callback(self):
        k = self.dataset.K.flatten().tolist()
        k_msg = Float32MultiArray()
        k_msg.data = k
        # print(k_msg)
        self.camera_k_pub.publish(k_msg)
        msg = String()
        msg.data = 'RGBD'

        # dist_coeffs = self.dataset.dist_coeffs.tolist()
        # dist_coeffs_msg = Float32MultiArray()
        # dist_coeffs_msg.data = dist_coeffs
        # # print(dist_coeffs_msg)
        # self.camera_dist_coeffs_pub.publish(dist_coeffs_msg)

        img_path,depth_path = self.dataset.path_list[self.count]
        img = cv2.imread(img_path)
        depth = cv2.imread(depth_path,cv2.IMREAD_ANYDEPTH)
        # print(depth.dtype)
        depth_arr = depth.flatten().tolist()
        depth_arr_msg = Int32MultiArray()
        depth_arr_msg.data = depth_arr

        depth_shape = [int(i)for i in depth.shape]
        depth_shape_msg = Int32MultiArray()
        depth_shape_msg.data = depth_shape

        colord_depth = self.depth2color(depth)
        print('number:',self.count)

        print(img.shape,depth.shape,colord_depth.shape)
        print(img.dtype,depth.dtype,colord_depth.dtype)

        self.depth_ori_publisher.publish(depth_arr_msg)
        self.depth_shape_publisher.publish(depth_shape_msg)

        self.clr_publisher.publish(self.bridge.cv2_to_imgmsg(img,'bgr8'))
        self.depth_publisher.publish(self.bridge.cv2_to_imgmsg(colord_depth,'bgr8'))
        self.count +=1
        if self.count >= self.imgs_num:
            self.count = 0
def main(args=None):
   
    tum_dataset = TUM_RGBD_dataset()
    rclpy.init(args=args)
    node = RGBD_publisher(node_name="RGBD_publisher",dataset=tum_dataset)  
    rclpy.spin(node) 
    rclpy.shutdown() 

if __name__ == '__main__':
    main()
