import cv2
import rclpy
from rclpy.node import Node
from .dataloader import Yolop_inf_dataset
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 

class Data_reader(Node):
    def __init__(self,node_name,dataset):
        super().__init__(node_name)
        self.dataset = dataset
        self.dataset = dataset
        self.cv_bridge = CvBridge()
        self.get_logger().info(f"start reader:{node_name}")
        if len(self.dataset.videos_list) == 0:
            self.get_logger().warn(f'No videos in dir {self.dataset.inf_dir}')
        else:
            self.get_logger().info('get {} videos'.format(self.dataset.nums_dict['videos']))
        self.videos_idx = 0
        self.video_cap = None
        self.num_frames = 0
        self.img_pub = self.create_publisher(Image,'image',10)
        self.timer = self.create_timer(0.1,self.videos_callback)

    def videos_callback(self):
        # first read video,init cap and publish first frame
        if self.video_cap is None:
            video_name = self.dataset.videos_list[self.videos_idx].split('/')[-1]
            self.get_logger().info(f'process video {video_name}')
            self.video_cap = cv2.VideoCapture(self.dataset.videos_list[self.videos_idx])
        else:
            ret,frame = self.video_cap.read()

            # video finished
            if not ret:
                print(self.videos_idx)
                video_name = self.dataset.videos_list[self.videos_idx].split('/')[-1]
                self.get_logger().warn(f'get {self.num_frames} images from {video_name}')
                
                self.video_cap.release()
                self.num_frames = 0

                self.videos_idx += 1
                # destroy_timer
                if self.videos_idx >= len(self.dataset.videos_list):

                    self.get_logger().warn('all videos has published,begin to publish images')
                    self.num_frames = 0
                    self.timer.cancel()
                    self.destroy_timer(self.timer)
                else:
                    # read next 
                    video_name = self.dataset.videos_list[self.videos_idx].split('/')[-1]
                    self.get_logger().info(f'process video {video_name}')
                    self.video_cap = cv2.VideoCapture(self.dataset.videos_list[self.videos_idx])
                
            # not finfished
            else:
                img_msg = self.cv_bridge.cv2_to_imgmsg(frame,'bgr8')
                self.img_pub.publish(img_msg)
                self.num_frames+=1
def main(args=None): 
    yolv5_inf_dataset = Yolop_inf_dataset()
    rclpy.init(args=args)
    node = Data_reader(node_name="data_reader",dataset=yolv5_inf_dataset)  
    rclpy.spin(node) 
    rclpy.shutdown() 

if __name__ == '__main__':
    main()