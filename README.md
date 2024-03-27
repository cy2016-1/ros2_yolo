# ROS2 yolo project
## RUN ROS2+yolov5
### Get dataset,you should [download this data](https://www.123pan.com/s/evBjVv-3TG0.html) into /home/yourname/dataset/yolov5/infr_dir/
### create workspace
```bash
cd ~
mkdir ros2_deployment
cd ros2_deployment/
mkdir hub_dir ros2_project
```
### copy source code
```bash
cd ~/ros2_deployment/hub_dir/
git clone https://gitee.com/Bin_lj/ros2_yolo.git
```
### create ros2 project
```bash
cd ~/ros2_deployment/ros2_project/
mkdir src 
colcon build
```
### create package
```bash
cd ~/ros2_deployment/ros2_project/src/
ros2 pkg create --build-type ament_python --node-name data_read_node ros2_yolov5 
```

### modify files
```bash
cd ~/ros2_deployment/ros2_project/src/ros2_yolov5/ros2_yolov5
cp  -r ~/ros2_deployment/hub_dir/ros2_yolo/src/ros2_yolov5/ros2_yolov5/* .
gedit ~/ros2_deployment/ros2_project/src/ros2_yolov5/setup.py
```
copy following in *console_scripts* list
`inference_node = ros2_yolov5.inference_node:main`

### build&run
```bash
cd ~/ros2_deployment/ros2_project/
colcon build
source install/setup.bash 
ros2 pkg list (check ros2_yolov5 pkg) 
ros2 run ros2_yolov5 inference_node
ros2 run ros2_yolov5 data_read_node
```