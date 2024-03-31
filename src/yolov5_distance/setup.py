from setuptools import find_packages, setup

package_name = 'yolov5_distance'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ljb',
    maintainer_email='lijiabin2018@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rgbd_pub_node = yolov5_distance.rgbd_publish_node:main',
            'det_dis = yolov5_distance.detection_disrance_node:main'
        ],
    },
)
