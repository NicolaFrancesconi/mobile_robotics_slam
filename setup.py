from setuptools import find_packages, setup
from glob import glob

package_name = 'mobile_robotics_slam'

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
    maintainer='lanzo',
    maintainer_email='davide.lanzoni@sacmigroup.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'slam_node = mobile_robotics_slam.RosNodes.slam_node:main',
            	'corner_extractor_node = mobile_robotics_slam.RosNodes.corner_extractor_node_ros1:main',
            	'graph_slam_node = mobile_robotics_slam.RosNodes.graph_slam_node_ros1:main',
        ],
    },
)