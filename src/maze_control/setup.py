from glob import glob

from setuptools import find_packages, setup

package_name = 'maze_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Steven Wu',
    maintainer_email='chiachien_wu@berkeley.edu',
    description='Ball controllers for the UR7e ball-in-maze (PID path tracking).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pid_node = maze_control.pid.node:main',
        ],
    },
)
