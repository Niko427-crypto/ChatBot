from setuptools import find_packages, setup

package_name = 'lidar_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='A LiDAR-based mouse detection system.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_node = lidar_node.lidar_node:main',
        ],
    },
)
