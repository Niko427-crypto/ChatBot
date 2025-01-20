import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2 as ROS2PointCloud2, PointField  # Standard ROS2 PointCloud2
from std_msgs.msg import Header  # Import Header for the header field
from mcap_ros2.reader import read_ros2_messages
import time

class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')
        self.publisher = self.create_publisher(ROS2PointCloud2, 'lidar_data', 10)

        # Path to your .mcap file
        self.pointcloud_mouse = "pointcloud_mouse.mcap"
        self.pointcloud_map = "pointcloud_map.mcap"

        # Call the publish_lidar_data method periodically (e.g., every 1 second)
        self.timer = self.create_timer(3.0, self.publish_lidar_data)

    def publish_lidar_data(self):
        # Read PointCloud data from the .mcap file
        for msg in read_ros2_messages(self.pointcloud_mouse, topics=["/unilidar/cloud"]):
            # print("Message type:", type(msg.ros_msg))  # Debug: Print the type of the message

            # Dynamically access the PointCloud2 type
            PointCloud2Type = type(msg.ros_msg)

            if isinstance(msg.ros_msg, PointCloud2Type):  # Use the dynamic PointCloud2 type
                print("publish pointcloud_mouse")
                # Convert the dynamic PointCloud2 to the standard ROS2 PointCloud2
                ros2_msg = self.convert_to_ros2_pointcloud2(msg.ros_msg)
                self.publisher.publish(ros2_msg)
                self.get_logger().info('Published PointCloud pointcloud_mouse')
                break  # Publish only the first message for now
        #second data publisher
        time.sleep(1)
        for msg in read_ros2_messages(self.pointcloud_map, topics=["/unilidar/cloud"]):
            # print("Message type:", type(msg.ros_msg))  # Debug: Print the type of the message

            # Dynamically access the PointCloud2 type
            PointCloud2Type = type(msg.ros_msg)

            if isinstance(msg.ros_msg, PointCloud2Type):  # Use the dynamic PointCloud2 type
                print("publish pointcloud_map")
                # Convert the dynamic PointCloud2 to the standard ROS2 PointCloud2
                ros2_msg = self.convert_to_ros2_pointcloud2(msg.ros_msg)
                self.publisher.publish(ros2_msg)
                self.get_logger().info('Published PointCloud pointcloud_map')
                break  # Publish only the first message for now

    def convert_to_ros2_pointcloud2(self, dynamic_msg):
        # Convert the dynamic PointCloud2 message to the standard ROS2 PointCloud2 message
        ros2_msg = ROS2PointCloud2()

        # Manually construct the header
        ros2_msg.header = Header()
        ros2_msg.header.stamp = self.get_clock().now().to_msg()
        ros2_msg.header.frame_id = dynamic_msg.header.frame_id  # Copy frame_id from dynamic message
        print(ros2_msg.header.frame_id)
        # Convert fields to a list of PointField objects
        ros2_msg.fields = [
            PointField(name=field.name, offset=field.offset, datatype=field.datatype, count=field.count)
            for field in dynamic_msg.fields
        ]

        # Copy other fields from the dynamic message
        ros2_msg.height = dynamic_msg.height
        ros2_msg.width = dynamic_msg.width
        ros2_msg.is_bigendian = dynamic_msg.is_bigendian
        ros2_msg.point_step = dynamic_msg.point_step
        ros2_msg.row_step = dynamic_msg.row_step
        ros2_msg.data = dynamic_msg.data
        ros2_msg.is_dense = dynamic_msg.is_dense

        return ros2_msg

def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()