import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2 as ROS2PointCloud2, PointField
from std_msgs.msg import Header
from mcap_ros2.reader import read_ros2_messages
import paho.mqtt.client as mqtt
import json


class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')
        self.publisher = self.create_publisher(ROS2PointCloud2, 'lidar_data', 10)
 
        # Paths to .mcap files
        self.pointcloud_mouse = "pointcloud_mouse.mcap"
        self.pointcloud_map = "pointcloud_map.mcap"

        # MQTT configuration
        self.mqtt_broker = "1af86e820ce44ff190357e5093946b95.s1.eu.hivemq.cloud"
        self.mqtt_port = 8883
        self.mqtt_topic = "ros2/lidar_data"
  
        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt_client.tls_set()
        self.mqtt_client.tls_insecure_set(True)
        self.mqtt_client.username_pw_set("hivemq.webclient.1736921946646", "pCLzFlbJ?,My76;f1#X0")
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.mqtt_client.loop_start()

        # Publish data immediately after initialization
        self.publish_lidar_data(self.pointcloud_mouse, "pointcloud_mouse")
        self.publish_lidar_data(self.pointcloud_map, "pointcloud_map")

    def on_mqtt_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self.get_logger().info("Connected to MQTT broker")
        else:
            self.get_logger().error(f"Failed to connect to MQTT broker: {reason_code}")

    def on_mqtt_disconnect(self, client, userdata, reason_code, properties):
        self.get_logger().warn(f"Disconnected from MQTT broker: {reason_code}")

    def publish_lidar_data(self, file_path, description):
        """
        Read PointCloud data from the specified .mcap file, publish it to ROS 2, and send it via MQTT.
        """
        try:
            for msg in read_ros2_messages(file_path, topics=["/unilidar/cloud"]):
                PointCloud2Type = type(msg.ros_msg)
                if isinstance(msg.ros_msg, PointCloud2Type):
                    self.get_logger().info(f"Publishing {description}")

                    # Convert to ROS2 PointCloud2
                    ros2_msg = self.convert_to_ros2_pointcloud2(msg.ros_msg)
                    self.publisher.publish(ros2_msg)
                    self.get_logger().info(f'Published PointCloud to ROS 2: {description}')

                    # Serialize the PointCloud2 message for MQTT
                    mqtt_data = self.serialize_pointcloud(ros2_msg)

                    # Publish the serialized data to the MQTT broker
                    self.mqtt_client.publish(self.mqtt_topic, mqtt_data)
                    self.get_logger().info(f'Published PointCloud to MQTT: {description}')

                    break  # Publish only the first message
        except Exception as e:
            self.get_logger().error(f"Failed to read or publish {description}: {e}")

    def convert_to_ros2_pointcloud2(self, dynamic_msg):
        """
        Convert a dynamic PointCloud2 message to the standard ROS2 PointCloud2 message.
        """
        ros2_msg = ROS2PointCloud2()
        ros2_msg.header = Header()
        ros2_msg.header.stamp = self.get_clock().now().to_msg()
        ros2_msg.header.frame_id = dynamic_msg.header.frame_id
        ros2_msg.fields = [
            PointField(name=field.name, offset=field.offset, datatype=field.datatype, count=field.count)
            for field in dynamic_msg.fields
        ]
        ros2_msg.height = dynamic_msg.height
        ros2_msg.width = dynamic_msg.width
        ros2_msg.is_bigendian = dynamic_msg.is_bigendian
        ros2_msg.point_step = dynamic_msg.point_step
        ros2_msg.row_step = dynamic_msg.row_step
        ros2_msg.data = dynamic_msg.data
        ros2_msg.is_dense = dynamic_msg.is_dense

        return ros2_msg
    
    def serialize_pointcloud(self, ros2_msg):
        """
        Serialize the ROS2 PointCloud2 message for MQTT.
        """
        data = {
            "header": {
                "frame_id": ros2_msg.header.frame_id,
                "stamp": {
                    "sec": ros2_msg.header.stamp.sec,
                    "nanosec": ros2_msg.header.stamp.nanosec,
                },
            },
            "height": ros2_msg.height,
            "width": ros2_msg.width,
            "is_bigendian": ros2_msg.is_bigendian,
            "point_step": ros2_msg.point_step,
            "row_step": ros2_msg.row_step,
            "is_dense": ros2_msg.is_dense,  # Add this field
            "fields": [
                {"name": field.name, "offset": field.offset, "datatype": field.datatype, "count": field.count}
                for field in ros2_msg.fields
            ],
            "data": list(ros2_msg.data),  # Convert byte array to list
        }
        return json.dumps(data)

    def destroy_node(self):
        """
        Clean up the MQTT client when the node is destroyed.
        """
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()