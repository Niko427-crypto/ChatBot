import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
import paho.mqtt.client as mqtt
import json
import logging
from playsound import playsound
from plyer import notification
import os
import threading
import csv
from datetime import datetime


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')

        # Parameters for mouse detection
        self.mouse_radius = 0.1  # DBSCAN radius for clustering
        self.mouse_threshold = 10  # Minimum points to form a cluster
        self.noise_threshold = 0.02  # Radius for noise removal

        # MQTT configuration
        self.mqtt_broker = "1af86e820ce44ff190357e5093946b95.s1.eu.hivemq.cloud"
        self.mqtt_port = 8883
        self.mqtt_topic = "ros2/lidar_data"
        self.mqtt_username = "hivemq.webclient.1736921946646"
        self.mqtt_password = "pCLzFlbJ?,My76;f1#X0"

        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt_client.tls_set()
        self.mqtt_client.tls_insecure_set(True)
        self.mqtt_client.username_pw_set(self.mqtt_username, self.mqtt_password)
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
        self.mqtt_client.loop_start()

        # Store received chunks
        self.received_chunks = []

        # Log file configuration
        self.log_file = os.path.expanduser("~") + "/mouse_detection_logs.csv"
        self.init_log_file()

    def init_log_file(self):
        """Initialize the log file with headers if it doesn't exist."""
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "Timestamp",
                    "Mouse Detected",
                    "Number of Points",
                    "Cluster Count",
                    "Mouse Count",
                    "Min Dimensions (x, y, z)",
                    "Max Dimensions (x, y, z)"
                ])
        else:
            self.get_logger().info(f"Log file already exists: {self.log_file}")

    def log_detection_data(self, mouse_detected, points, mouse_count=0, cluster=None):
        """Log detection data to a CSV file for future analysis."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        num_points = len(points)
        cluster_count = len(np.unique(cluster)) if cluster is not None else 0
        min_dims = np.min(points, axis=0) if len(points) > 0 else [0, 0, 0]
        max_dims = np.max(points, axis=0) if len(points) > 0 else [0, 0, 0]

        # Append data to the log file
        with open(self.log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp,
                mouse_detected,
                num_points,
                cluster_count,
                mouse_count,
                f"({min_dims[0]:.3f}, {min_dims[1]:.3f}, {min_dims[2]:.3f})",
                f"({max_dims[0]:.3f}, {max_dims[1]:.3f}, {max_dims[2]:.3f})"
            ])

    def on_mqtt_connect(self, client, userdata, flags, reason_code, properties):
        """Callback for MQTT connection."""
        if reason_code == 0:
            self.get_logger().info("Connected to MQTT broker")
            client.subscribe(self.mqtt_topic)
        else:
            self.get_logger().error(f"Failed to connect to MQTT broker: {reason_code}")

    def on_mqtt_message(self, client, userdata, msg):
        """Callback for MQTT message reception."""
        try:
            data = json.loads(msg.payload.decode())
            chunk_msg = self.deserialize_pointcloud(data)
            self.received_chunks.append(chunk_msg)

            # Check if all chunks are received
            if len(self.received_chunks) == data.get("total_chunks", 1):
                self.process_full_pointcloud()
                self.received_chunks = []  # Reset for the next dataset
        except Exception as e:
            self.get_logger().error(f"Failed to process MQTT message: {e}")

    def deserialize_pointcloud(self, data):
        """Deserialize JSON data into a PointCloud2 message."""
        pointcloud_msg = PointCloud2()
        pointcloud_msg.header.frame_id = data["header"]["frame_id"]
        pointcloud_msg.header.stamp.sec = data["header"]["stamp"]["sec"]
        pointcloud_msg.header.stamp.nanosec = data["header"]["stamp"]["nanosec"]
        pointcloud_msg.fields = [
            PointField(
                name=field["name"],
                offset=field["offset"],
                datatype=field["datatype"],
                count=field["count"]
            )
            for field in data["fields"]
        ]
        pointcloud_msg.height = data["height"]
        pointcloud_msg.width = data["width"]
        pointcloud_msg.is_bigendian = data["is_bigendian"]
        pointcloud_msg.point_step = data["point_step"]
        pointcloud_msg.row_step = data["row_step"]
        pointcloud_msg.data = bytes(data["data"])
        pointcloud_msg.is_dense = data["is_dense"]
        return pointcloud_msg

    def process_full_pointcloud(self):
        """Combine all chunks and process the full PointCloud2 data."""
        try:
            combined_data = b"".join([chunk.data for chunk in self.received_chunks])
            combined_msg = self.received_chunks[0]  # Use the first chunk as a template
            combined_msg.data = combined_data
            combined_msg.width = len(combined_data) // combined_msg.point_step

            # Process the combined PointCloud2 message in a separate thread
            processing_thread = threading.Thread(target=self.process_pointcloud, args=(combined_msg,))
            processing_thread.start()
        except Exception as e:
            self.get_logger().error(f"Failed to process full point cloud: {e}")

    def process_pointcloud(self, msg):
        """Process the PointCloud2 message to detect mice."""
        try:
            points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.array([(point['x'], point['y'], point['z']) for point in points], dtype=np.float64)
            points = self.remove_noise(points)

            mouse_count = self.detect_mouse(points)  # Get the number of mice detected

            if mouse_count > 0:
                self.get_logger().info(f'{mouse_count} mice detected!')
                self.trigger_alarm(mouse_count)  # Pass the number of mice to the alarm
                self.log_detection_data(True, points, mouse_count=mouse_count)  # Log the number of mice
            else:
                self.get_logger().info('No mice detected.')
                self.log_detection_data(False, points, mouse_count=0)  # Log 0 mice
        except Exception as e:
            self.get_logger().error(f"Failed to process point cloud: {e}")

    def remove_noise(self, points):
        """Remove noise from the point cloud using KDTree."""
        if len(points) == 0:
            return points
        tree = KDTree(points)
        indices = tree.query_ball_point(points, r=self.noise_threshold)
        filtered_points = [points[i] for i in range(len(points)) if len(indices[i]) > 1]
        return np.array(filtered_points)

    def detect_mouse(self, points):
        """Detect mice using DBSCAN clustering and return the number of mice detected."""
        if len(points) == 0:
            return 0  # Return 0 if no points are available

        dbscan = DBSCAN(eps=self.mouse_radius, min_samples=self.mouse_threshold)
        labels = dbscan.fit_predict(points)
        unique_labels, counts = np.unique(labels, return_counts=True)

        mouse_count = 0
        for label, count in zip(unique_labels, counts):
            if label == -1:
                continue  # Skip noise points
            cluster = points[labels == label]
            if self.is_mouse(cluster):
                mouse_count += 1  # Increment mouse count if the cluster matches a mouse

        return mouse_count

    def is_mouse(self, cluster):
        """Check if a cluster matches the size of a mouse."""
        min_coords = np.min(cluster, axis=0)
        max_coords = np.max(cluster, axis=0)
        dimensions = max_coords - min_coords
        min_size = np.array([0.02, 0.02, 0.02])  # Minimum size of a mouse
        max_size = np.array([0.15, 0.15, 0.15])  # Maximum size of a mouse
        return np.all(dimensions >= min_size) and np.all(dimensions <= max_size)

    def trigger_alarm(self, mouse_count):
        """Trigger an alarm when mice are detected."""
        self.get_logger().info(f'Triggering alarm for {mouse_count} mice...')

        # Log the alarm event
        self.log_alarm(mouse_count)

        # Play a sound alert in a separate thread
        sound_file = self.get_sound_file_path("alarm.wav")
        if os.path.exists(sound_file):
            sound_thread = threading.Thread(target=self.play_alarm_sound, args=(sound_file,))
            sound_thread.start()
        else:
            self.get_logger().error(f"Sound file not found: {sound_file}")

        # Send a desktop notification with the number of mice
        self.send_desktop_notification(mouse_count)

    def log_alarm(self, mouse_count):
        """Log the alarm event with additional details."""
        self.get_logger().warning(f"ALARM: {mouse_count} mice detected in the area!")
        self.get_logger().info("Timestamp: {}".format(self.get_clock().now().to_msg()))

    def get_sound_file_path(self, filename):
        """Get the full path to the sound file."""
        home_path = os.path.expanduser("~")
        return os.path.join(home_path, "Sounds", filename)

    def play_alarm_sound(self, sound_file):
        """Play a sound alert using the 'playsound' library."""
        try:
            playsound(sound_file)
            self.get_logger().info(f"Played alarm sound: {sound_file}")
        except Exception as e:
            self.get_logger().error(f"Failed to play alarm sound: {e}")

    def send_desktop_notification(self, mouse_count):
        """Send a desktop notification using the 'plyer' library."""
        try:
            notification.notify(
                title=f"{mouse_count} Mice Detected!",
                message=f"{mouse_count} mice have been detected in the area. Please check immediately!",
                app_name="DetectionNode",
                timeout=10  # Notification will disappear after 10 seconds
            )
            self.get_logger().info("Desktop notification sent.")
        except Exception as e:
            self.get_logger().error(f"Failed to send desktop notification: {e}")

    def destroy_node(self):
        """Clean up resources on node destruction."""
        self.mqtt_client.loop_stop()
        self.mqtt_client.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()