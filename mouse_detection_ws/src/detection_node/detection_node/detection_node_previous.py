import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
from sklearn.cluster import DBSCAN  # Add this import at the top of your file


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.subscription = self.create_subscription(
            PointCloud2, 'lidar_data', self.listener_callback, 10)

        # Parameters for mouse detection
        self.mouse_radius = 0.1  # Radius to detect a mouse (in meters)
        self.mouse_threshold = 10  # Minimum number of points to consider as a mouse

    def listener_callback(self, msg):
        # Convert PointCloud2 message to numpy array
        points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(points))

        # Extract x, y, z coordinates from the structured array
        xyz = np.array([(point['x'], point['y'], point['z']) for point in points])

        # Detect mice
        if self.detect_mouse(xyz):
            self.get_logger().info('Mouse detected!')
            self.trigger_alarm()
        else:
            self.get_logger().info('No mouse detected.')
            
    def detect_mouse(self, points):
        print("____________________")
        print(points)
        print("____________________")

        # Check if there are enough points
        if len(points) == 0:
            return False

        # Use DBSCAN to cluster the points
        dbscan = DBSCAN(eps=self.mouse_radius, min_samples=self.mouse_threshold)
        labels = dbscan.fit_predict(points)

        # Check if any cluster has enough points
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label != -1 and count >= self.mouse_threshold:  # -1 represents noise in DBSCAN
                return True

        return False

    def trigger_alarm(self):
        # Implement alarm functionality (e.g., sound, light, or notification)
        self.get_logger().info('Triggering alarm...')
        # Example: Play a sound (requires a sound-playing library)
        # os.system('aplay /path/to/alarm_sound.wav')

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()