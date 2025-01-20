import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree


class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.subscription = self.create_subscription(
            PointCloud2, 'lidar_data', self.listener_callback, 10)

        # Parameters for mouse detection
        self.mouse_radius = 0.1  # Radius to detect a mouse (in meters)
        self.mouse_threshold = 10  # Minimum number of points to consider as a mouse
        self.noise_threshold = 0.02  # Threshold for noise removal (in meters)

    def listener_callback(self, msg):
        # Convert PointCloud2 message to numpy array
        points = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(points))

        # Convert structured array to regular float64 array
        points = np.array([(point['x'], point['y'], point['z']) for point in points], dtype=np.float64)

        # Preprocess the point cloud
        points = self.remove_noise(points)

        # Detect mice
        if self.detect_mouse(points):
            self.get_logger().info('Mouse detected!')
            self.trigger_alarm()
        else:
            self.get_logger().info('No mouse detected.')

    def remove_noise(self, points):
        """
        Remove noise from the point cloud using a noise threshold.
        """
        if len(points) == 0:
            return points

        # Use KDTree to find neighbors and remove isolated points
        tree = KDTree(points)
        indices = tree.query_ball_point(points, r=self.noise_threshold)
        filtered_points = [points[i] for i in range(len(points)) if len(indices[i]) > 1]
        return np.array(filtered_points)

    def detect_mouse(self, points):
        """
        Detect mice in the point cloud using clustering and feature extraction.
        """
        if len(points) == 0:
            return False

        # Use DBSCAN to cluster the points
        dbscan = DBSCAN(eps=self.mouse_radius, min_samples=self.mouse_threshold)
        labels = dbscan.fit_predict(points)

        # Analyze clusters
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label == -1:  # Skip noise
                continue

            # Extract the cluster
            cluster = points[labels == label]

            # Check if the cluster is a mouse based on size and shape
            if self.is_mouse(cluster):
                return True

        return False

    def is_mouse(self, cluster):
        """
        Determine if a cluster represents a mouse based on size and shape.
        """
        # Calculate the bounding box dimensions
        min_coords = np.min(cluster, axis=0)
        max_coords = np.max(cluster, axis=0)
        dimensions = max_coords - min_coords

        # Define mouse size constraints (in meters)
        min_size = np.array([0.02, 0.02, 0.02])  # Minimum size of a mouse
        max_size = np.array([0.15, 0.15, 0.15])  # Maximum size of a mouse

        # Check if the cluster dimensions are within the mouse size range
        if np.all(dimensions >= min_size) and np.all(dimensions <= max_size):
            return True
        return False

    def trigger_alarm(self):
        """
        Trigger an alarm (e.g., sound, light, or notification).
        """
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