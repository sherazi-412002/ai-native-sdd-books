---
sidebar_position: 2
---

# rclpy Bridging: Python Integration for Autonomous Humanoid Robotics

## Introduction to rclpy

rclpy is the Python client library for ROS 2, providing access to ROS 2 capabilities from Python. This library serves as a critical bridge between the Python-based ecosystem of scientific computing, machine learning, and AI tools and the ROS 2 framework. For humanoid robotics applications, rclpy enables the integration of sophisticated Python-based algorithms such as computer vision, machine learning models, and high-level planning systems into the real-time control framework of ROS 2.

Python's rich ecosystem of libraries for robotics and AI makes rclpy an invaluable tool for humanoid robot development. Libraries such as NumPy, SciPy, OpenCV, TensorFlow, PyTorch, and scikit-learn can be seamlessly integrated with ROS 2 through rclpy, enabling the development of intelligent humanoid behaviors that leverage state-of-the-art AI and machine learning techniques.

## Theoretical Foundation of Python Integration

The integration of Python with ROS 2 through rclpy is built on several key theoretical concepts:

### Language Bridge Architecture

rclpy implements a Python wrapper around the underlying ROS 2 C++ client library (rcl). This architecture provides:

1. **Memory Management**: Automatic garbage collection and memory management through Python's reference counting
2. **Type Conversion**: Automatic conversion between Python and ROS 2 message types
3. **Exception Handling**: Python exception handling integrated with ROS 2 error reporting
4. **Threading Model**: Integration with Python's Global Interpreter Lock (GIL) and ROS 2's threading model

### Performance Considerations

While Python provides ease of development and access to rich libraries, performance considerations are important for real-time robotics applications:

- **Computationally Intensive Tasks**: Should be offloaded to C++ nodes when real-time performance is critical
- **Message Processing**: Python is suitable for message processing that doesn't require hard real-time guarantees
- **Algorithm Development**: Python excels in prototyping and development of complex algorithms that can later be optimized

## Advanced Publisher Implementation

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Header
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Vector3
from builtin_interfaces.msg import Time
import numpy as np
import threading
import time
from collections import deque
import json

class AdvancedHumanoidPublisherNode(Node):
    """
    Advanced publisher node demonstrating complex rclpy usage patterns for humanoid robotics.
    This implementation includes multiple publishers with different QoS profiles,
    message buffering, and real-time performance considerations.
    """

    def __init__(self):
        super().__init__('advanced_humanoid_publisher_node')

        # Declare parameters
        self.declare_parameter('publish_frequency', 100)  # Hz
        self.declare_parameter('robot_name', 'autonomous_humanoid')
        self.declare_parameter('max_buffer_size', 100)
        self.declare_parameter('enable_diagnostics', True)

        # Get parameter values
        self.publish_frequency = self.get_parameter('publish_frequency').value
        self.robot_name = self.get_parameter('robot_name').value
        self.max_buffer_size = self.get_parameter('max_buffer_size').value
        self.enable_diagnostics = self.get_parameter('enable_diagnostics').value

        # Create QoS profiles for different types of data
        # High-frequency sensor data with reliable delivery
        self.qos_sensor = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Control commands requiring best-effort delivery but with low latency
        self.qos_control = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST
        )

        # Diagnostics data that should be kept for late-joining subscribers
        self.qos_diagnostics = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST
        )

        # Publishers for different types of data
        self.joint_state_publisher = self.create_publisher(
            JointState,
            f'/{self.robot_name}/joint_states',
            self.qos_sensor
        )

        self.imu_publisher = self.create_publisher(
            Imu,
            f'/{self.robot_name}/imu/data',
            self.qos_sensor
        )

        self.diagnostic_publisher = self.create_publisher(
            String,
            f'/{self.robot_name}/diagnostics',
            self.qos_diagnostics
        )

        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            f'/{self.robot_name}/cmd_vel',
            self.qos_control
        )

        # Initialize message buffers for performance optimization
        self.joint_state_buffer = deque(maxlen=self.max_buffer_size)
        self.imu_buffer = deque(maxlen=self.max_buffer_size)
        self.diagnostic_buffer = deque(maxlen=self.max_buffer_size)

        # Initialize joint names for humanoid robot
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint',
            'head_yaw_joint', 'head_pitch_joint'
        ]

        # Initialize joint state data
        self.initialize_joint_state()

        # Create timer for publishing loop
        self.publish_timer = self.create_timer(
            1.0 / self.publish_frequency,
            self.publish_loop
        )

        # Performance tracking
        self.publish_times = deque(maxlen=100)
        self.message_counts = {
            'joint_states': 0,
            'imu_data': 0,
            'diagnostics': 0,
            'cmd_vel': 0
        }

        self.get_logger().info(
            f'Advanced Humanoid Publisher Node initialized for {self.robot_name} '
            f'with publish frequency {self.publish_frequency}Hz'
        )

    def initialize_joint_state(self):
        """Initialize joint state with default values."""
        self.joint_positions = [0.0] * len(self.joint_names)
        self.joint_velocities = [0.0] * len(self.joint_names)
        self.joint_efforts = [0.0] * len(self.joint_names)

    def generate_joint_state_data(self):
        """Generate realistic joint state data for humanoid robot."""
        # Simulate joint positions with some realistic movement patterns
        current_time = self.get_clock().now()
        time_sec = current_time.nanoseconds / 1e9

        # Create oscillating patterns for different joint groups
        for i, joint_name in enumerate(self.joint_names):
            # Different oscillation patterns for different joint types
            if 'hip' in joint_name:
                # Hip joints with slower oscillation
                self.joint_positions[i] = 0.1 * np.sin(0.5 * time_sec + i)
                self.joint_velocities[i] = 0.1 * 0.5 * np.cos(0.5 * time_sec + i)
            elif 'knee' in joint_name:
                # Knee joints with medium oscillation
                self.joint_positions[i] = 0.05 * np.sin(1.0 * time_sec + i)
                self.joint_velocities[i] = 0.05 * 1.0 * np.cos(1.0 * time_sec + i)
            elif 'shoulder' in joint_name:
                # Shoulder joints with faster oscillation
                self.joint_positions[i] = 0.2 * np.sin(2.0 * time_sec + i)
                self.joint_velocities[i] = 0.2 * 2.0 * np.cos(2.0 * time_sec + i)
            else:
                # Other joints with random small movements
                self.joint_positions[i] = 0.01 * np.sin(3.0 * time_sec + i)
                self.joint_velocities[i] = 0.01 * 3.0 * np.cos(3.0 * time_sec + i)

            # Effort is proportional to velocity (simplified model)
            self.joint_efforts[i] = 0.1 * self.joint_velocities[i]

    def generate_imu_data(self):
        """Generate realistic IMU data for humanoid robot."""
        current_time = self.get_clock().now()
        time_sec = current_time.nanoseconds / 1e9

        # Simulate IMU data with realistic noise and movement
        imu_msg = Imu()

        # Timestamp
        imu_msg.header.stamp = current_time.to_msg()
        imu_msg.header.frame_id = f'{self.robot_name}_imu_link'

        # Orientation (simplified - just small oscillations around upright)
        imu_msg.orientation.x = 0.01 * np.sin(0.1 * time_sec)
        imu_msg.orientation.y = 0.01 * np.cos(0.1 * time_sec)
        imu_msg.orientation.z = 0.0
        imu_msg.orientation.w = np.sqrt(1 - (
            imu_msg.orientation.x**2 +
            imu_msg.orientation.y**2 +
            imu_msg.orientation.z**2
        ))

        # Angular velocity (body rotation)
        imu_msg.angular_velocity.x = 0.05 * np.cos(0.2 * time_sec)
        imu_msg.angular_velocity.y = 0.05 * np.sin(0.2 * time_sec)
        imu_msg.angular_velocity.z = 0.01 * np.sin(0.5 * time_sec)

        # Linear acceleration (with gravity component)
        imu_msg.linear_acceleration.x = 0.5 * np.sin(0.3 * time_sec)
        imu_msg.linear_acceleration.y = 0.5 * np.cos(0.3 * time_sec)
        imu_msg.linear_acceleration.z = -9.81 + 0.2 * np.sin(0.4 * time_sec)

        return imu_msg

    def generate_diagnostics(self):
        """Generate diagnostic information for the humanoid robot."""
        diag_data = {
            'timestamp': self.get_clock().now().nanoseconds / 1e9,
            'robot_name': self.robot_name,
            'publish_frequency': self.publish_frequency,
            'joint_count': len(self.joint_names),
            'message_counts': dict(self.message_counts),
            'average_publish_time_ms': np.mean(self.publish_times) if self.publish_times else 0.0,
            'buffer_sizes': {
                'joint_state_buffer': len(self.joint_state_buffer),
                'imu_buffer': len(self.imu_buffer),
                'diagnostic_buffer': len(self.diagnostic_buffer)
            }
        }

        diag_msg = String()
        diag_msg.data = json.dumps(diag_data, indent=2)
        return diag_msg

    def publish_loop(self):
        """Main publishing loop executing at specified frequency."""
        start_time = time.time()

        # Generate and publish joint states
        self.generate_joint_state_data()
        joint_msg = self.create_joint_state_message()
        self.joint_state_publisher.publish(joint_msg)
        self.message_counts['joint_states'] += 1

        # Generate and publish IMU data
        imu_msg = self.generate_imu_data()
        self.imu_publisher.publish(imu_msg)
        self.message_counts['imu_data'] += 1

        # Publish diagnostics periodically (every 10th cycle if enabled)
        if self.enable_diagnostics and self.message_counts['joint_states'] % 10 == 0:
            diag_msg = self.generate_diagnostics()
            self.diagnostic_publisher.publish(diag_msg)
            self.message_counts['diagnostics'] += 1

        # Publish command velocity (for demonstration)
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0.1  # Small forward velocity
        cmd_vel_msg.angular.z = 0.01  # Small angular velocity
        self.cmd_vel_publisher.publish(cmd_vel_msg)
        self.message_counts['cmd_vel'] += 1

        # Track performance
        end_time = time.time()
        publish_time_ms = (end_time - start_time) * 1000
        self.publish_times.append(publish_time_ms)

        # Log performance if it exceeds threshold
        if publish_time_ms > 10.0:  # More than 10ms to publish
            self.get_logger().warn(f'Publishing took {publish_time_ms:.2f}ms')

    def create_joint_state_message(self):
        """Create a JointState message with current joint data."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f'{self.robot_name}_base_link'
        msg.name = self.joint_names
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocities
        msg.effort = self.joint_efforts

        return msg

    def get_performance_metrics(self):
        """Get current performance metrics."""
        return {
            'average_publish_time_ms': np.mean(self.publish_times) if self.publish_times else 0.0,
            'publish_frequency_actual': self.publish_frequency,
            'message_counts': dict(self.message_counts)
        }

def main(args=None):
    rclpy.init(args=args)

    node = AdvancedHumanoidPublisherNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Subscriber Implementation

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Time
import numpy as np
import threading
from collections import deque
import time
from scipy import signal
import json

class AdvancedHumanoidSubscriberNode(Node):
    """
    Advanced subscriber node demonstrating complex rclpy usage patterns for humanoid robotics.
    This implementation includes message filtering, real-time processing, and data fusion.
    """

    def __init__(self):
        super().__init__('advanced_humanoid_subscriber_node')

        # Declare parameters
        self.declare_parameter('processing_frequency', 100)  # Hz
        self.declare_parameter('robot_name', 'autonomous_humanoid')
        self.declare_parameter('enable_filtering', True)
        self.declare_parameter('filter_cutoff_freq', 10.0)  # Hz

        # Get parameter values
        self.processing_frequency = self.get_parameter('processing_frequency').value
        self.robot_name = self.get_parameter('robot_name').value
        self.enable_filtering = self.get_parameter('enable_filtering').value
        self.filter_cutoff_freq = self.get_parameter('filter_cutoff_freq').value

        # Create QoS profiles
        self.qos_sensor = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.qos_control = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Subscribers for different types of data
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            f'/{self.robot_name}/joint_states',
            self.joint_state_callback,
            self.qos_sensor
        )

        self.imu_subscriber = self.create_subscription(
            Imu,
            f'/{self.robot_name}/imu/data',
            self.imu_callback,
            self.qos_sensor
        )

        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            f'/{self.robot_name}/cmd_vel',
            self.cmd_vel_callback,
            self.qos_control
        )

        self.diagnostic_subscriber = self.create_subscription(
            String,
            f'/{self.robot_name}/diagnostics',
            self.diagnostic_callback,
            self.qos_sensor
        )

        # Initialize data storage with buffers
        self.joint_state_buffer = deque(maxlen=100)
        self.imu_buffer = deque(maxlen=100)
        self.cmd_vel_buffer = deque(maxlen=50)

        # Initialize filtered data storage
        self.filtered_joint_positions = {}
        self.filtered_joint_velocities = {}
        self.filtered_imu_data = {}

        # Initialize joint names and create filters
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint',
            'head_yaw_joint', 'head_pitch_joint'
        ]

        # Create low-pass filters for each joint
        self.joint_filters = {}
        if self.enable_filtering:
            for joint_name in self.joint_names:
                self.joint_filters[joint_name] = self.create_low_pass_filter()

        # Initialize state variables
        self.current_joint_state = None
        self.current_imu_data = None
        self.current_cmd_vel = None

        # Create timer for processing loop
        self.processing_timer = self.create_timer(
            1.0 / self.processing_frequency,
            self.process_sensor_data
        )

        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.message_counts = {
            'joint_states_received': 0,
            'imu_received': 0,
            'cmd_vel_received': 0,
            'diagnostics_received': 0
        }

        self.get_logger().info(
            f'Advanced Humanoid Subscriber Node initialized for {self.robot_name} '
            f'with processing frequency {self.processing_frequency}Hz'
        )

    def create_low_pass_filter(self):
        """Create a low-pass filter for signal processing."""
        # Create a simple low-pass filter
        # This is a simplified implementation - in practice, you might use more sophisticated filters
        sos = signal.butter(4, self.filter_cutoff_freq, 'low', fs=self.processing_frequency, output='sos')
        return sos

    def joint_state_callback(self, msg):
        """Callback for joint state messages."""
        self.message_counts['joint_states_received'] += 1

        # Store in buffer for real-time processing
        self.joint_state_buffer.append({
            'timestamp': msg.header.stamp,
            'joint_names': msg.name,
            'positions': list(msg.position),
            'velocities': list(msg.velocity),
            'efforts': list(msg.effort)
        })

        # Update current state
        self.current_joint_state = msg

        # Log if message rate is too high/low
        if self.message_counts['joint_states_received'] % 1000 == 0:
            self.get_logger().info(
                f'Received {self.message_counts["joint_states_received"]} joint state messages'
            )

    def imu_callback(self, msg):
        """Callback for IMU messages."""
        self.message_counts['imu_received'] += 1

        # Store in buffer
        self.imu_buffer.append({
            'timestamp': msg.header.stamp,
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        })

        # Update current state
        self.current_imu_data = msg

    def cmd_vel_callback(self, msg):
        """Callback for velocity command messages."""
        self.message_counts['cmd_vel_received'] += 1

        # Store in buffer
        self.cmd_vel_buffer.append({
            'timestamp': self.get_clock().now(),
            'linear': [msg.linear.x, msg.linear.y, msg.linear.z],
            'angular': [msg.angular.x, msg.angular.y, msg.angular.z]
        })

        # Update current command
        self.current_cmd_vel = msg

    def diagnostic_callback(self, msg):
        """Callback for diagnostic messages."""
        self.message_counts['diagnostics_received'] += 1

        try:
            # Parse diagnostic data
            diag_data = json.loads(msg.data)
            self.get_logger().debug(f'Diagnostic data: {diag_data}')
        except json.JSONDecodeError:
            self.get_logger().warn('Failed to parse diagnostic message')

    def process_sensor_data(self):
        """Process sensor data in real-time."""
        start_time = time.time()

        # Process joint state data
        if self.current_joint_state:
            self.process_joint_data()

        # Process IMU data
        if self.current_imu_data:
            self.process_imu_data()

        # Process command velocity
        if self.current_cmd_vel:
            self.process_cmd_vel_data()

        # Calculate processing time
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        self.processing_times.append(processing_time_ms)

        # Log warning if processing takes too long
        if processing_time_ms > 10.0:  # More than 10ms
            self.get_logger().warn(f'Processing took {processing_time_ms:.2f}ms')

    def process_joint_data(self):
        """Process joint state data with filtering and analysis."""
        if not self.current_joint_state:
            return

        # Apply filtering if enabled
        if self.enable_filtering:
            filtered_positions = []
            for i, joint_name in enumerate(self.current_joint_state.name):
                if joint_name in self.joint_filters and i < len(self.current_joint_state.position):
                    # Apply filter (simplified - in practice, maintain filter state)
                    pos = self.current_joint_state.position[i]
                    filtered_pos = self.apply_filter(self.joint_filters[joint_name], pos)
                    filtered_positions.append(filtered_pos)
                else:
                    filtered_positions.append(self.current_joint_state.position[i])

            # Store filtered data
            for i, joint_name in enumerate(self.current_joint_state.name):
                if i < len(filtered_positions):
                    self.filtered_joint_positions[joint_name] = filtered_positions[i]

        # Perform joint analysis
        self.analyze_joint_positions()
        self.analyze_joint_velocities()
        self.analyze_joint_efforts()

    def apply_filter(self, filter_coeff, value):
        """Apply a simple filter to a value."""
        # This is a simplified implementation
        # In practice, you would maintain the filter's internal state
        return value  # Return unfiltered for now

    def analyze_joint_positions(self):
        """Analyze joint position data for humanoid robot."""
        if not self.current_joint_state:
            return

        # Check for joint limits
        for i, joint_name in enumerate(self.current_joint_state.name):
            if i < len(self.current_joint_state.position):
                pos = self.current_joint_state.position[i]

                # Define joint limits (simplified)
                if 'hip' in joint_name:
                    limit_min, limit_max = -1.57, 1.57  # ±90 degrees
                elif 'knee' in joint_name:
                    limit_min, limit_max = 0.0, 2.36  # 0 to 135 degrees
                elif 'shoulder' in joint_name:
                    limit_min, limit_max = -2.0, 2.0  # ±115 degrees
                else:
                    limit_min, limit_max = -3.14, 3.14  # ±180 degrees

                if pos < limit_min or pos > limit_max:
                    self.get_logger().warn(f'Joint {joint_name} position {pos} exceeds limits [{limit_min}, {limit_max}]')

    def analyze_joint_velocities(self):
        """Analyze joint velocity data."""
        if not self.current_joint_state:
            return

        for i, joint_name in enumerate(self.current_joint_state.name):
            if i < len(self.current_joint_state.velocity):
                vel = self.current_joint_state.velocity[i]
                # Check for excessive velocities
                if abs(vel) > 10.0:  # 10 rad/s threshold
                    self.get_logger().warn(f'Joint {joint_name} velocity {vel} is excessive')

    def analyze_joint_efforts(self):
        """Analyze joint effort data."""
        if not self.current_joint_state:
            return

        for i, joint_name in enumerate(self.current_joint_state.name):
            if i < len(self.current_joint_state.effort):
                effort = self.current_joint_state.effort[i]
                # Check for excessive efforts
                if abs(effort) > 100.0:  # 100 Nm threshold
                    self.get_logger().warn(f'Joint {joint_name} effort {effort} is excessive')

    def process_imu_data(self):
        """Process IMU data for balance and orientation analysis."""
        if not self.current_imu_data:
            return

        # Extract orientation data
        orientation = self.current_imu_data.orientation
        angular_velocity = self.current_imu_data.angular_velocity
        linear_acceleration = self.current_imu_data.linear_acceleration

        # Calculate roll, pitch, yaw from quaternion
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )

        # Check for balance issues
        if abs(pitch) > 0.5 or abs(roll) > 0.5:  # 28.6 degrees threshold
            self.get_logger().warn(f'Potential balance issue: pitch={pitch:.3f}, roll={roll:.3f}')

        # Store filtered IMU data
        self.filtered_imu_data = {
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
            'angular_velocity': [angular_velocity.x, angular_velocity.y, angular_velocity.z],
            'linear_acceleration': [linear_acceleration.x, linear_acceleration.y, linear_acceleration.z]
        }

    def process_cmd_vel_data(self):
        """Process velocity command data."""
        if not self.current_cmd_vel:
            return

        # Check for excessive velocity commands
        linear_speed = np.sqrt(
            self.current_cmd_vel.linear.x**2 +
            self.current_cmd_vel.linear.y**2 +
            self.current_cmd_vel.linear.z**2
        )

        angular_speed = np.sqrt(
            self.current_cmd_vel.angular.x**2 +
            self.current_cmd_vel.angular.y**2 +
            self.current_cmd_vel.angular.z**2
        )

        max_linear = 2.0  # 2 m/s
        max_angular = 1.0  # 1 rad/s

        if linear_speed > max_linear:
            self.get_logger().warn(f'Linear velocity command {linear_speed:.3f} exceeds limit {max_linear}')

        if angular_speed > max_angular:
            self.get_logger().warn(f'Angular velocity command {angular_speed:.3f} exceeds limit {max_angular}')

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles (roll, pitch, yaw)."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def get_current_state_summary(self):
        """Get a summary of the current robot state."""
        summary = {
            'joint_state_available': self.current_joint_state is not None,
            'imu_data_available': self.current_imu_data is not None,
            'cmd_vel_available': self.current_cmd_vel is not None,
            'message_counts': dict(self.message_counts),
            'average_processing_time_ms': np.mean(self.processing_times) if self.processing_times else 0.0,
            'joint_count': len(self.joint_names) if self.current_joint_state else 0
        }

        return summary

def main(args=None):
    rclpy.init(args=args)

    node = AdvancedHumanoidSubscriberNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Service and Client Implementation for Humanoid Robotics

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from example_interfaces.srv import Trigger, SetBool
from std_srvs.srv import Empty
from builtin_interfaces.msg import Time
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float64MultiArray
import time
import threading
from collections import deque
import numpy as np
import json
from typing import Dict, List, Optional, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

class HumanoidServiceNode(Node):
    """
    Advanced service node implementing real-time safe services for humanoid robotics.
    Includes sophisticated state management, safety monitoring, and performance tracking.
    """

    def __init__(self):
        super().__init__('advanced_humanoid_service_node')

        # Declare parameters for service configuration
        self.declare_parameter('service_thread_pool_size', 4)
        self.declare_parameter('service_timeout_sec', 5.0)
        self.declare_parameter('enable_service_monitoring', True)
        self.declare_parameter('max_concurrent_requests', 10)

        # Get parameter values
        self.service_thread_pool_size = self.get_parameter('service_thread_pool_size').value
        self.service_timeout_sec = self.get_parameter('service_timeout_sec').value
        self.enable_service_monitoring = self.get_parameter('enable_service_monitoring').value
        self.max_concurrent_requests = self.get_parameter('max_concurrent_requests').value

        # Initialize thread pool for service execution
        self.service_executor = ThreadPoolExecutor(max_workers=self.service_thread_pool_size)

        # Create services with appropriate QoS profiles
        self.qos_services = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Core humanoid services
        self.calibrate_service = self.create_service(
            Trigger,
            'autonomous_humanoid/calibrate',
            self.calibrate_callback,
            qos_profile=self.qos_services
        )

        self.emergency_stop_service = self.create_service(
            Trigger,
            'autonomous_humanoid/emergency_stop',
            self.emergency_stop_callback,
            qos_profile=self.qos_services
        )

        self.reset_position_service = self.create_service(
            Empty,
            'autonomous_humanoid/reset_position',
            self.reset_position_callback,
            qos_profile=self.qos_services
        )

        self.enable_motors_service = self.create_service(
            SetBool,
            'autonomous_humanoid/enable_motors',
            self.enable_motors_callback,
            qos_profile=self.qos_services
        )

        # Advanced services for humanoid control
        self.set_joint_positions_service = self.create_service(
            Float64MultiArray,
            'autonomous_humanoid/set_joint_positions',
            self.set_joint_positions_callback,
            qos_profile=self.qos_services
        )

        self.get_robot_state_service = self.create_service(
            Trigger,
            'autonomous_humanoid/get_robot_state',
            self.get_robot_state_callback,
            qos_profile=self.qos_services
        )

        # Initialize service state and monitoring
        self.motors_enabled = False
        self.calibration_state = False
        self.emergency_stop_active = False
        self.balance_state = {'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0}
        self.joint_states = {}

        # Service statistics and monitoring
        self.service_stats = {
            'calibrate_calls': 0,
            'emergency_stop_calls': 0,
            'reset_position_calls': 0,
            'enable_motors_calls': 0,
            'set_joint_positions_calls': 0,
            'get_robot_state_calls': 0,
            'total_service_time': 0.0,
            'service_errors': 0
        }

        # Initialize subscribers for state monitoring
        self.joint_state_sub = self.create_subscription(
            JointState,
            'autonomous_humanoid/joint_states',
            self.joint_state_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            'autonomous_humanoid/imu/data',
            self.imu_callback,
            10
        )

        self.get_logger().info(
            f'Advanced Humanoid Service Node initialized with {self.service_thread_pool_size} service threads'
        )

    def joint_state_callback(self, msg: JointState):
        """Update internal joint state from sensor data."""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_states[name] = {
                    'position': msg.position[i],
                    'velocity': msg.velocity[i] if i < len(msg.velocity) else 0.0,
                    'effort': msg.effort[i] if i < len(msg.effort) else 0.0,
                    'timestamp': time.time()
                }

    def imu_callback(self, msg: Imu):
        """Update internal balance state from IMU data."""
        # Convert quaternion to Euler angles (simplified)
        w, x, y, z = msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z

        # Calculate roll, pitch, yaw (simplified approximation)
        self.balance_state['roll'] = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        self.balance_state['pitch'] = np.arcsin(2.0 * (w * y - z * x))
        self.balance_state['yaw'] = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def calibrate_callback(self, request, response):
        """Calibrate all humanoid joints with safety checks."""
        start_time = time.time()
        self.service_stats['calibrate_calls'] += 1

        try:
            self.get_logger().info('Starting calibration procedure with safety checks')

            # Safety checks before calibration
            if self.emergency_stop_active:
                raise Exception('Cannot calibrate while emergency stop is active')

            if abs(self.balance_state['pitch']) > 0.5 or abs(self.balance_state['roll']) > 0.5:
                raise Exception('Robot not in safe balance state for calibration')

            # Activate safety mode during calibration
            self.emergency_stop_active = True
            self.motors_enabled = False

            # Perform calibration steps
            self.get_logger().info('Calibrating joint positions...')
            self._calibrate_joints()

            self.get_logger().info('Calibrating sensors...')
            self._calibrate_sensors()

            # Update calibration state
            self.calibration_state = True

            # Restore normal operation
            self.emergency_stop_active = False
            self.motors_enabled = True

            response.success = True
            response.message = 'Calibration completed successfully'

            self.get_logger().info('Calibration completed successfully')

        except Exception as e:
            response.success = False
            response.message = f'Calibration failed: {str(e)}'
            self.get_logger().error(f'Calibration error: {str(e)}')
            self.service_stats['service_errors'] += 1

        end_time = time.time()
        self.service_stats['total_service_time'] += (end_time - start_time)
        return response

    def _calibrate_joints(self):
        """Internal method to calibrate all joints."""
        # In a real implementation, this would interface with motor controllers
        # to move each joint to its calibration position
        for joint_name, joint_data in self.joint_states.items():
            # Simulate calibration movement
            time.sleep(0.05)  # Simulate movement time
            self.get_logger().debug(f'Calibrated joint: {joint_name}')

    def _calibrate_sensors(self):
        """Internal method to calibrate sensors."""
        # In a real implementation, this would perform sensor calibration procedures
        time.sleep(0.1)  # Simulate calibration time
        self.get_logger().debug('Sensors calibrated')

    def emergency_stop_callback(self, request, response):
        """Activate emergency stop with safety monitoring."""
        start_time = time.time()
        self.service_stats['emergency_stop_calls'] += 1
        self.get_logger().warn('EMERGENCY STOP ACTIVATED')

        try:
            # Stop all motor movements immediately
            self.emergency_stop_active = True
            self.motors_enabled = False

            # Log the emergency stop with state information
            self.get_logger().info(
                f'Emergency stop activated. Balance state: pitch={self.balance_state["pitch"]:.3f}, '
                f'roll={self.balance_state["roll"]:.3f}'
            )

            response.success = True
            response.message = 'Emergency stop activated'

        except Exception as e:
            response.success = False
            response.message = f'Emergency stop failed: {str(e)}'
            self.get_logger().error(f'Emergency stop error: {str(e)}')
            self.service_stats['service_errors'] += 1

        end_time = time.time()
        self.service_stats['total_service_time'] += (end_time - start_time)
        return response

    def reset_position_callback(self, request, response):
        """Reset humanoid to default position with safety checks."""
        start_time = time.time()
        self.service_stats['reset_position_calls'] += 1
        self.get_logger().info('Resetting humanoid to default position')

        try:
            # Safety checks
            if not self.motors_enabled:
                raise Exception('Motors not enabled - cannot reset position')

            if self.emergency_stop_active:
                raise Exception('Emergency stop active - cannot reset position')

            # Calculate safe reset position based on current balance
            if abs(self.balance_state['pitch']) > 0.5 or abs(self.balance_state['roll']) > 0.5:
                raise Exception('Robot is in unstable position - cannot safely reset')

            # Move to default position
            default_positions = self._calculate_safe_default_position()
            self._move_to_position(default_positions)

            response.success = True
            response.message = 'Position reset completed safely'

            self.get_logger().info('Position reset completed successfully')

        except Exception as e:
            response.success = False
            response.message = f'Position reset failed: {str(e)}'
            self.get_logger().error(f'Position reset error: {str(e)}')
            self.service_stats['service_errors'] += 1

        end_time = time.time()
        self.service_stats['total_service_time'] += (end_time - start_time)
        return response

    def _calculate_safe_default_position(self) -> List[float]:
        """Calculate a safe default position based on current robot state."""
        # This would implement a safe standing position based on the robot's kinematic structure
        # For now, return a simple default position
        default_positions = []
        for joint_name in sorted(self.joint_states.keys()):
            # Calculate safe position based on joint type and current state
            default_positions.append(0.0)  # Simplified default

        return default_positions

    def _move_to_position(self, positions: List[float]):
        """Internal method to move joints to specified positions."""
        # In a real implementation, this would interface with the motion controller
        # For simulation, just sleep for the movement time
        movement_time = 1.0  # seconds
        time.sleep(movement_time)
        self.get_logger().debug(f'Moved to position with {len(positions)} joints')

    def enable_motors_callback(self, request, response):
        """Enable or disable motors with comprehensive safety checks."""
        start_time = time.time()
        self.service_stats['enable_motors_calls'] += 1
        enable = request.data
        self.get_logger().info(f'{"Enabling" if enable else "Disabling"} motors')

        try:
            if enable:
                # Comprehensive safety checks before enabling
                if self.emergency_stop_active:
                    raise Exception('Cannot enable motors while emergency stop is active')

                if abs(self.balance_state['pitch']) > 0.7 or abs(self.balance_state['roll']) > 0.7:
                    raise Exception('Robot not in safe balance state for motor enable')

                # Check for joint limit violations
                if self._has_joint_limit_violations():
                    raise Exception('Joint limit violations detected - cannot enable motors')

                # Enable motors
                self.motors_enabled = True
                response.success = True
                response.message = 'Motors enabled successfully'

                self.get_logger().info('Motors enabled safely')
            else:
                # Disable motors
                self.motors_enabled = False
                response.success = True
                response.message = 'Motors disabled successfully'

                self.get_logger().info('Motors disabled')

        except Exception as e:
            response.success = False
            response.message = f'Motor control failed: {str(e)}'
            self.get_logger().error(f'Motor control error: {str(e)}')
            self.service_stats['service_errors'] += 1

        end_time = time.time()
        self.service_stats['total_service_time'] += (end_time - start_time)
        return response

    def _has_joint_limit_violations(self) -> bool:
        """Check for joint limit violations."""
        # In a real implementation, this would check joint positions against limits
        return False  # Simplified for now

    def set_joint_positions_callback(self, request, response):
        """Set joint positions with trajectory planning and safety checks."""
        start_time = time.time()
        self.service_stats['set_joint_positions_calls'] += 1

        try:
            target_positions = list(request.data)

            # Safety checks
            if not self.motors_enabled:
                raise Exception('Motors not enabled - cannot set joint positions')

            if self.emergency_stop_active:
                raise Exception('Emergency stop active - cannot set joint positions')

            # Validate joint position limits
            if not self._validate_joint_positions(target_positions):
                raise Exception('Target positions exceed joint limits')

            # Plan smooth trajectory
            trajectory = self._plan_smooth_trajectory(target_positions)

            # Execute trajectory
            self._execute_trajectory(trajectory)

            response.success = True
            response.message = f'Set {len(target_positions)} joint positions successfully'

        except Exception as e:
            response.success = False
            response.message = f'Set joint positions failed: {str(e)}'
            self.get_logger().error(f'Set joint positions error: {str(e)}')
            self.service_stats['service_errors'] += 1

        end_time = time.time()
        self.service_stats['total_service_time'] += (end_time - start_time)
        return response

    def _validate_joint_positions(self, positions: List[float]) -> bool:
        """Validate that joint positions are within limits."""
        # In a real implementation, this would check against actual joint limits
        for pos in positions:
            if abs(pos) > 3.14:  # Basic check for extreme positions
                return False
        return True

    def _plan_smooth_trajectory(self, target_positions: List[float]) -> List[List[float]]:
        """Plan a smooth trajectory to reach target positions."""
        # Simplified trajectory planning - in reality this would use more sophisticated algorithms
        current_positions = [self.joint_states.get(name, {}).get('position', 0.0)
                            for name in sorted(self.joint_states.keys())]

        # Create 10 intermediate points for smooth movement
        trajectory = []
        for i in range(10):
            fraction = i / 9.0
            intermediate_positions = [
                current + fraction * (target - current)
                for current, target in zip(current_positions, target_positions)
            ]
            trajectory.append(intermediate_positions)

        return trajectory

    def _execute_trajectory(self, trajectory: List[List[float]]):
        """Execute a planned trajectory."""
        for positions in trajectory:
            # In a real implementation, this would send commands to motor controllers
            time.sleep(0.05)  # Simulate execution time

    def get_robot_state_callback(self, request, response):
        """Get comprehensive robot state information."""
        start_time = time.time()
        self.service_stats['get_robot_state_calls'] += 1

        try:
            # Compile comprehensive robot state
            robot_state = {
                'motors_enabled': self.motors_enabled,
                'calibration_state': self.calibration_state,
                'emergency_stop_active': self.emergency_stop_active,
                'balance_state': self.balance_state.copy(),
                'joint_states': {name: data.copy() for name, data in self.joint_states.items()},
                'timestamp': time.time(),
                'service_stats': self.service_stats.copy()
            }

            # Encode as JSON string in the response
            response.success = True
            response.message = json.dumps(robot_state, indent=2)

        except Exception as e:
            response.success = False
            response.message = f'Get robot state failed: {str(e)}'
            self.get_logger().error(f'Get robot state error: {str(e)}')
            self.service_stats['service_errors'] += 1

        end_time = time.time()
        self.service_stats['total_service_time'] += (end_time - start_time)
        return response

    def get_service_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics."""
        total_calls = sum([
            self.service_stats['calibrate_calls'],
            self.service_stats['emergency_stop_calls'],
            self.service_stats['reset_position_calls'],
            self.service_stats['enable_motors_calls'],
            self.service_stats['set_joint_positions_calls'],
            self.service_stats['get_robot_state_calls']
        ])

        avg_time = self.service_stats['total_service_time'] / total_calls if total_calls > 0 else 0

        return {
            'calibrate_calls': self.service_stats['calibrate_calls'],
            'emergency_stop_calls': self.service_stats['emergency_stop_calls'],
            'reset_position_calls': self.service_stats['reset_position_calls'],
            'enable_motors_calls': self.service_stats['enable_motors_calls'],
            'set_joint_positions_calls': self.service_stats['set_joint_positions_calls'],
            'get_robot_state_calls': self.service_stats['get_robot_state_calls'],
            'total_service_calls': total_calls,
            'total_errors': self.service_stats['service_errors'],
            'average_response_time_ms': avg_time * 1000,
            'error_rate_percent': (self.service_stats['service_errors'] / total_calls * 100) if total_calls > 0 else 0
        }

class AdvancedHumanoidClientNode(Node):
    """
    Advanced client node demonstrating sophisticated service interaction patterns for humanoid robotics.
    """

    def __init__(self):
        super().__init__('advanced_humanoid_client_node')

        # Create clients for all humanoid services
        self.clients = {
            'calibrate': self.create_client(Trigger, 'autonomous_humanoid/calibrate'),
            'emergency_stop': self.create_client(Trigger, 'autonomous_humanoid/emergency_stop'),
            'reset_position': self.create_client(Empty, 'autonomous_humanoid/reset_position'),
            'enable_motors': self.create_client(SetBool, 'autonomous_humanoid/enable_motors'),
            'set_joint_positions': self.create_client(Float64MultiArray, 'autonomous_humanoid/set_joint_positions'),
            'get_robot_state': self.create_client(Trigger, 'autonomous_humanoid/get_robot_state')
        }

        # Wait for all services to be available
        self.get_logger().info('Waiting for all humanoid services to become available...')
        for service_name, client in self.clients.items():
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'{service_name} service not available, waiting...')

        self.get_logger().info('All humanoid services are available')

        # Create timer for periodic service calls
        self.service_test_timer = self.create_timer(10.0, self.run_system_tests)

        # Performance monitoring
        self.request_times = deque(maxlen=100)
        self.error_count = 0

        self.get_logger().info('Advanced Humanoid Client Node initialized and connected to all services')

    async def call_service_async(self, service_name: str, request, timeout_sec: float = 5.0) -> Optional[Any]:
        """Asynchronously call a service with timeout."""
        client = self.clients[service_name]
        start_time = time.time()

        try:
            future = client.call_async(request)
            result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_sec)

            response_time = time.time() - start_time
            self.request_times.append(response_time)

            self.get_logger().debug(f'{service_name} service call completed in {response_time:.3f}s')
            return result

        except asyncio.TimeoutError:
            self.get_logger().error(f'{service_name} service call timed out after {timeout_sec}s')
            self.error_count += 1
            return None
        except Exception as e:
            self.get_logger().error(f'{service_name} service call failed: {str(e)}')
            self.error_count += 1
            return None

    def run_system_tests(self):
        """Run comprehensive system tests calling various services."""
        self.get_logger().info('Running humanoid system tests...')

        # Test 1: Get current robot state
        self.test_get_robot_state()

        # Test 2: Enable motors (if not already enabled)
        self.test_enable_motors()

        # Test 3: Reset position (if in safe state)
        self.test_reset_position()

        # Test 4: Set some joint positions
        self.test_set_joint_positions()

        # Test 5: Get robot state again to verify changes
        self.test_get_robot_state()

        # Log performance statistics
        self.log_performance_stats()

    def test_get_robot_state(self):
        """Test getting robot state."""
        request = Trigger.Request()
        future = self.clients['get_robot_state'].call_async(request)

        # Handle response in a callback to avoid blocking
        future.add_done_callback(self.handle_robot_state_response)

    def handle_robot_state_response(self, future):
        """Handle robot state response."""
        try:
            response = future.result()
            if response.success:
                state_data = json.loads(response.message)
                self.get_logger().info(f'Robot state: Motors enabled={state_data["motors_enabled"]}, '
                                     f'Balance pitch={state_data["balance_state"]["pitch"]:.3f}')
            else:
                self.get_logger().error(f'Get robot state failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Error processing robot state response: {str(e)}')

    def test_enable_motors(self):
        """Test enabling motors."""
        request = SetBool.Request()
        request.data = True
        future = self.clients['enable_motors'].call_async(request)
        future.add_done_callback(lambda f: self.handle_enable_motors_response(f, "enable"))

    def test_reset_position(self):
        """Test resetting position."""
        # Only reset if robot is in a safe state
        request = Empty.Request()
        future = self.clients['reset_position'].call_async(request)
        future.add_done_callback(lambda f: self.handle_simple_response(f, "reset position"))

    def test_set_joint_positions(self):
        """Test setting joint positions."""
        # Send a simple joint position command
        request = Float64MultiArray.Request()
        # Send zeros as a safe default (would be actual positions in real use)
        request.data = [0.0] * 12  # 12 joints
        future = self.clients['set_joint_positions'].call_async(request)
        future.add_done_callback(lambda f: self.handle_simple_response(f, "set joint positions"))

    def handle_enable_motors_response(self, future, action_name):
        """Handle enable/disable motors response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Motors {action_name}d successfully')
            else:
                self.get_logger().error(f'Failed to {action_name} motors: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Error processing {action_name} response: {str(e)}')

    def handle_simple_response(self, future, action_name):
        """Handle simple service response."""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'{action_name.title()} completed successfully')
            else:
                self.get_logger().error(f'{action_name.title()} failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Error processing {action_name} response: {str(e)}')

    def log_performance_stats(self):
        """Log performance statistics."""
        if self.request_times:
            avg_time = sum(self.request_times) / len(self.request_times)
            max_time = max(self.request_times)
            min_time = min(self.request_times)

            self.get_logger().info(
                f'Service performance - Avg: {avg_time*1000:.1f}ms, '
                f'Max: {max_time*1000:.1f}ms, Min: {min_time*1000:.1f}ms, '
                f'Errors: {self.error_count}'
            )

def main(args=None):
    rclpy.init(args=args)

    # Create both service and client nodes
    service_node = AdvancedHumanoidServiceNode()
    client_node = AdvancedHumanoidClientNode()

    # Create multi-threaded executor to run both nodes
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(service_node)
    executor.add_node(client_node)

    try:
        service_node.get_logger().info('Starting humanoid service and client nodes...')
        executor.spin()
    except KeyboardInterrupt:
        service_node.get_logger().info('Service node interrupted by user')
        client_node.get_logger().info('Client node interrupted by user')
    finally:
        service_node.destroy_node()
        client_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Parameter Management with Dynamic Reconfiguration

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float64, String
from geometry_msgs.msg import Vector3
import json
import time
from typing import List, Dict, Any, Optional
import threading
from dataclasses import dataclass
from enum import Enum

class ControlMode(Enum):
    """Enumeration of control modes for humanoid robot."""
    IDLE = "idle"
    POSITION_CONTROL = "position_control"
    VELOCITY_CONTROL = "velocity_control"
    TORQUE_CONTROL = "torque_control"
    IMPEDANCE_CONTROL = "impedance_control"
    TRAJECTORY_FOLLOWING = "trajectory_following"

@dataclass
class GaitParameters:
    """Data class for walking gait parameters."""
    step_height: float = 0.1
    step_length: float = 0.3
    step_duration: float = 1.0
    stance_phase_ratio: float = 0.6
    swing_phase_ratio: float = 0.4
    max_step_frequency: float = 2.0

@dataclass
class BalanceParameters:
    """Data class for balance control parameters."""
    com_height: float = 0.8
    com_offset_x: float = 0.0
    com_offset_y: float = 0.0
    balance_gain_p: float = 10.0
    balance_gain_i: float = 1.0
    balance_gain_d: float = 0.5
    max_lean_angle: float = 0.3

class AdvancedHumanoidParameterNode(Node):
    """
    Advanced parameter node with dynamic reconfiguration for humanoid robotics.
    Features include parameter validation, inter-parameter dependencies, and runtime updates.
    """

    def __init__(self):
        super().__init__('advanced_humanoid_parameter_node')

        # Declare complex parameters with validators
        self.declare_parameter('control.frequency', 100,
                              ParameterDescriptor(description='Control loop frequency in Hz',
                                                integer_range=[{'from_value': 10, 'to_value': 1000, 'step': 10}]))
        self.declare_parameter('control.max_velocity', 1.0,
                              ParameterDescriptor(description='Maximum joint velocity in rad/s',
                                                double_range=[{'from_value': 0.1, 'to_value': 10.0, 'step': 0.1}]))
        self.declare_parameter('control.max_acceleration', 5.0,
                              ParameterDescriptor(description='Maximum joint acceleration in rad/s^2',
                                                double_range=[{'from_value': 0.5, 'to_value': 50.0, 'step': 0.1}]))
        self.declare_parameter('safety.timeout', 1.0,
                              ParameterDescriptor(description='Safety timeout in seconds',
                                                double_range=[{'from_value': 0.1, 'to_value': 10.0, 'step': 0.1}]))
        self.declare_parameter('balance.threshold', 0.1,
                              ParameterDescriptor(description='Balance threshold in radians',
                                                double_range=[{'from_value': 0.01, 'to_value': 1.0, 'step': 0.01}]))
        self.declare_parameter('robot.name', 'autonomous_humanoid',
                              ParameterDescriptor(description='Name of the robot'))

        # Declare array parameters for joint limits
        self.declare_parameter('joints.limits.position',
                              [1.57, 1.57, 0.78, 1.57, 1.57, 0.78],  # Example limits
                              ParameterDescriptor(description='Joint position limits (rad)'))

        # Declare complex gait parameters
        self.declare_parameter('gait.step_height', 0.1,
                              ParameterDescriptor(description='Walking step height (m)',
                                                double_range=[{'from_value': 0.01, 'to_value': 0.5, 'step': 0.01}]))
        self.declare_parameter('gait.step_length', 0.3,
                              ParameterDescriptor(description='Walking step length (m)',
                                                double_range=[{'from_value': 0.05, 'to_value': 1.0, 'step': 0.01}]))
        self.declare_parameter('gait.step_duration', 1.0,
                              ParameterDescriptor(description='Walking step duration (s)',
                                                double_range=[{'from_value': 0.1, 'to_value': 5.0, 'step': 0.01}]))
        self.declare_parameter('gait.stance_ratio', 0.6,
                              ParameterDescriptor(description='Stance phase ratio',
                                                double_range=[{'from_value': 0.1, 'to_value': 0.9, 'step': 0.01}]))

        # Declare balance control parameters
        self.declare_parameter('balance.com_height', 0.8,
                              ParameterDescriptor(description='Center of mass height (m)',
                                                double_range=[{'from_value': 0.5, 'to_value': 1.5, 'step': 0.01}]))
        self.declare_parameter('balance.gain_p', 10.0,
                              ParameterDescriptor(description='Balance control P gain',
                                                double_range=[{'from_value': 0.1, 'to_value': 100.0, 'step': 0.1}]))
        self.declare_parameter('balance.gain_i', 1.0,
                              ParameterDescriptor(description='Balance control I gain',
                                                double_range=[{'from_value': 0.0, 'to_value': 10.0, 'step': 0.01}]))
        self.declare_parameter('balance.gain_d', 0.5,
                              ParameterDescriptor(description='Balance control D gain',
                                                double_range=[{'from_value': 0.0, 'to_value': 10.0, 'step': 0.01}]))

        # Create publisher for parameter change notifications
        self.param_change_pub = self.create_publisher(
            String,
            'parameter_changes',
            QoSProfile(depth=20, reliability=ReliabilityPolicy.RELIABLE)
        )

        # Set up parameter callback with validation
        self.set_parameters_callback(self.parameters_callback)

        # Initialize parameter values
        self.update_parameter_values()

        # Create timer for parameter validation
        self.param_validation_timer = self.create_timer(2.0, self.validate_parameters)

        # Parameter change history for rollback capability
        self.param_history = deque(maxlen=50)

        # Lock for thread-safe parameter access
        self.param_lock = threading.Lock()

        self.get_logger().info('Advanced Humanoid Parameter Node initialized with validation and monitoring')

    def parameters_callback(self, params):
        """
        Callback for parameter changes with comprehensive validation.
        """
        start_time = time.time()
        successful_params = []
        failed_params = []

        with self.param_lock:
            for param in params:
                validation_result = self.validate_parameter(param)

                if validation_result.is_valid:
                    successful_params.append(param)

                    # Store in history for potential rollback
                    self.param_history.append({
                        'parameter': param.name,
                        'old_value': self.get_parameter(param.name).value if self.has_parameter(param.name) else None,
                        'new_value': param.value,
                        'timestamp': time.time()
                    })
                else:
                    failed_params.append(f'{param.name}: {validation_result.error_message}')

            # Apply successful changes
            if successful_params:
                for param in successful_params:
                    self.get_logger().info(f'Parameter {param.name} changed to {param.value}')

                    # Notify about parameter change
                    self.notify_parameter_change(param)

        # Calculate validation time
        validation_time = time.time() - start_time

        # Log performance if validation took too long
        if validation_time > 0.01:  # 10ms threshold
            self.get_logger().warn(f'Parameter validation took {validation_time*1000:.2f}ms')

        return SetParametersResult(successful=(len(failed_params) == 0))

    def validate_parameter(self, param):
        """Validate a single parameter with context-aware rules."""
        class ValidationResult:
            def __init__(self, is_valid=True, error_message=""):
                self.is_valid = is_valid
                self.error_message = error_message

        # Validate control frequency
        if param.name == 'control.frequency':
            if not isinstance(param.value, int) and not isinstance(param.value, float):
                return ValidationResult(False, "Control frequency must be numeric")
            if param.value < 10 or param.value > 1000:
                return ValidationResult(False, "Control frequency must be between 10-1000 Hz")
            return ValidationResult(True)

        # Validate maximum velocity
        elif param.name == 'control.max_velocity':
            if not isinstance(param.value, (int, float)):
                return ValidationResult(False, "Max velocity must be numeric")
            if param.value <= 0 or param.value > 10.0:
                return ValidationResult(False, "Max velocity must be between 0.1-10.0 rad/s")
            return ValidationResult(True)

        # Validate maximum acceleration
        elif param.name == 'control.max_acceleration':
            if not isinstance(param.value, (int, float)):
                return ValidationResult(False, "Max acceleration must be numeric")
            if param.value <= 0 or param.value > 100.0:
                return ValidationResult(False, "Max acceleration must be between 0.5-100.0 rad/s²")
            return ValidationResult(True)

        # Validate balance threshold
        elif param.name == 'balance.threshold':
            if not isinstance(param.value, (int, float)):
                return ValidationResult(False, "Balance threshold must be numeric")
            if param.value <= 0 or param.value > 1.57:  # Max 90 degrees
                return ValidationResult(False, "Balance threshold must be between 0.01-1.57 rad")
            return ValidationResult(True)

        # Validate gait parameters
        elif param.name.startswith('gait.'):
            if not isinstance(param.value, (int, float)):
                return ValidationResult(False, f"{param.name} must be numeric")

            if param.name == 'gait.step_height':
                if param.value < 0.01 or param.value > 0.5:
                    return ValidationResult(False, "Step height must be between 0.01-0.5m")
            elif param.name == 'gait.step_length':
                if param.value < 0.05 or param.value > 1.0:
                    return ValidationResult(False, "Step length must be between 0.05-1.0m")
            elif param.name == 'gait.step_duration':
                if param.value < 0.1 or param.value > 5.0:
                    return ValidationResult(False, "Step duration must be between 0.1-5.0s")
            elif param.name == 'gait.stance_ratio':
                if param.value < 0.1 or param.value > 0.9:
                    return ValidationResult(False, "Stance ratio must be between 0.1-0.9")

            return ValidationResult(True)

        # Validate balance parameters
        elif param.name.startswith('balance.'):
            if not isinstance(param.value, (int, float)):
                return ValidationResult(False, f"{param.name} must be numeric")

            if param.name == 'balance.com_height':
                if param.value < 0.5 or param.value > 1.5:
                    return ValidationResult(False, "COM height must be between 0.5-1.5m")
            elif param.name == 'balance.gain_p':
                if param.value < 0.1 or param.value > 100.0:
                    return ValidationResult(False, "P gain must be between 0.1-100.0")
            elif param.name == 'balance.gain_i':
                if param.value < 0.0 or param.value > 10.0:
                    return ValidationResult(False, "I gain must be between 0.0-10.0")
            elif param.name == 'balance.gain_d':
                if param.value < 0.0 or param.value > 10.0:
                    return ValidationResult(False, "D gain must be between 0.0-10.0")

            return ValidationResult(True)

        # For other parameters, accept them
        return ValidationResult(True)

    def notify_parameter_change(self, param):
        """Notify about parameter changes via publisher."""
        change_msg = String()
        change_msg.data = json.dumps({
            'parameter': param.name,
            'value': param.value,
            'timestamp': time.time(),
            'node': self.get_name()
        })
        self.param_change_pub.publish(change_msg)

    def update_parameter_values(self):
        """Update internal variables from parameter values."""
        with self.param_lock:
            self.control_frequency = self.get_parameter('control.frequency').value
            self.max_velocity = self.get_parameter('control.max_velocity').value
            self.max_acceleration = self.get_parameter('control.max_acceleration').value
            self.safety_timeout = self.get_parameter('safety.timeout').value
            self.balance_threshold = self.get_parameter('balance.threshold').value
            self.robot_name = self.get_parameter('robot.name').value
            self.joint_limits = self.get_parameter('joints.limits.position').value

            # Update gait parameters
            self.gait_params = GaitParameters(
                step_height=self.get_parameter('gait.step_height').value,
                step_length=self.get_parameter('gait.step_length').value,
                step_duration=self.get_parameter('gait.step_duration').value,
                stance_phase_ratio=self.get_parameter('gait.stance_ratio').value,
                swing_phase_ratio=1.0 - self.get_parameter('gait.stance_ratio').value
            )

            # Update balance parameters
            self.balance_params = BalanceParameters(
                com_height=self.get_parameter('balance.com_height').value,
                balance_gain_p=self.get_parameter('balance.gain_p').value,
                balance_gain_i=self.get_parameter('balance.gain_i').value,
                balance_gain_d=self.get_parameter('balance.gain_d').value
            )

    def validate_parameters(self):
        """Periodically validate parameter consistency and relationships."""
        with self.param_lock:
            # Check for parameter consistency
            if (self.gait_params.step_duration <= 0 or
                self.gait_params.step_length / self.gait_params.step_duration > 2.0):  # Max speed check
                self.get_logger().warn(
                    f'Gait parameters may result in excessive speed: '
                    f'{self.gait_params.step_length/self.gait_params.step_duration:.2f} m/s'
                )

            # Check balance control gains
            total_gain = self.balance_params.balance_gain_p + self.balance_params.balance_gain_i + self.balance_params.balance_gain_d
            if total_gain > 200.0:
                self.get_logger().warn(f'High balance control gains may cause instability: total={total_gain}')

    def get_current_configuration(self) -> Dict[str, Any]:
        """Get current configuration as a comprehensive dictionary."""
        with self.param_lock:
            return {
                'control': {
                    'frequency': self.control_frequency,
                    'max_velocity': self.max_velocity,
                    'max_acceleration': self.max_acceleration,
                    'safety_timeout': self.safety_timeout
                },
                'balance': {
                    'threshold': self.balance_threshold,
                    'parameters': {
                        'com_height': self.balance_params.com_height,
                        'com_offset_x': self.balance_params.com_offset_x,
                        'com_offset_y': self.balance_params.com_offset_y,
                        'gain_p': self.balance_params.balance_gain_p,
                        'gain_i': self.balance_params.balance_gain_i,
                        'gain_d': self.balance_params.balance_gain_d,
                        'max_lean_angle': self.balance_params.max_lean_angle
                    }
                },
                'gait': {
                    'step_height': self.gait_params.step_height,
                    'step_length': self.gait_params.step_length,
                    'step_duration': self.gait_params.step_duration,
                    'stance_phase_ratio': self.gait_params.stance_phase_ratio,
                    'swing_phase_ratio': self.gait_params.swing_phase_ratio,
                    'max_step_frequency': self.gait_params.max_step_frequency
                },
                'robot': {
                    'name': self.robot_name,
                    'joint_limits': self.joint_limits
                },
                'system': {
                    'parameter_count': len(self._parameters),
                    'history_size': len(self.param_history)
                }
            }

    def save_configuration(self, file_path: str):
        """Save current configuration to a file."""
        config = self.get_current_configuration()
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.get_logger().info(f'Configuration saved to {file_path}')

    def load_configuration(self, file_path: str):
        """Load configuration from a file and update parameters."""
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)

            # Update parameters from loaded configuration
            updates = []
            updates.append(Parameter('control.frequency', Parameter.Type.INTEGER, config['control']['frequency']))
            updates.append(Parameter('control.max_velocity', Parameter.Type.DOUBLE, config['control']['max_velocity']))
            updates.append(Parameter('control.max_acceleration', Parameter.Type.DOUBLE, config['control']['max_acceleration']))
            updates.append(Parameter('safety.timeout', Parameter.Type.DOUBLE, config['control']['safety_timeout']))
            updates.append(Parameter('balance.threshold', Parameter.Type.DOUBLE, config['balance']['threshold']))

            # Update gait parameters
            updates.append(Parameter('gait.step_height', Parameter.Type.DOUBLE, config['gait']['step_height']))
            updates.append(Parameter('gait.step_length', Parameter.Type.DOUBLE, config['gait']['step_length']))
            updates.append(Parameter('gait.step_duration', Parameter.Type.DOUBLE, config['gait']['step_duration']))
            updates.append(Parameter('gait.stance_ratio', Parameter.Type.DOUBLE, config['gait']['stance_phase_ratio']))

            # Update balance parameters
            updates.append(Parameter('balance.com_height', Parameter.Type.DOUBLE, config['balance']['parameters']['com_height']))
            updates.append(Parameter('balance.gain_p', Parameter.Type.DOUBLE, config['balance']['parameters']['gain_p']))
            updates.append(Parameter('balance.gain_i', Parameter.Type.DOUBLE, config['balance']['parameters']['gain_i']))
            updates.append(Parameter('balance.gain_d', Parameter.Type.DOUBLE, config['balance']['parameters']['gain_d']))

            # Apply updates
            self.set_parameters(updates)

            self.get_logger().info(f'Configuration loaded from {file_path}')

        except FileNotFoundError:
            self.get_logger().error(f'Configuration file not found: {file_path}')
        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid JSON in configuration file: {file_path}')
        except KeyError as e:
            self.get_logger().error(f'Missing key in configuration file: {str(e)}')

def main(args=None):
    rclpy.init(args=args)

    node = AdvancedHumanoidParameterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Parameter node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration with NVIDIA Isaac Sim for Humanoid Robotics

When developing humanoid robots, simulation is crucial for testing and validation. NVIDIA Isaac Sim provides advanced physics simulation capabilities that are essential for humanoid robotics development. Here are the key parameters and configurations for Isaac Sim setup:

```json
{
  "isaac_sim_config": {
    "simulation_settings": {
      "physics_engine": "PhysX",
      "gravity": [0.0, 0.0, -9.81],
      "timestep": 0.001,
      "substeps": 1,
      "solver_type": "TGS",
      "solver_iterations": 4,
      "collision_margin": 0.001,
      "contact_offset": 0.02
    },
    "robot_settings": {
      "robot_name": "autonomous_humanoid",
      "urdf_path": "/assets/robots/humanoid.urdf",
      "scale": [1.0, 1.0, 1.0],
      "initial_position": [0.0, 0.0, 1.0],
      "initial_orientation": [0.0, 0.0, 0.0, 1.0],
      "joints": {
        "left_hip_yaw_joint": {
          "type": "revolute",
          "limits": {"lower": -0.5, "upper": 0.5, "effort": 200.0, "velocity": 3.0},
          "damping": 0.5,
          "friction": 0.0
        },
        "left_hip_roll_joint": {
          "type": "revolute",
          "limits": {"lower": -0.5, "upper": 0.5, "effort": 200.0, "velocity": 3.0},
          "damping": 0.5,
          "friction": 0.0
        },
        "left_hip_pitch_joint": {
          "type": "revolute",
          "limits": {"lower": -1.57, "upper": 1.57, "effort": 200.0, "velocity": 3.0},
          "damping": 0.5,
          "friction": 0.0
        },
        "left_knee_joint": {
          "type": "revolute",
          "limits": {"lower": 0.0, "upper": 2.36, "effort": 200.0, "velocity": 3.0},
          "damping": 0.4,
          "friction": 0.0
        },
        "left_ankle_pitch_joint": {
          "type": "revolute",
          "limits": {"lower": -0.5, "upper": 0.5, "effort": 100.0, "velocity": 3.0},
          "damping": 0.2,
          "friction": 0.0
        },
        "left_ankle_roll_joint": {
          "type": "revolute",
          "limits": {"lower": -0.5, "upper": 0.5, "effort": 100.0, "velocity": 3.0},
          "damping": 0.2,
          "friction": 0.0
        }
      }
    },
    "sensor_settings": {
      "imu": {
        "position": [0.0, 0.0, 0.3],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "linear_acceleration_noise_density": 0.01,
        "angular_velocity_noise_density": 0.001,
        "linear_acceleration_random_walk": 0.001,
        "angular_velocity_random_walk": 0.0001
      },
      "camera": {
        "position": [0.05, 0.0, 0.05],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "resolution": [640, 480],
        "fov": 60.0,
        "near_plane": 0.1,
        "far_plane": 100.0
      },
      "lidar": {
        "position": [0.1, 0.0, 0.8],
        "orientation": [0.0, 0.0, 0.0, 1.0],
        "rotation_frequency": 10.0,
        "channels": 16,
        "points_per_channel": 1000,
        "range": 25.0
      }
    },
    "ros_bridge_settings": {
      "enabled": true,
      "namespace": "autonomous_humanoid",
      "qos_profiles": {
        "sensor_data": {
          "reliability": "reliable",
          "durability": "volatile",
          "depth": 10
        },
        "control_commands": {
          "reliability": "best_effort",
          "durability": "volatile",
          "depth": 1
        }
      },
      "topics": {
        "joint_states": "/autonomous_humanoid/joint_states",
        "imu_data": "/autonomous_humanoid/imu/data_raw",
        "camera_image": "/autonomous_humanoid/camera/image_rect_color",
        "lidar_scan": "/autonomous_humanoid/lidar/scan",
        "cmd_vel": "/autonomous_humanoid/cmd_vel",
        "joint_commands": "/autonomous_humanoid/joint_commands"
      }
    }
  }
}
```

## Best Practices for rclpy in Humanoid Robotics

When developing humanoid robotics applications with rclpy, consider the following best practices:

### 1. Performance Optimization
- Use appropriate QoS profiles for different types of data
- Implement message filtering and decimation for high-frequency data
- Use threading carefully to avoid blocking the main ROS loop
- Consider using C++ for computationally intensive real-time tasks

### 2. Error Handling and Safety
- Implement proper exception handling for all ROS operations
- Use timeouts for service calls and action clients
- Implement safety checks before executing commands
- Monitor system resources and implement graceful degradation

### 3. Memory Management
- Use message buffers appropriately to avoid memory leaks
- Implement proper cleanup in node destruction
- Monitor memory usage in long-running nodes
- Use generators for processing large datasets

### 4. Parameter Management
- Use ROS 2 parameters for runtime configuration
- Implement parameter validation callbacks
- Provide reasonable default values
- Document parameter purposes and constraints

### 5. Testing and Debugging
- Implement diagnostic publishers for system health
- Use ROS 2 logging appropriately (debug, info, warn, error)
- Implement test nodes for individual components
- Use rqt and other ROS 2 tools for debugging

## Summary

rclpy provides a powerful bridge between Python's rich ecosystem of scientific computing and AI libraries and the ROS 2 framework. For humanoid robotics applications, this enables the integration of sophisticated algorithms while maintaining the real-time capabilities needed for robot control. The examples provided demonstrate advanced patterns for publishers, subscribers, services, and parameter management that are essential for developing robust humanoid robot systems.

The integration with simulation environments like NVIDIA Isaac Sim allows for safe testing and validation of complex humanoid behaviors before deployment on physical hardware. Proper use of QoS profiles, parameter management, and error handling enables the development of reliable, high-performance humanoid robot systems that can operate safely in real-world environments.

## Integration with NVIDIA Isaac Sim

The following configuration shows how to set up rclpy nodes to work with NVIDIA Isaac Sim for humanoid robotics simulation:

```python
# Isaac Sim integration configuration
isaac_sim_config = {
    "ros_bridge": {
        "enabled": True,
        "namespace": "autonomous_humanoid",
        "qos_settings": {
            "sensor_data": {
                "reliability": "reliable",
                "durability": "volatile",
                "depth": 10
            },
            "control_commands": {
                "reliability": "best_effort",
                "durability": "volatile",
                "depth": 1
            }
        }
    },
    "robot_interface": {
        "joint_state_topic": "/autonomous_humanoid/joint_states",
        "cmd_vel_topic": "/autonomous_humanoid/cmd_vel",
        "imu_topic": "/autonomous_humanoid/imu/data",
        "camera_topic": "/autonomous_humanoid/camera/image_raw",
        "lidar_topic": "/autonomous_humanoid/lidar/scan"
    },
    "simulation_parameters": {
        "physics_update_rate": 500,  # Hz
        "rendering_update_rate": 60,  # Hz
        "gravity": [0.0, 0.0, -9.81],
        "time_scale": 1.0
    },
    "robot_config": {
        "urdf_path": "/assets/robots/humanoid.urdf",
        "initial_position": [0.0, 0.0, 1.0],
        "initial_orientation": [0.0, 0.0, 0.0, 1.0]
    }
}
```

## Best Practices for rclpy in Humanoid Robotics

When developing humanoid robotics applications with rclpy, consider the following best practices:

### 1. Performance Optimization
- Use appropriate QoS profiles for different types of data
- Implement message filtering and decimation for high-frequency data
- Use threading carefully to avoid blocking the main ROS loop
- Consider using C++ for computationally intensive real-time tasks

### 2. Error Handling and Safety
- Implement proper exception handling for all ROS operations
- Use timeouts for service calls and action clients
- Implement safety checks before executing commands
- Monitor system resources and implement graceful degradation

### 3. Memory Management
- Use message buffers appropriately to avoid memory leaks
- Implement proper cleanup in node destruction
- Monitor memory usage in long-running nodes
- Use generators for processing large datasets

### 4. Parameter Management
- Use ROS 2 parameters for runtime configuration
- Implement parameter validation callbacks
- Provide reasonable default values
- Document parameter purposes and constraints

### 5. Testing and Debugging
- Implement diagnostic publishers for system health
- Use ROS 2 logging appropriately (debug, info, warn, error)
- Implement test nodes for individual components
- Use rqt and other ROS 2 tools for debugging

## Summary

rclpy provides a powerful bridge between Python's rich ecosystem of scientific computing and AI libraries and the ROS 2 framework. For humanoid robotics applications, this enables the integration of sophisticated algorithms while maintaining the real-time capabilities needed for robot control. The examples provided demonstrate advanced patterns for publishers, subscribers, services, and parameter management that are essential for developing robust humanoid robot systems.

The integration with simulation environments like NVIDIA Isaac Sim allows for safe testing and validation of complex humanoid behaviors before deployment on physical hardware. Proper use of QoS profiles, parameter management, and error handling enables the development of reliable, high-performance humanoid robot systems that can operate safely in real-world environments.