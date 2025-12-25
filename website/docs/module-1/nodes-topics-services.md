---
sidebar_position: 1
---

# Nodes, Topics, and Services: The Nervous System of Autonomous Humanoid Robots

## Introduction to ROS 2 Fundamentals

ROS 2 (Robot Operating System 2) serves as the fundamental communication and coordination framework for autonomous humanoid robots. This distributed computing framework enables complex robotic systems to be built from modular, reusable components that communicate through standardized interfaces. The architecture of ROS 2 is designed to support real-time performance, distributed computing, and fault tolerance - all critical requirements for humanoid robotics applications.

ROS 2 implements a data-centric publish-subscribe model where nodes communicate through topics, services, and actions. This architecture enables the development of complex humanoid robots with multiple subsystems (locomotion, perception, planning, control) that can operate independently while maintaining tight coordination through the ROS 2 middleware. The system uses Data Distribution Service (DDS) as its underlying communication layer, providing reliable message delivery, quality of service (QoS) controls, and real-time performance guarantees essential for humanoid robot control.

## Theoretical Foundation of ROS 2 Architecture

The ROS 2 architecture is built upon several key theoretical concepts that make it particularly suitable for humanoid robotics applications:

### Distributed Computing Principles

ROS 2 implements a peer-to-peer distributed computing model where each node operates as an independent process. This architecture provides several advantages for humanoid robotics:

1. **Fault Isolation**: Individual node failures do not bring down the entire system
2. **Scalability**: New capabilities can be added without modifying existing nodes
3. **Language Interoperability**: Nodes can be written in different programming languages (C++, Python, etc.)
4. **Resource Management**: Individual nodes can be allocated specific computational resources

### Middleware Architecture

The middleware layer in ROS 2 provides several critical services:
- **Discovery**: Automatic detection of available nodes and services
- **Transport**: Reliable message delivery with configurable QoS policies
- **Serialization**: Efficient conversion of data structures to network format
- **Security**: Authentication, encryption, and access control mechanisms

### Quality of Service (QoS) Policies

QoS policies in ROS 2 allow fine-tuning of communication behavior based on application requirements. For humanoid robots, these policies are crucial for ensuring real-time performance:

- **Reliability**: Ensuring message delivery (reliable vs best-effort)
- **Durability**: Persistence of messages for late-joining subscribers (transient_local vs volatile)
- **History**: Number of messages to retain (keep_last vs keep_all)
- **Deadline**: Maximum time between consecutive messages
- **Liveliness**: Detection of active publishers/subscribers

## Nodes: The Building Blocks of ROS 2

Nodes represent the fundamental execution units in ROS 2. They encapsulate specific functionality and communicate with other nodes through topics, services, and actions. In humanoid robotics, nodes typically correspond to specific subsystems such as:

- Sensor processing nodes (IMU, cameras, LiDAR)
- Control nodes (joint controllers, trajectory planners)
- Perception nodes (object detection, SLAM)
- Planning nodes (motion planning, path planning)
- Communication nodes (network interfaces, data logging)

### Advanced Node Creation with Parameters and Callbacks

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Time
import numpy as np
import threading
import time
from collections import deque

class AdvancedHumanoidNode(Node):
    """
    Advanced Humanoid Node demonstrating complex ROS 2 patterns for humanoid robotics.
    This node implements multiple communication patterns and real-time processing capabilities.
    """

    def __init__(self):
        super().__init__('advanced_humanoid_node')

        # Initialize parameters with default values
        self.declare_parameter('control_frequency', 100)  # Hz
        self.declare_parameter('robot_name', 'autonomous_humanoid')
        self.declare_parameter('safety_timeout', 1.0)    # seconds
        self.declare_parameter('max_velocity', 1.0)      # m/s

        # Get parameter values
        self.control_frequency = self.get_parameter('control_frequency').value
        self.robot_name = self.get_parameter('robot_name').value
        self.safety_timeout = self.get_parameter('safety_timeout').value
        self.max_velocity = self.get_parameter('max_velocity').value

        # Create QoS profiles for different types of communication
        self.qos_sensor = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.qos_control = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Publishers for different types of data
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            f'/{self.robot_name}/cmd_vel',
            self.qos_control
        )

        self.joint_state_publisher = self.create_publisher(
            JointState,
            f'/{self.robot_name}/joint_states',
            self.qos_sensor
        )

        self.status_publisher = self.create_publisher(
            String,
            f'/{self.robot_name}/status',
            10
        )

        # Subscribers for sensor data and commands
        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            f'/{self.robot_name}/cmd_vel_input',
            self.cmd_vel_callback,
            self.qos_control
        )

        # Timer for control loop
        self.control_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.control_loop
        )

        # Data structures for state management
        self.current_cmd_vel = Twist()
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}
        self.last_command_time = self.get_clock().now()

        # Initialize joint state
        self.initialize_joint_state()

        self.get_logger().info(
            f'Advanced Humanoid Node initialized for {self.robot_name} '
            f'with control frequency {self.control_frequency}Hz'
        )

    def initialize_joint_state(self):
        """Initialize joint state with default values for humanoid robot."""
        # Humanoid robot joint names
        joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint',
            'head_yaw_joint', 'head_pitch_joint'
        ]

        # Initialize all joints to zero position
        for joint_name in joint_names:
            self.joint_positions[joint_name] = 0.0
            self.joint_velocities[joint_name] = 0.0
            self.joint_efforts[joint_name] = 0.0

    def cmd_vel_callback(self, msg):
        """Callback for velocity commands."""
        self.current_cmd_vel = msg
        self.last_command_time = self.get_clock().now()
        self.get_logger().debug(f'Received velocity command: {msg}')

    def control_loop(self):
        """Main control loop executing at specified frequency."""
        current_time = self.get_clock().now()

        # Check for command timeout
        time_since_last_cmd = (current_time - self.last_command_time).nanoseconds / 1e9
        if time_since_last_cmd > self.safety_timeout:
            # Stop the robot if no command received within timeout
            self.current_cmd_vel.linear.x = 0.0
            self.current_cmd_vel.angular.z = 0.0

        # Process control logic here
        self.process_control_logic()

        # Publish joint states
        self.publish_joint_states()

        # Publish status
        status_msg = String()
        status_msg.data = f'Operating normally - Last command: {time_since_last_cmd:.2f}s ago'
        self.status_publisher.publish(status_msg)

    def process_control_logic(self):
        """Implement humanoid-specific control logic."""
        # Example: Simple velocity limiting
        if abs(self.current_cmd_vel.linear.x) > self.max_velocity:
            self.current_cmd_vel.linear.x = np.sign(self.current_cmd_vel.linear.x) * self.max_velocity

        # Example: Update joint positions based on velocity commands
        # This is a simplified example - real humanoid control would be much more complex
        for joint_name in self.joint_positions:
            # Apply some basic joint movement based on velocity command
            # In a real implementation, this would involve inverse kinematics,
            # dynamics calculations, and safety checks
            self.joint_positions[joint_name] += 0.001  # Small increment for demonstration

    def publish_joint_states(self):
        """Publish current joint states."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self.joint_positions.keys())
        msg.position = list(self.joint_positions.values())
        msg.velocity = list(self.joint_velocities.values())
        msg.effort = list(self.joint_efforts.values())

        self.joint_state_publisher.publish(msg)

    def get_joint_position(self, joint_name):
        """Get position of a specific joint."""
        return self.joint_positions.get(joint_name, 0.0)

    def set_joint_position(self, joint_name, position):
        """Set position of a specific joint."""
        if joint_name in self.joint_positions:
            self.joint_positions[joint_name] = position
        else:
            self.get_logger().warn(f'Joint {joint_name} not found in joint state')

def main(args=None):
    rclpy.init(args=args)

    node = AdvancedHumanoidNode()

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

## Topics: Asynchronous Communication in ROS 2

Topics form the backbone of ROS 2's publish-subscribe communication model. This asynchronous communication pattern is particularly well-suited for humanoid robotics where multiple sensors and actuators need to operate simultaneously without blocking each other.

### Topic Communication Characteristics

Topics in ROS 2 have several key characteristics that make them ideal for humanoid robotics:

1. **Many-to-Many**: Multiple publishers can send to the same topic, and multiple subscribers can receive from the same topic
2. **Asynchronous**: Publishers and subscribers operate independently - no direct coupling
3. **Real-time Capable**: With appropriate QoS settings, topics can provide real-time performance
4. **Language Agnostic**: Publishers and subscribers can be written in different languages

### Advanced Topic Usage with Custom Message Types

```cpp
// Example C++ node demonstrating advanced topic usage for humanoid robotics
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <map>

class HumanoidSensorFusionNode : public rclcpp::Node
{
public:
    HumanoidSensorFusionNode() : Node("humanoid_sensor_fusion_node")
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Humanoid Sensor Fusion Node");

        // Create publishers for fused sensor data
        sensor_fusion_publisher_ = this->create_publisher<std_msgs::msg::String>(
            "/autonomous_humanoid/sensor_fusion", 10);

        // Create subscribers for various sensor data
        imu_subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/autonomous_humanoid/imu/data", 10,
            std::bind(&HumanoidSensorFusionNode::imu_callback, this, std::placeholders::_1));

        joint_state_subscriber_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/autonomous_humanoid/joint_states", 10,
            std::bind(&HumanoidSensorFusionNode::joint_state_callback, this, std::placeholders::_1));

        // Timer for processing loop
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10), // 100Hz processing
            std::bind(&HumanoidSensorFusionNode::process_sensor_data, this));

        // Initialize sensor data structures
        initialize_sensor_data();
    }

private:
    void initialize_sensor_data()
    {
        // Initialize data structures for sensor fusion
        for (int i = 0; i < 12; ++i) { // Assuming 12 joints for humanoid
            joint_positions_.push_back(0.0);
            joint_velocities_.push_back(0.0);
            joint_efforts_.push_back(0.0);
        }

        // Initialize IMU data
        imu_orientation_.w = 1.0;
        imu_orientation_.x = 0.0;
        imu_orientation_.y = 0.0;
        imu_orientation_.z = 0.0;

        for (int i = 0; i < 3; ++i) {
            imu_angular_velocity_[i] = 0.0;
            imu_linear_acceleration_[i] = 0.0;
        }
    }

    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        // Store IMU data for fusion
        imu_orientation_ = msg->orientation;
        imu_angular_velocity_[0] = msg->angular_velocity.x;
        imu_angular_velocity_[1] = msg->angular_velocity.y;
        imu_angular_velocity_[2] = msg->angular_velocity.z;
        imu_linear_acceleration_[0] = msg->linear_acceleration.x;
        imu_linear_acceleration_[1] = msg->linear_acceleration.y;
        imu_linear_acceleration_[2] = msg->linear_acceleration.z;

        last_imu_time_ = this->get_clock()->now();
    }

    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        // Store joint state data for fusion
        for (size_t i = 0; i < msg->position.size(); ++i) {
            if (i < joint_positions_.size()) {
                joint_positions_[i] = msg->position[i];
            }
        }

        for (size_t i = 0; i < msg->velocity.size(); ++i) {
            if (i < joint_velocities_.size()) {
                joint_velocities_[i] = msg->velocity[i];
            }
        }

        for (size_t i = 0; i < msg->effort.size(); ++i) {
            if (i < joint_efforts_.size()) {
                joint_efforts_[i] = msg->effort[i];
            }
        }

        last_joint_time_ = this->get_clock()->now();
    }

    void process_sensor_data()
    {
        // Perform sensor fusion calculations
        auto current_time = this->get_clock()->now();

        // Example: Calculate balance state based on IMU and joint data
        double balance_state = calculate_balance_state();

        // Example: Detect potential falls based on sensor data
        bool potential_fall = detect_potential_fall();

        // Create and publish fused sensor data
        auto fusion_msg = std_msgs::msg::String();
        fusion_msg.data = "Balance State: " + std::to_string(balance_state) +
                         ", Potential Fall: " + (potential_fall ? "YES" : "NO");

        sensor_fusion_publisher_->publish(fusion_msg);

        RCLCPP_DEBUG(this->get_logger(), "Published sensor fusion data: %s", fusion_msg.data.c_str());
    }

    double calculate_balance_state()
    {
        // Simplified balance calculation - in real implementation would use
        // more sophisticated algorithms like ZMP (Zero Moment Point) or
        // whole-body control approaches
        double pitch = 2 * asin(imu_orientation_.y); // Simplified pitch calculation
        double roll = 2 * asin(imu_orientation_.x);  // Simplified roll calculation

        // Combine with joint position data for balance estimate
        double balance_score = 0.7 * abs(pitch) + 0.3 * abs(roll);

        return balance_score;
    }

    bool detect_potential_fall()
    {
        // Simple fall detection based on IMU data
        double linear_acceleration_magnitude =
            sqrt(pow(imu_linear_acceleration_[0], 2) +
                 pow(imu_linear_acceleration_[1], 2) +
                 pow(imu_linear_acceleration_[2], 2));

        // Check if acceleration is outside normal bounds (indicating impact)
        if (linear_acceleration_magnitude > 20.0) { // 20 m/s^2 threshold
            return true;
        }

        // Check if orientation is too extreme
        double pitch = 2 * asin(imu_orientation_.y);
        if (abs(pitch) > 1.0) { // 57 degrees threshold
            return true;
        }

        return false;
    }

    // Publishers
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr sensor_fusion_publisher_;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber_;

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;

    // Sensor data storage
    std::vector<double> joint_positions_;
    std::vector<double> joint_velocities_;
    std::vector<double> joint_efforts_;
    geometry_msgs::msg::Quaternion imu_orientation_;
    double imu_angular_velocity_[3];
    double imu_linear_acceleration_[3];

    // Timestamps
    rclcpp::Time last_imu_time_;
    rclcpp::Time last_joint_time_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HumanoidSensorFusionNode>());
    rclcpp::shutdown();
    return 0;
}
```

## Services: Synchronous Communication in ROS 2

Services provide synchronous request-response communication in ROS 2, which is essential for humanoid robotics when immediate responses are required. Unlike topics which are asynchronous, services guarantee that a response will be received before the calling process continues.

### Service Definition and Implementation

```python
# Example service definition (would be in srv/ directory)
# HumanoidControl.srv
# float64[] joint_positions
# float64[] joint_velocities
# float64[] joint_efforts
# ---
# bool success
# string message
# float64 execution_time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from example_interfaces.srv import Trigger, SetBool
from std_srvs.srv import Empty
from builtin_interfaces.msg import Time
import time
import threading
from collections import deque

class HumanoidServiceNode(Node):
    """
    Humanoid Service Node demonstrating various service patterns for humanoid robotics.
    """

    def __init__(self):
        super().__init__('humanoid_service_node')

        # Create services for different humanoid operations
        self.calibrate_service = self.create_service(
            Trigger,
            'calibrate_humanoid',
            self.calibrate_callback
        )

        self.emergency_stop_service = self.create_service(
            Trigger,
            'emergency_stop',
            self.emergency_stop_callback
        )

        self.reset_position_service = self.create_service(
            Empty,
            'reset_position',
            self.reset_position_callback
        )

        self.enable_motors_service = self.create_service(
            SetBool,
            'enable_motors',
            self.enable_motors_callback
        )

        # Store service state
        self.motors_enabled = False
        self.calibration_state = False
        self.emergency_stop_active = False

        self.get_logger().info('Humanoid Service Node initialized with multiple services')

    def calibrate_callback(self, request, response):
        """Calibrate all humanoid joints."""
        self.get_logger().info('Starting calibration procedure')

        try:
            # Simulate calibration process
            self.emergency_stop_active = True  # Stop all movement during calibration

            # Perform calibration steps
            self.get_logger().info('Calibrating joint positions...')
            time.sleep(2)  # Simulate calibration time

            self.get_logger().info('Calibrating sensors...')
            time.sleep(1)  # Simulate sensor calibration

            # Update calibration state
            self.calibration_state = True

            # Re-enable movement after calibration
            self.emergency_stop_active = False

            response.success = True
            response.message = 'Calibration completed successfully'

            self.get_logger().info('Calibration completed successfully')

        except Exception as e:
            response.success = False
            response.message = f'Calibration failed: {str(e)}'
            self.get_logger().error(f'Calibration error: {str(e)}')

        return response

    def emergency_stop_callback(self, request, response):
        """Activate emergency stop."""
        self.get_logger().warn('EMERGENCY STOP ACTIVATED')

        try:
            # Stop all motor movements immediately
            self.emergency_stop_active = True

            # Log the emergency stop
            self.get_logger().info('All motors stopped due to emergency stop')

            response.success = True
            response.message = 'Emergency stop activated'

        except Exception as e:
            response.success = False
            response.message = f'Emergency stop failed: {str(e)}'
            self.get_logger().error(f'Emergency stop error: {str(e)}')

        return response

    def reset_position_callback(self, request, response):
        """Reset humanoid to default position."""
        self.get_logger().info('Resetting humanoid to default position')

        try:
            # Check if motors are enabled
            if not self.motors_enabled:
                raise Exception('Motors not enabled - cannot reset position')

            # Check if emergency stop is active
            if self.emergency_stop_active:
                raise Exception('Emergency stop active - cannot reset position')

            # Move to default position (simplified example)
            default_positions = [0.0] * 12  # 12 joints
            self.move_to_position(default_positions)

            response.success = True
            response.message = 'Position reset completed'

            self.get_logger().info('Position reset completed successfully')

        except Exception as e:
            response.success = False
            response.message = f'Position reset failed: {str(e)}'
            self.get_logger().error(f'Position reset error: {str(e)}')

        return response

    def enable_motors_callback(self, request, response):
        """Enable or disable motors."""
        enable = request.data
        self.get_logger().info(f'{"Enabling" if enable else "Disabling"} motors')

        try:
            if enable:
                # Perform safety checks before enabling
                if self.emergency_stop_active:
                    raise Exception('Cannot enable motors while emergency stop is active')

                # Enable motors (simulated)
                self.motors_enabled = True
                response.success = True
                response.message = 'Motors enabled successfully'

                self.get_logger().info('Motors enabled')
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

        return response

    def move_to_position(self, positions):
        """Move joints to specified positions (simplified implementation)."""
        # This would interface with actual motor controllers in a real implementation
        self.get_logger().debug(f'Moving to positions: {positions}')
        time.sleep(0.1)  # Simulate movement time

def main(args=None):
    rclpy.init(args=args)

    node = HumanoidServiceNode()

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

## Advanced System Architecture for Humanoid Robotics

### ROS 2 Communication Architecture

The following Mermaid.js diagram illustrates the comprehensive communication architecture for an autonomous humanoid robot using ROS 2:

```mermaid
graph TB
    subgraph "Humanoid Robot Control System"
        Perception[Perception Stack<br/>Vision, LiDAR, IMU, Proprioception]
        Planning[Planning Stack<br/>Motion, Path, Trajectory Planning]
        Control[Control Stack<br/>Joint Control, Balance, Whole-Body Control]
        Navigation[Navigation Stack<br/>SLAM, Localization, Path Following]
        Sensing[Sensor Stack<br/>Joint Encoders, Force Sensors, Cameras]
        Actuation[Actuator Stack<br/>Servo Drivers, Motor Controllers]
        Safety[SAFETY SYSTEMS<br/>Emergency Stop, Collision Prevention]
    end

    subgraph "AI & Intelligence Layer"
        Reasoning[Reasoning Engine<br/>Logic, Planning, Decision Making]
        Learning[Learning Engine<br/>RL, Imitation, Online Adaptation]
        PerceptionAI[AI Perception<br/>Object Detection, Scene Understanding]
    end

    subgraph "External Systems"
        Remote[Remote Control<br/>Teleoperation, Commands]
        Monitoring[Monitoring<br/>Logging, Diagnostics, Telemetry]
        Cloud[Cloud Services<br/>AI Training, Data Analytics, OTA Updates]
    end

    subgraph "Communication Protocols"
        DDS[(DDS/RTPS<br/>Data Distribution Service)]
        TCP[TCP/IP<br/>Reliable Communication]
        UDP[UDP<br/>Real-time Streaming]
        SHM[Shared Memory<br/>Intra-process]
    end

    %% Core Data Flow
    Sensing --> Perception
    Perception --> PerceptionAI
    Perception --> Planning
    Planning --> Control
    Control --> Actuation
    Navigation --> Planning
    Safety --> Control

    %% AI Integration
    PerceptionAI --> Reasoning
    Learning --> Control
    Reasoning --> Planning

    %% External Interfaces
    Remote --> Planning
    Remote --> Control
    Monitoring --> All Stacks
    Cloud --> Learning
    Cloud --> Reasoning

    %% Communication Protocols
    DDS -.-> Perception
    DDS -.-> Planning
    TCP -.-> Monitoring
    UDP -.-> Remote
    SHM -.-> Control

    %% Safety Critical Path
    Safety ==> Control
    Control ==> Safety

    style Perception fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    style Planning fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style Control fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Navigation fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style Sensing fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    style Actuation fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    style Safety fill:#ffcdd2,stroke:#c62828,stroke-width:3px
    style Reasoning fill:#e1bee7,stroke:#7b1fa2,stroke-width:2px
    style Learning fill:#d1c4e9,stroke:#5e35b1,stroke-width:2px
    style PerceptionAI fill:#c5e1a5,stroke:#558b2f,stroke-width:2px
    style DDS fill:#b2ebf2,stroke:#006064,stroke-width:1px
    style TCP fill:#b39ddb,stroke:#4527a0,stroke-width:1px
    style UDP fill:#a5d6a7,stroke:#33691e,stroke-width:1px
    style SHM fill:#ffcc80,stroke:#ef6c00,stroke-width:1px
```

### Advanced ROS 2 Middleware Configuration

For humanoid robotics applications, the ROS 2 middleware configuration plays a critical role in ensuring real-time performance and reliable communication. Here's an example of a production-ready configuration:

```cpp
#include <rclcpp/rclcpp.hpp>
#include <rmw/qos_profiles.h>
#include <rmw/types.h>
#include <dds/dds.h>

class HumanoidMiddlewareConfig {
public:
    static rclcpp::QoS getCriticalControlQoS() {
        // Ultra-low latency, high-reliability profile for critical control
        return rclcpp::QoS(
            rclcpp::KeepLast(1))  // Minimal history to reduce latency
            .reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE)
            .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE)
            .deadline(rclcpp::Duration(10000000))  // 10ms deadline
            .liveliness(RMW_QOS_POLICY_LIVELINESS_AUTOMATIC)
            .avoid_ros_namespace_conventions(false);
    }

    static rclcpp::QoS getSensorDataQoS() {
        // High-frequency sensor data with best-effort delivery
        return rclcpp::QoS(
            rclcpp::KeepLast(10))  // Keep recent samples
            .reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT)
            .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE)
            .lifespan(rclcpp::Duration(50000000))  // 50ms lifespan
            .avoid_ros_namespace_conventions(false);
    }

    static rclcpp::QoS getDiagnosticQoS() {
        // Diagnostic data with transient-local durability
        return rclcpp::QoS(
            rclcpp::KeepLast(5))  // Keep last 5 samples
            .reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE)
            .durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL)
            .avoid_ros_namespace_conventions(false);
    }

    // Advanced DDS configuration for real-time performance
    static dds::core::QosPolicy::Partition getPartitionConfig() {
        dds::core::QosPolicy::Partition partition;

        // Separate partitions for different subsystems
        std::vector<std::string> partitions = {
            "control_critical",      // Critical control commands
            "sensing",              // Sensor data streams
            "planning",             // Planning and trajectory data
            "diagnostics",          // System health monitoring
            "ai_intelligence"       // AI/ML inference results
        };

        partition = dds::core::QosPolicy::Partition(partitions);
        return partition;
    }

    // Transport configuration for optimal performance
    static dds::core::policy::TransportPriority getTransportPriority(uint8_t priority = 7) {
        // Higher priority for critical control messages
        return dds::core::policy::TransportPriority(priority);
    }
};

// Example usage in a critical control node
class JointControllerNode : public rclcpp::Node {
public:
    JointControllerNode() : Node("joint_controller_node") {
        // Create publishers with appropriate QoS for different data types

        // Critical joint commands with ultra-low latency
        joint_cmd_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/autonomous_humanoid/joint_commands",
            HumanoidMiddlewareConfig::getCriticalControlQoS()
        );

        // Joint state feedback with sensor-appropriate QoS
        joint_state_publisher_ = this->create_publisher<sensor_msgs::msg::JointState>(
            "/autonomous_humanoid/joint_states",
            HumanoidMiddlewareConfig::getSensorDataQoS()
        );

        // Diagnostic information
        diagnostic_publisher_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
            "/autonomous_humanoid/diagnostics",
            HumanoidMiddlewareConfig::getDiagnosticQoS()
        );

        // Control timer for real-time control loop
        control_timer_ = this->create_wall_timer(
            std::chrono::microseconds(1000), // 1kHz control loop
            std::bind(&JointControllerNode::controlLoop, this)
        );
    }

private:
    void controlLoop() {
        // Real-time control logic with guaranteed timing
        auto start_time = std::chrono::high_resolution_clock::now();

        // Critical control computations
        performCriticalControlCalculations();

        // Publish joint commands
        publishJointCommands();

        // Check timing constraints
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        if (duration.count() > 1000) { // Exceeded 1ms budget
            RCLCPP_WARN(this->get_logger(),
                "Control loop exceeded timing budget: %ld microseconds",
                duration.count());
        }
    }

    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostic_publisher_;
    rclcpp::TimerBase::SharedPtr control_timer_;
};
```

### Real-Time Performance Considerations

For humanoid robotics applications, real-time performance is critical. The following considerations ensure deterministic behavior:

#### 1. Memory Management
```cpp
// Custom allocator for real-time safety
template<typename T>
class RealTimeAllocator {
public:
    using value_type = T;

    RealTimeAllocator() noexcept {}
    template<typename U> RealTimeAllocator(const RealTimeAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        // Pre-allocated memory pool to avoid dynamic allocation during control loops
        static thread_local std::array<T, 1024> memory_pool;
        static thread_local size_t offset = 0;

        if (offset + n > memory_pool.size()) {
            throw std::bad_alloc();
        }

        T* ptr = &memory_pool[offset];
        offset += n;
        return ptr;
    }

    void deallocate(T* p, std::size_t n) noexcept {
        // No-op for memory pool - deallocation happens automatically
    }
};
```

#### 2. Thread Configuration for Real-Time Priority
```cpp
#include <pthread.h>
#include <sched.h>

class RealTimeThreadManager {
public:
    static bool configureThreadForRealTime(pthread_t thread, int priority = 80) {
        struct sched_param param;
        param.sched_priority = priority;  // RT priority (1-99)

        int result = pthread_setschedparam(thread, SCHED_FIFO, &param);
        if (result != 0) {
            RCLCPP_ERROR(rclcpp::get_logger("realtime_manager"),
                "Failed to set real-time priority: %d", result);
            return false;
        }

        // Lock memory pages to prevent page faults during real-time execution
        if (mlockall(MCL_CURRENT | MCL_FUTURE) == -1) {
            RCLCPP_WARN(rclcpp::get_logger("realtime_manager"),
                "Failed to lock memory pages for real-time execution");
        }

        return true;
    }
};
```

#### 3. Deterministic Message Processing
```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import time
from collections import deque
import threading

class DeterministicMessageProcessor(Node):
    """
    Implements deterministic message processing for humanoid robotics applications.
    Ensures bounded execution time and predictable message handling.
    """

    def __init__(self):
        super().__init__('deterministic_message_processor')

        # Configuration parameters
        self.declare_parameter('max_queue_size', 10)
        self.declare_parameter('max_processing_time_ms', 5)
        self.declare_parameter('enable_statistics', True)

        self.max_queue_size = self.get_parameter('max_queue_size').value
        self.max_processing_time_ms = self.get_parameter('max_processing_time_ms').value
        self.enable_statistics = self.get_parameter('enable_statistics').value

        # Message queues with bounded size
        self.joint_state_queue = deque(maxlen=self.max_queue_size)
        self.command_queue = deque(maxlen=self.max_queue_size)

        # Statistics
        self.stats_lock = threading.Lock()
        self.processing_times = deque(maxlen=100)
        self.dropped_messages = 0
        self.processed_messages = 0

        # Subscribers
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/autonomous_humanoid/joint_states',
            self.joint_state_callback,
            self.create_qos_profile()
        )

        self.cmd_sub = self.create_subscription(
            JointState,
            '/autonomous_humanoid/joint_commands',
            self.command_callback,
            self.create_qos_profile()
        )

        # Processing timer
        self.process_timer = self.create_timer(
            0.001,  # 1kHz processing rate
            self.process_messages
        )

        self.get_logger().info(
            f'Initialized deterministic message processor with queue size {self.max_queue_size} '
            f'and max processing time {self.max_processing_time_ms}ms'
        )

    def create_qos_profile(self):
        """Create appropriate QoS profile for deterministic processing."""
        return QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

    def joint_state_callback(self, msg):
        """Non-blocking message callback that adds to processing queue."""
        try:
            self.joint_state_queue.append({
                'timestamp': time.time(),
                'message': msg,
                'seq_num': getattr(msg, 'header', {}).get('seq', 0) if hasattr(msg, 'header') else 0
            })
        except IndexError:
            # Queue full, message dropped
            with self.stats_lock:
                self.dropped_messages += 1

    def command_callback(self, msg):
        """Non-blocking command callback."""
        try:
            self.command_queue.append({
                'timestamp': time.time(),
                'message': msg,
                'seq_num': getattr(msg, 'header', {}).get('seq', 0) if hasattr(msg, 'header') else 0
            })
        except IndexError:
            with self.stats_lock:
                self.dropped_messages += 1

    def process_messages(self):
        """Deterministic message processing with bounded execution time."""
        start_time = time.perf_counter()

        # Process joint states
        while self.joint_state_queue:
            if self._exceeds_processing_budget(start_time):
                break

            try:
                msg_data = self.joint_state_queue.popleft()
                self._process_joint_state(msg_data['message'])

                with self.stats_lock:
                    self.processed_messages += 1
            except IndexError:
                break  # Queue became empty

        # Process commands
        while self.command_queue:
            if self._exceeds_processing_budget(start_time):
                break

            try:
                msg_data = self.command_queue.popleft()
                self._process_command(msg_data['message'])

                with self.stats_lock:
                    self.processed_messages += 1
            except IndexError:
                break  # Queue became empty

        # Record processing time statistics
        if self.enable_statistics:
            end_time = time.perf_counter()
            processing_time_ms = (end_time - start_time) * 1000
            with self.stats_lock:
                self.processing_times.append(processing_time_ms)

    def _exceeds_processing_budget(self, start_time):
        """Check if processing time exceeds budget."""
        current_time = time.perf_counter()
        elapsed_ms = (current_time - start_time) * 1000
        return elapsed_ms >= self.max_processing_time_ms

    def _process_joint_state(self, msg):
        """Process joint state message with bounded execution time."""
        # Implement joint state processing logic here
        # This should be a fast, deterministic operation
        pass

    def _process_command(self, msg):
        """Process command message with bounded execution time."""
        # Implement command processing logic here
        # This should be a fast, deterministic operation
        pass

    def get_processing_statistics(self):
        """Get processing statistics for performance monitoring."""
        with self.stats_lock:
            if self.processing_times:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                max_time = max(self.processing_times)
                min_time = min(self.processing_times)
            else:
                avg_time = max_time = min_time = 0.0

            stats = {
                'processed_messages': self.processed_messages,
                'dropped_messages': self.dropped_messages,
                'avg_processing_time_ms': avg_time,
                'max_processing_time_ms': max_time,
                'min_processing_time_ms': min_time,
                'queue_utilization': len(self.joint_state_queue) / self.max_queue_size
            }

            return stats
```

### Advanced Communication Patterns for Humanoid Robotics

Humanoid robots require sophisticated communication patterns to coordinate their complex subsystems. Here's an example of a state machine implementation with real-time safety considerations:

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import time
import threading
from collections import deque
import numpy as np

class HumanoidState(Enum):
    """Enumeration of possible humanoid states with safety considerations."""
    IDLE = "idle"
    WALKING = "walking"
    STANDING = "standing"
    BALANCING = "balancing"
    CALIBRATING = "calibrating"
    EMERGENCY_STOP = "emergency_stop"
    RECOVERY = "recovery"
    SAFETY_LOCKOUT = "safety_lockout"

@dataclass
class StateTransition:
    """Represents a state transition with safety conditions."""
    from_state: HumanoidState
    to_state: HumanoidState
    condition: str
    action: str
    safety_check: Callable[[], bool]  # Safety function that must return True

class HumanoidStateMachine:
    """
    Real-time safe state machine for managing humanoid robot behavior.
    Implements safety checks and bounded execution time for all operations.
    """

    def __init__(self, node):
        self.node = node
        self.current_state = HumanoidState.IDLE
        self.previous_state = HumanoidState.IDLE
        self.state_lock = threading.RLock()  # Recursive lock for state operations
        self.state_entry_time = time.time()

        # Initialize transition rules with safety functions
        self.transitions = [
            StateTransition(
                HumanoidState.IDLE, HumanoidState.WALKING,
                "walking_command_received", "start_walking_sequence",
                lambda: self._is_safe_to_walk()
            ),
            StateTransition(
                HumanoidState.WALKING, HumanoidState.STANDING,
                "stop_command_received", "execute_stop_sequence",
                lambda: True  # Standing is always safe
            ),
            StateTransition(
                HumanoidState.WALKING, HumanoidState.BALANCING,
                "imbalance_detected", "activate_balance_control",
                lambda: self._is_safe_to_balance()
            ),
            StateTransition(
                HumanoidState.BALANCING, HumanoidState.STANDING,
                "balance_restored", "return_to_standing",
                lambda: self._is_safe_to_stand()
            ),
            StateTransition(
                HumanoidState.IDLE, HumanoidState.CALIBRATING,
                "calibrate_command", "execute_calibration",
                lambda: self._is_safe_to_calibrate()
            ),
            StateTransition(
                HumanoidState.WALKING, HumanoidState.EMERGENCY_STOP,
                "emergency_stop_triggered", "execute_emergency_stop",
                lambda: True  # Emergency stop is always allowed
            ),
            StateTransition(
                HumanoidState.EMERGENCY_STOP, HumanoidState.SAFETY_LOCKOUT,
                "manual_override_required", "activate_safety_lockout",
                lambda: True  # Safety lockout is always safe
            ),
            StateTransition(
                HumanoidState.SAFETY_LOCKOUT, HumanoidState.RECOVERY,
                "safety_clear_manual", "execute_recovery_sequence",
                lambda: self._is_safe_to_recover()
            ),
            StateTransition(
                HumanoidState.RECOVERY, HumanoidState.STANDING,
                "recovery_complete", "stand_up_sequence",
                lambda: self._is_safe_to_stand()
            )
        ]

        # Initialize sensor data with bounded buffers
        self.sensor_data = {
            'imu_orientation': deque(maxlen=10),
            'imu_angular_velocity': deque(maxlen=10),
            'imu_linear_acceleration': deque(maxlen=10),
            'joint_positions': {},
            'joint_velocities': {},
            'joint_efforts': {},
            'force_sensors': deque(maxlen=10),
            'contact_sensors': deque(maxlen=10)
        }

        # Command buffers with bounded size
        self.command_buffers = {
            'walking_commands': deque(maxlen=5),
            'balance_commands': deque(maxlen=5),
            'safety_commands': deque(maxlen=10)
        }

        # Safety monitoring
        self.safety_monitoring_active = True
        self.emergency_stop_active = False
        self.safety_violations = 0

        # Performance monitoring
        self.state_transition_times = deque(maxlen=50)

        self.node.get_logger().info(
            f'Humanoid State Machine initialized in {self.current_state.value} state with safety monitoring'
        )

    def update_sensor_data(self, sensor_msg):
        """Update internal sensor data with bounded execution time."""
        start_time = time.time()

        with self.state_lock:
            # Add timestamped sensor data to appropriate buffers
            timestamp = time.time()

            if hasattr(sensor_msg, 'orientation'):
                # IMU data
                self.sensor_data['imu_orientation'].append({
                    'data': [
                        sensor_msg.orientation.x,
                        sensor_msg.orientation.y,
                        sensor_msg.orientation.z,
                        sensor_msg.orientation.w
                    ],
                    'timestamp': timestamp
                })

                self.sensor_data['imu_angular_velocity'].append({
                    'data': [
                        sensor_msg.angular_velocity.x,
                        sensor_msg.angular_velocity.y,
                        sensor_msg.angular_velocity.z
                    ],
                    'timestamp': timestamp
                })

                self.sensor_data['imu_linear_acceleration'].append({
                    'data': [
                        sensor_msg.linear_acceleration.x,
                        sensor_msg.linear_acceleration.y,
                        sensor_msg.linear_acceleration.z
                    ],
                    'timestamp': timestamp
                })

            # Bounded execution time check
            if time.time() - start_time > 0.001:  # 1ms budget
                self.node.get_logger().warn('Sensor data update exceeded time budget')

    def update_commands(self, cmd_type: str, value, max_age_seconds: float = 1.0):
        """Update command buffers with age-based filtering."""
        with self.state_lock:
            timestamp = time.time()

            # Clean old commands based on age
            if cmd_type in self.command_buffers:
                # Remove commands older than max_age_seconds
                current_buffer = self.command_buffers[cmd_type]
                filtered_buffer = deque(maxlen=current_buffer.maxlen)

                for cmd in current_buffer:
                    if timestamp - cmd['timestamp'] <= max_age_seconds:
                        filtered_buffer.append(cmd)

                self.command_buffers[cmd_type] = filtered_buffer

                # Add new command
                self.command_buffers[cmd_type].append({
                    'data': value,
                    'timestamp': timestamp
                })

    def evaluate_conditions(self) -> Dict[str, bool]:
        """Evaluate all transition conditions with bounded execution time."""
        start_time = time.time()
        conditions = {}

        # Check balance condition (bounded execution)
        pitch, roll = self._get_balance_state()
        conditions['imbalance_detected'] = abs(pitch) > 0.5 or abs(roll) > 0.5
        conditions['balance_restored'] = abs(pitch) < 0.2 and abs(roll) < 0.2

        # Check command conditions (bounded execution)
        conditions['walking_command_received'] = self._has_recent_command('walking_commands')
        conditions['stop_command_received'] = self._has_recent_command('walking_commands', check_stop=True)
        conditions['calibrate_command'] = self._has_recent_command('safety_commands', check_calibrate=True)
        conditions['emergency_stop_triggered'] = self.emergency_stop_active
        conditions['manual_override_required'] = self._needs_manual_override()
        conditions['safety_clear_manual'] = self._is_safety_cleared_manually()
        conditions['recovery_complete'] = self._is_recovery_complete()

        # Time budget check
        if time.time() - start_time > 0.002:  # 2ms budget
            self.node.get_logger().warn('Condition evaluation exceeded time budget')

        return conditions

    def _get_balance_state(self) -> tuple:
        """Get current balance state (pitch, roll) with bounded execution."""
        # Extract latest IMU data
        if self.sensor_data['imu_orientation']:
            latest_orientation = self.sensor_data['imu_orientation'][-1]['data']
            # Convert quaternion to Euler angles (simplified for performance)
            x, y, z, w = latest_orientation

            # Calculate pitch and roll (simplified approximation)
            pitch = 2 * np.arcsin(y) if abs(y) < 0.999 else np.sign(y) * np.pi/2
            roll = 2 * np.arcsin(x) if abs(x) < 0.999 else np.sign(x) * np.pi/2

            return pitch, roll
        else:
            return 0.0, 0.0

    def _has_recent_command(self, buffer_name: str, check_stop: bool = False, check_calibrate: bool = False) -> bool:
        """Check if there are recent commands in the specified buffer."""
        if buffer_name not in self.command_buffers:
            return False

        buffer = self.command_buffers[buffer_name]
        if not buffer:
            return False

        # Check latest command timestamp (within 100ms)
        latest_cmd = buffer[-1]
        return time.time() - latest_cmd['timestamp'] <= 0.1

    def _needs_manual_override(self) -> bool:
        """Check if manual override is needed."""
        return self.safety_violations > 3  # Require manual intervention after 3 violations

    def _is_safety_cleared_manually(self) -> bool:
        """Check if safety has been cleared manually."""
        # In real implementation, this would check for manual safety clear command
        return False  # Placeholder

    def _is_recovery_complete(self) -> bool:
        """Check if recovery sequence is complete."""
        # In real implementation, this would check for recovery completion
        return True  # Placeholder

    def transition_to_state(self, new_state: HumanoidState) -> bool:
        """Safely transition to a new state with safety checks."""
        with self.state_lock:
            # Find transition rule
            transition_rule = None
            for transition in self.transitions:
                if (transition.from_state == self.current_state and
                    transition.to_state == new_state):
                    transition_rule = transition
                    break

            if not transition_rule:
                self.node.get_logger().warn(
                    f'Invalid transition from {self.current_state.value} to {new_state.value}'
                )
                return False

            # Check safety condition
            if not transition_rule.safety_check():
                self.node.get_logger().error(
                    f'Safety check failed for transition to {new_state.value}: {transition_rule.condition}'
                )
                self.safety_violations += 1
                return False

            # Perform transition
            old_state = self.current_state
            start_time = time.time()

            self.previous_state = old_state
            self.current_state = new_state
            self.state_entry_time = time.time()

            # Execute state-specific action
            self._execute_state_action(new_state)

            # Record transition time
            transition_time = time.time() - start_time
            self.state_transition_times.append(transition_time)

            self.node.get_logger().info(
                f'Safe state transition: {old_state.value} -> {new_state.value} '
                f'(time: {transition_time*1000:.2f}ms)'
            )

            return True

    def _execute_state_action(self, state: HumanoidState):
        """Execute actions specific to the current state."""
        start_time = time.time()

        if state == HumanoidState.WALKING:
            self.node.get_logger().info('Executing walking sequence with safety monitoring')
            # Implement walking control logic with safety checks
        elif state == HumanoidState.STANDING:
            self.node.get_logger().info('Executing standing sequence with balance maintenance')
            # Implement standing control logic
        elif state == HumanoidState.BALANCING:
            self.node.get_logger().info('Executing balance control with active stabilization')
            # Implement balance control logic
        elif state == HumanoidState.CALIBRATING:
            self.node.get_logger().info('Executing calibration sequence with motion freeze')
            # Implement calibration logic
        elif state == HumanoidState.EMERGENCY_STOP:
            self.node.get_logger().warn('Executing emergency stop - all motion frozen')
            # Implement emergency stop logic
        elif state == HumanoidState.RECOVERY:
            self.node.get_logger().info('Executing recovery sequence with gradual activation')
            # Implement recovery logic
        elif state == HumanoidState.SAFETY_LOCKOUT:
            self.node.get_logger().warn('Entering safety lockout - manual intervention required')
            # Implement safety lockout logic

        # Check execution time
        execution_time = time.time() - start_time
        if execution_time > 0.005:  # 5ms budget
            self.node.get_logger().warn(
                f'State action for {state.value} exceeded time budget: {execution_time*1000:.2f}ms'
            )

    def update_state_machine(self) -> bool:
        """Update the state machine based on current conditions with safety monitoring."""
        start_time = time.time()

        # Evaluate conditions
        conditions = self.evaluate_conditions()

        # Check for valid transitions
        for transition in self.transitions:
            if (transition.from_state == self.current_state and
                conditions.get(transition.condition, False)):

                # Attempt to transition (includes safety checks)
                if self.transition_to_state(transition.to_state):
                    # Successfully transitioned
                    break
                else:
                    # Transition failed due to safety violation
                    self.node.get_logger().error(
                        f'Transition from {self.current_state.value} to {transition.to_state.value} '
                        f'failed due to safety violation: {transition.condition}'
                    )

        # Safety monitoring
        if self.safety_monitoring_active:
            self._perform_safety_checks()

        # Time budget check
        total_time = time.time() - start_time
        if total_time > 0.01:  # 10ms budget for entire update
            self.node.get_logger().warn(
                f'State machine update exceeded time budget: {total_time*1000:.2f}ms'
            )

        return True

    def _perform_safety_checks(self):
        """Perform continuous safety monitoring."""
        # Check for dangerous conditions that require immediate action
        pitch, roll = self._get_balance_state()

        # Extreme imbalance - trigger emergency stop
        if abs(pitch) > 1.0 or abs(roll) > 1.0:  # 57 degrees
            self.node.get_logger().error('Extreme imbalance detected - triggering emergency stop')
            self.emergency_stop_active = True
            self.transition_to_state(HumanoidState.EMERGENCY_STOP)

        # Check for joint limit violations
        self._check_joint_limits()

    def _check_joint_limits(self):
        """Check for joint limit violations."""
        # In real implementation, this would check joint position/velocity/effort limits
        pass

    def _is_safe_to_walk(self) -> bool:
        """Safety check for walking transition."""
        pitch, roll = self._get_balance_state()
        return abs(pitch) < 0.3 and abs(roll) < 0.3  # Less than 17 degrees tilt

    def _is_safe_to_balance(self) -> bool:
        """Safety check for balancing transition."""
        return True  # Balancing is always safer than falling

    def _is_safe_to_stand(self) -> bool:
        """Safety check for standing transition."""
        pitch, roll = self._get_balance_state()
        return abs(pitch) < 0.5 and abs(roll) < 0.5  # Less than 29 degrees tilt

    def _is_safe_to_calibrate(self) -> bool:
        """Safety check for calibration transition."""
        return self.current_state in [HumanoidState.IDLE, HumanoidState.STANDING]

    def _is_safe_to_recover(self) -> bool:
        """Safety check for recovery transition."""
        return self.safety_violations <= 5  # Don't recover if too many violations

    def get_current_state(self) -> HumanoidState:
        """Get the current state with thread safety."""
        with self.state_lock:
            return self.current_state

    def is_in_state(self, state: HumanoidState) -> bool:
        """Check if the robot is in a specific state with thread safety."""
        with self.state_lock:
            return self.current_state == state

    def get_state_duration(self) -> float:
        """Get the duration since entering the current state."""
        with self.state_lock:
            return time.time() - self.state_entry_time

    def get_performance_metrics(self) -> Dict:
        """Get performance and safety metrics."""
        with self.state_lock:
            if self.state_transition_times:
                avg_transition_time = sum(self.state_transition_times) / len(self.state_transition_times)
                max_transition_time = max(self.state_transition_times)
            else:
                avg_transition_time = max_transition_time = 0.0

            return {
                'current_state': self.current_state.value,
                'state_duration': self.get_state_duration(),
                'safety_violations': self.safety_violations,
                'avg_transition_time_ms': avg_transition_time * 1000,
                'max_transition_time_ms': max_transition_time * 1000,
                'emergency_stop_active': self.emergency_stop_active,
                'safety_monitoring_active': self.safety_monitoring_active
            }

# Integration with ROS 2 node
class HumanoidStateNode(Node):
    def __init__(self):
        super().__init__('humanoid_state_node')

        # Declare parameters
        self.declare_parameter('control_frequency', 100)  # Hz
        self.declare_parameter('enable_safety_monitoring', True)
        self.declare_parameter('max_safety_violations_before_lockout', 5)

        # Get parameter values
        self.control_frequency = self.get_parameter('control_frequency').value
        self.enable_safety_monitoring = self.get_parameter('enable_safety_monitoring').value
        self.max_safety_violations = self.get_parameter('max_safety_violations_before_lockout').value

        # Initialize state machine
        self.state_machine = HumanoidStateMachine(self)

        # Create timer for state machine updates
        self.state_timer = self.create_timer(
            1.0 / self.control_frequency,
            self.state_machine.update_state_machine
        )

        # Safety monitoring timer
        if self.enable_safety_monitoring:
            self.safety_timer = self.create_timer(
                0.1,  # 10Hz safety checks
                self.perform_additional_safety_checks
            )

        self.get_logger().info(
            f'Humanoid State Node initialized with {self.control_frequency}Hz control frequency'
        )

    def perform_additional_safety_checks(self):
        """Perform additional safety checks outside the main control loop."""
        metrics = self.state_machine.get_performance_metrics()

        # Check safety violation count
        if metrics['safety_violations'] >= self.max_safety_violations:
            self.get_logger().critical(
                f'Maximum safety violations reached ({metrics["safety_violations"]}). '
                'Activating safety lockout.'
            )
            self.state_machine.transition_to_state(HumanoidState.SAFETY_LOCKOUT)

        # Log safety metrics periodically
        if self.get_clock().now().nanoseconds % 1000000000 < 10000000:  # Every ~1 second
            self.get_logger().info(
                f'Safety metrics - Violations: {metrics["safety_violations"]}, '
                f'State: {metrics["current_state"]}, '
                f'Transition time: {metrics["avg_transition_time_ms"]:.2f}ms'
            )

def main(args=None):
    rclpy.init(args=args)

    node = HumanoidStateNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('State node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## NVIDIA Isaac Sim Integration for Humanoid Robotics

When developing humanoid robots, simulation is crucial for testing and validation. NVIDIA Isaac Sim provides advanced physics simulation capabilities that are essential for humanoid robotics development. Here are the key parameters and configurations for Isaac Sim setup:

### Isaac Sim Configuration Parameters

```json
{
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
      "left_hip_joint": {
        "type": "revolute",
        "limits": {"lower": -1.57, "upper": 1.57, "effort": 100.0, "velocity": 5.0},
        "damping": 0.1,
        "friction": 0.0
      },
      "left_knee_joint": {
        "type": "revolute",
        "limits": {"lower": 0.0, "upper": 2.36, "effort": 100.0, "velocity": 5.0},
        "damping": 0.1,
        "friction": 0.0
      },
      "right_hip_joint": {
        "type": "revolute",
        "limits": {"lower": -1.57, "upper": 1.57, "effort": 100.0, "velocity": 5.0},
        "damping": 0.1,
        "friction": 0.0
      },
      "right_knee_joint": {
        "type": "revolute",
        "limits": {"lower": 0.0, "upper": 2.36, "effort": 100.0, "velocity": 5.0},
        "damping": 0.1,
        "friction": 0.0
      }
    }
  },
  "sensor_settings": {
    "camera": {
      "resolution": [640, 480],
      "fov": 60.0,
      "near_plane": 0.1,
      "far_plane": 100.0,
      "position": [0.5, 0.0, 1.5]
    },
    "lidar": {
      "rotation_frequency": 10.0,
      "channels": 16,
      "points_per_channel": 1000,
      "range": 25.0,
      "position": [0.3, 0.0, 1.2]
    },
    "imu": {
      "linear_acceleration_noise_density": 0.017,
      "angular_velocity_noise_density": 0.001,
      "linear_acceleration_random_walk": 0.00017,
      "angular_velocity_random_walk": 0.00001
    }
  },
  "environment_settings": {
    "ground_plane": {
      "size": [10.0, 10.0],
      "static_friction": 0.5,
      "dynamic_friction": 0.5,
      "restitution": 0.0
    },
    "lighting": {
      "ambient_light": [0.3, 0.3, 0.3],
      "directional_light": {
        "direction": [-0.5, -0.5, -1.0],
        "color": [1.0, 1.0, 1.0],
        "intensity": 500.0
      }
    }
  },
  "ros_bridge_settings": {
    "enabled": true,
    "topics": {
      "joint_states": "/autonomous_humanoid/joint_states",
      "cmd_vel": "/autonomous_humanoid/cmd_vel",
      "imu_data": "/autonomous_humanoid/imu/data",
      "camera_image": "/autonomous_humanoid/camera/image_raw",
      "lidar_scan": "/autonomous_humanoid/lidar/scan"
    }
  }
}
```

## Advanced Communication Patterns for Humanoid Robotics

Humanoid robots require sophisticated communication patterns to coordinate their complex subsystems. Here's an example of a state machine implementation for humanoid behavior:

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
from threading import Lock

class HumanoidState(Enum):
    """Enumeration of possible humanoid states."""
    IDLE = "idle"
    WALKING = "walking"
    STANDING = "standing"
    BALANCING = "balancing"
    CALIBRATING = "calibrating"
    EMERGENCY_STOP = "emergency_stop"
    RECOVERY = "recovery"

@dataclass
class HumanoidStateTransition:
    """Represents a state transition with conditions."""
    from_state: HumanoidState
    to_state: HumanoidState
    condition: str
    action: str

class HumanoidStateMachine:
    """
    State machine for managing humanoid robot behavior.
    This implementation demonstrates advanced ROS 2 usage patterns.
    """

    def __init__(self, node):
        self.node = node
        self.current_state = HumanoidState.IDLE
        self.previous_state = HumanoidState.IDLE
        self.state_lock = Lock()

        # Initialize state transition rules
        self.transitions = [
            HumanoidStateTransition(
                HumanoidState.IDLE, HumanoidState.WALKING,
                "walking_command_received", "start_walking_sequence"
            ),
            HumanoidStateTransition(
                HumanoidState.WALKING, HumanoidState.STANDING,
                "stop_command_received", "execute_stop_sequence"
            ),
            HumanoidStateTransition(
                HumanoidState.WALKING, HumanoidState.BALANCING,
                "imbalance_detected", "activate_balance_control"
            ),
            HumanoidStateTransition(
                HumanoidState.BALANCING, HumanoidState.STANDING,
                "balance_restored", "return_to_standing"
            ),
            HumanoidStateTransition(
                HumanoidState.IDLE, HumanoidState.CALIBRATING,
                "calibrate_command", "execute_calibration"
            ),
            HumanoidStateTransition(
                HumanoidState.WALKING, HumanoidState.EMERGENCY_STOP,
                "emergency_stop", "execute_emergency_stop"
            ),
            HumanoidStateTransition(
                HumanoidState.EMERGENCY_STOP, HumanoidState.RECOVERY,
                "manual_recovery", "execute_recovery_sequence"
            ),
            HumanoidStateTransition(
                HumanoidState.RECOVERY, HumanoidState.STANDING,
                "recovery_complete", "stand_up_sequence"
            )
        ]

        # Initialize sensor data
        self.sensor_data = {
            'imu_orientation': [0.0, 0.0, 0.0, 1.0],
            'imu_angular_velocity': [0.0, 0.0, 0.0],
            'imu_linear_acceleration': [0.0, 0.0, -9.81],
            'joint_positions': {},
            'joint_velocities': {},
            'joint_efforts': {},
            'force_sensors': {}
        }

        # Initialize command flags
        self.commands = {
            'walking_requested': False,
            'stop_requested': False,
            'calibrate_requested': False,
            'emergency_stop': False,
            'manual_recovery': False
        }

        self.node.get_logger().info(f'Humanoid State Machine initialized in {self.current_state.value} state')

    def update_sensor_data(self, sensor_msg):
        """Update internal sensor data from ROS messages."""
        with self.state_lock:
            # Update sensor data based on message type
            if hasattr(sensor_msg, 'orientation'):
                # IMU data
                self.sensor_data['imu_orientation'] = [
                    sensor_msg.orientation.x,
                    sensor_msg.orientation.y,
                    sensor_msg.orientation.z,
                    sensor_msg.orientation.w
                ]
                self.sensor_data['imu_angular_velocity'] = [
                    sensor_msg.angular_velocity.x,
                    sensor_msg.angular_velocity.y,
                    sensor_msg.angular_velocity.z
                ]
                self.sensor_data['imu_linear_acceleration'] = [
                    sensor_msg.linear_acceleration.x,
                    sensor_msg.linear_acceleration.y,
                    sensor_msg.linear_acceleration.z
                ]

    def update_commands(self, cmd_type, value=True):
        """Update command flags."""
        with self.state_lock:
            if cmd_type in self.commands:
                self.commands[cmd_type] = value

    def evaluate_conditions(self):
        """Evaluate all transition conditions."""
        conditions = {}

        # Check balance condition
        pitch = 2 * self.node.get_parameter('balance_threshold').value
        roll = 2 * self.node.get_parameter('balance_threshold').value
        conditions['imbalance_detected'] = abs(pitch) > 0.5 or abs(roll) > 0.5
        conditions['balance_restored'] = abs(pitch) < 0.2 and abs(roll) < 0.2

        # Check command conditions
        conditions['walking_command_received'] = self.commands['walking_requested']
        conditions['stop_command_received'] = self.commands['stop_requested']
        conditions['calibrate_command'] = self.commands['calibrate_requested']
        conditions['emergency_stop'] = self.commands['emergency_stop']
        conditions['manual_recovery'] = self.commands['manual_recovery']

        # Check recovery conditions
        conditions['recovery_complete'] = True  # Simplified for example

        return conditions

    def transition_to_state(self, new_state: HumanoidState):
        """Transition to a new state with proper cleanup."""
        with self.state_lock:
            if new_state != self.current_state:
                self.previous_state = self.current_state
                old_state = self.current_state
                self.current_state = new_state

                self.node.get_logger().info(
                    f'State transition: {old_state.value} -> {new_state.value}'
                )

                # Execute state-specific actions
                self.execute_state_action(new_state)

    def execute_state_action(self, state: HumanoidState):
        """Execute actions specific to the current state."""
        if state == HumanoidState.WALKING:
            self.node.get_logger().info('Executing walking sequence')
            # Implement walking control logic
        elif state == HumanoidState.STANDING:
            self.node.get_logger().info('Executing standing sequence')
            # Implement standing control logic
        elif state == HumanoidState.BALANCING:
            self.node.get_logger().info('Executing balance control')
            # Implement balance control logic
        elif state == HumanoidState.CALIBRATING:
            self.node.get_logger().info('Executing calibration sequence')
            # Implement calibration logic
        elif state == HumanoidState.EMERGENCY_STOP:
            self.node.get_logger().info('Executing emergency stop')
            # Implement emergency stop logic
        elif state == HumanoidState.RECOVERY:
            self.node.get_logger().info('Executing recovery sequence')
            # Implement recovery logic

    def update_state_machine(self):
        """Update the state machine based on current conditions."""
        conditions = self.evaluate_conditions()

        # Check for valid transitions
        for transition in self.transitions:
            if (transition.from_state == self.current_state and
                conditions.get(transition.condition, False)):

                self.transition_to_state(transition.to_state)
                break  # Only one transition per update

    def get_current_state(self) -> HumanoidState:
        """Get the current state."""
        with self.state_lock:
            return self.current_state

    def is_in_state(self, state: HumanoidState) -> bool:
        """Check if the robot is in a specific state."""
        with self.state_lock:
            return self.current_state == state

# Integration with ROS 2 node
class HumanoidStateNode(Node):
    def __init__(self):
        super().__init__('humanoid_state_node')

        # Declare parameters
        self.declare_parameter('balance_threshold', 0.1)
        self.declare_parameter('control_frequency', 100)

        # Initialize state machine
        self.state_machine = HumanoidStateMachine(self)

        # Create timer for state machine updates
        self.state_timer = self.create_timer(
            1.0 / self.get_parameter('control_frequency').value,
            self.state_machine.update_state_machine
        )

        self.get_logger().info('Humanoid State Node initialized')

def main(args=None):
    rclpy.init(args=args)

    node = HumanoidStateNode()

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

## Summary

Understanding nodes, topics, and services is crucial for building distributed robotic systems, particularly for autonomous humanoid robots. These communication patterns enable the development of complex, modular systems where different components can operate independently while maintaining coordinated behavior. The advanced examples provided demonstrate how these basic concepts can be extended to create sophisticated humanoid control systems with proper error handling, safety mechanisms, and state management.

The integration with NVIDIA Isaac Sim provides realistic physics simulation capabilities that are essential for testing humanoid robot algorithms before deployment on physical hardware. The configuration parameters ensure that the simulation accurately represents the real-world dynamics of the humanoid robot, enabling safe and efficient development and testing of complex behaviors.

Proper use of QoS policies, state machines, and service-based communication patterns enables the development of robust, real-time capable humanoid robot systems that can handle the complex requirements of autonomous operation.