---
sidebar_position: 3
---

# URDF Humanoids: Advanced Robot Modeling for Autonomous Humanoid Systems

## Introduction to URDF for Humanoid Robotics

Unified Robot Description Format (URDF) is an XML-based format for representing robot models in ROS. For humanoid robotics, URDF serves as the foundation for simulation, visualization, kinematic analysis, and motion planning. A well-designed URDF model is essential for developing and testing humanoid robots in simulation environments like Gazebo, NVIDIA Isaac Sim, and other physics simulators before deploying to physical hardware.

URDF allows for the specification of robot geometry, kinematics, dynamics, and visual properties in a standardized format that can be consumed by various ROS tools and simulation environments. For humanoid robots, which have complex multi-degree-of-freedom structures with numerous joints and links, URDF provides the necessary framework to accurately model the robot's physical characteristics.

## Theoretical Foundation of URDF Modeling

### Kinematic Chain Structure

URDF models represent robots as kinematic chains composed of links connected by joints. Each link represents a rigid body with mass, inertia, and visual/collision properties, while each joint defines the relationship between two links with specific degrees of freedom.

For humanoid robots, the kinematic structure typically includes:
- A base link (usually the pelvis or torso)
- Multiple limbs (arms and legs) with serial kinematic chains
- End effectors (hands and feet)
- Additional links for sensors, cameras, and other components

### Dynamic Properties

URDF supports the specification of dynamic properties including:
- Mass and inertia tensors for each link
- Joint friction and damping coefficients
- Joint limits and safety constraints
- Center of mass information for stability analysis

## Advanced URDF Structure for Humanoid Robots

### Complete Humanoid URDF Model

```xml
<?xml version="1.0"?>
<robot name="advanced_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include common definitions -->
  <xacro:include filename="$(find advanced_humanoid_description)/urdf/materials.urdf.xacro"/>
  <xacro:include filename="$(find advanced_humanoid_description)/urdf/transmissions.urdf.xacro"/>
  <xacro:include filename="$(find advanced_humanoid_description)/urdf/gazebo.urdf.xacro"/>

  <!-- Base torso link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.8" radius="0.15"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.8" radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0.5" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head link -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head_link"/>
    <origin xyz="0 0 0.95" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.785" upper="0.785" effort="50" velocity="3.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="head_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left arm - shoulder -->
  <joint name="left_shoulder_yaw_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_shoulder_yaw_link"/>
    <origin xyz="0.0 0.15 0.7" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="3.0"/>
    <dynamics damping="0.2" friction="0.0"/>
  </joint>

  <link name="left_shoulder_yaw_link">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_pitch_joint" type="revolute">
    <parent link="left_shoulder_yaw_link"/>
    <child link="left_shoulder_pitch_link"/>
    <origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="3.0"/>
    <dynamics damping="0.2" friction="0.0"/>
  </joint>

  <link name="left_shoulder_pitch_link">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_shoulder_roll_joint" type="revolute">
    <parent link="left_shoulder_pitch_link"/>
    <child link="left_upper_arm_link"/>
    <origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="2.0" effort="100" velocity="3.0"/>
    <dynamics damping="0.2" friction="0.0"/>
  </joint>

  <link name="left_upper_arm_link">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.3" radius="0.06"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.3" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Left arm - elbow -->
  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_upper_arm_link"/>
    <child link="left_lower_arm_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="2.36" effort="80" velocity="3.0"/>
    <dynamics damping="0.15" friction="0.0"/>
  </joint>

  <link name="left_lower_arm_link">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.3" radius="0.05"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.007" ixy="0.0" ixz="0.0" iyy="0.007" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>

  <!-- Left hand -->
  <joint name="left_wrist_joint" type="revolute">
    <parent link="left_lower_arm_link"/>
    <child link="left_hand_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="3.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="left_hand_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.3"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Right arm (mirrored) -->
  <joint name="right_shoulder_yaw_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_shoulder_yaw_link"/>
    <origin xyz="0.0 -0.15 0.7" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="3.0"/>
    <dynamics damping="0.2" friction="0.0"/>
  </joint>

  <link name="right_shoulder_yaw_link">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_shoulder_pitch_joint" type="revolute">
    <parent link="right_shoulder_yaw_link"/>
    <child link="right_shoulder_pitch_link"/>
    <origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="3.0"/>
    <dynamics damping="0.2" friction="0.0"/>
  </joint>

  <link name="right_shoulder_pitch_link">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_shoulder_roll_joint" type="revolute">
    <parent link="right_shoulder_pitch_link"/>
    <child link="right_upper_arm_link"/>
    <origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="2.0" effort="100" velocity="3.0"/>
    <dynamics damping="0.2" friction="0.0"/>
  </joint>

  <link name="right_upper_arm_link">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.3" radius="0.06"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.3" radius="0.06"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_upper_arm_link"/>
    <child link="right_lower_arm_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="2.36" effort="80" velocity="3.0"/>
    <dynamics damping="0.15" friction="0.0"/>
  </joint>

  <link name="right_lower_arm_link">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.3" radius="0.05"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.007" ixy="0.0" ixz="0.0" iyy="0.007" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>

  <joint name="right_wrist_joint" type="revolute">
    <parent link="right_lower_arm_link"/>
    <child link="right_hand_link"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="3.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </joint>

  <link name="right_hand_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.3"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Left leg - hip -->
  <joint name="left_hip_yaw_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_hip_yaw_link"/>
    <origin xyz="0.0 0.08 0.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="200" velocity="3.0"/>
    <dynamics damping="0.5" friction="0.0"/>
  </joint>

  <link name="left_hip_yaw_link">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_hip_roll_joint" type="revolute">
    <parent link="left_hip_yaw_link"/>
    <child link="left_hip_roll_link"/>
    <origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="200" velocity="3.0"/>
    <dynamics damping="0.5" friction="0.0"/>
  </joint>

  <link name="left_hip_roll_link">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="left_hip_pitch_joint" type="revolute">
    <parent link="left_hip_roll_link"/>
    <child link="left_thigh_link"/>
    <origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="200" velocity="3.0"/>
    <dynamics damping="0.5" friction="0.0"/>
  </joint>

  <link name="left_thigh_link">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.4" radius="0.08"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.4" radius="0.08"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="3.0"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left leg - knee -->
  <joint name="left_knee_joint" type="revolute">
    <parent link="left_thigh_link"/>
    <child link="left_shin_link"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.36" effort="200" velocity="3.0"/>
    <dynamics damping="0.4" friction="0.0"/>
  </joint>

  <link name="left_shin_link">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.4" radius="0.07"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.4" radius="0.07"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="2.5"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left foot -->
  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_shin_link"/>
    <child link="left_foot_link"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="3.0"/>
    <dynamics damping="0.2" friction="0.0"/>
  </joint>

  <link name="left_foot_link">
    <visual>
      <origin xyz="0.05 0 -0.02" rpy="0 0 0"/>
      <geometry>
        <box size="0.18 0.1 0.04"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0.05 0 -0.02" rpy="0 0 0"/>
      <geometry>
        <box size="0.18 0.1 0.04"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0.05 0 -0.02" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right leg (mirrored) -->
  <joint name="right_hip_yaw_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_hip_yaw_link"/>
    <origin xyz="0.0 -0.08 0.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="200" velocity="3.0"/>
    <dynamics damping="0.5" friction="0.0"/>
  </joint>

  <link name="right_hip_yaw_link">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_hip_roll_joint" type="revolute">
    <parent link="right_hip_yaw_link"/>
    <child link="right_hip_roll_link"/>
    <origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="200" velocity="3.0"/>
    <dynamics damping="0.5" friction="0.0"/>
  </joint>

  <link name="right_hip_roll_link">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <joint name="right_hip_pitch_joint" type="revolute">
    <parent link="right_hip_roll_link"/>
    <child link="right_thigh_link"/>
    <origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="200" velocity="3.0"/>
    <dynamics damping="0.5" friction="0.0"/>
  </joint>

  <link name="right_thigh_link">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.4" radius="0.08"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.4" radius="0.08"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="3.0"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_thigh_link"/>
    <child link="right_shin_link"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.36" effort="200" velocity="3.0"/>
    <dynamics damping="0.4" friction="0.0"/>
  </joint>

  <link name="right_shin_link">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.4" radius="0.07"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <capsule length="0.4" radius="0.07"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <mass value="2.5"/>
      <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_shin_link"/>
    <child link="right_foot_link"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="3.0"/>
    <dynamics damping="0.2" friction="0.0"/>
  </joint>

  <link name="right_foot_link">
    <visual>
      <origin xyz="0.05 0 -0.02" rpy="0 0 0"/>
      <geometry>
        <box size="0.18 0.1 0.04"/>
      </geometry>
      <material name="light_grey"/>
    </visual>
    <collision>
      <origin xyz="0.05 0 -0.02" rpy="0 0 0"/>
      <geometry>
        <box size="0.18 0.1 0.04"/>
      </geometry>
    </collision>
    <inertial>
      <origin xyz="0.05 0 -0.02" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.002" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- IMU sensor -->
  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu_link"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
  </joint>

  <link name="imu_link">
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/>
    </inertial>
  </link>

  <!-- Camera sensor -->
  <joint name="camera_joint" type="fixed">
    <parent link="head_link"/>
    <child link="camera_link"/>
    <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
  </joint>

  <link name="camera_link">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
  </link>

</robot>
```

## Humanoid Robot Kinematics

### Forward Kinematics

Forward kinematics in URDF defines how joint angles translate to the position and orientation of end-effectors. For humanoid robots, this is crucial for understanding how arm and leg movements affect the position of hands and feet.

The kinematic chain in URDF is defined by the joint connections. Each joint specifies a transformation from the parent link to the child link based on the joint type and current joint value. The complete kinematic chain from the base to any end-effector can be computed by multiplying all the individual transformations.

### Inverse Kinematics Considerations

While URDF itself doesn't implement inverse kinematics, it provides the kinematic structure needed for IK solvers. For humanoid robots, common IK applications include:

- Foot placement for walking
- Hand positioning for manipulation
- Balance control through center of mass adjustment
- Whole-body IK for coordinated motion

### Center of Mass and Stability

For humanoid robots, the center of mass (CoM) is critical for stability analysis. The CoM position changes dynamically as the robot moves its limbs. Proper URDF modeling includes accurate mass and inertia properties for each link to enable accurate CoM calculations.

The Zero-Moment Point (ZMP) and Capture Point are important concepts in humanoid stability that rely on accurate URDF models with proper dynamic properties.

## Creating Humanoid Joint Structures

### Joint Types for Humanoid Robots

Different joint types serve specific purposes in humanoid robot modeling:

1. **Revolute Joints**: Most common for humanoid robots, allowing rotation around a single axis. Used for shoulders, elbows, hips, knees, etc.

2. **Continuous Joints**: Similar to revolute but without limits, useful for wheels or continuously rotating parts.

3. **Prismatic Joints**: Allow linear motion, used for telescoping mechanisms or linear actuators.

4. **Fixed Joints**: Connect links rigidly without allowing relative motion, useful for mounting sensors or combining parts.

5. **Floating Joints**: Allow 6 degrees of freedom, used for base links in some simulation scenarios.

### Joint Limits and Safety

Joint limits are critical for humanoid robots to:
- Prevent self-collision
- Ensure realistic motion ranges
- Protect physical hardware
- Maintain structural integrity

For humanoid joints, typical limits include:
- Shoulder joints: ±150° to ±180° depending on axis
- Elbow joints: 0° to 160° (flexion only)
- Hip joints: ±45° to ±90° depending on axis
- Knee joints: 0° to 150° (flexion only)
- Ankle joints: ±30° to ±45°

## Visual and Collision Models

### Visual Elements

Visual elements define how the robot appears in simulation and visualization tools:

- **Geometry**: Can be meshes (STL, DAE, OBJ), primitive shapes (box, cylinder, sphere), or capsules
- **Materials**: Define color and appearance properties
- **Origin**: Position and orientation of the visual element relative to the link frame

For humanoid robots, visual elements should be lightweight for real-time rendering while maintaining recognizability.

### Collision Elements

Collision elements define the physical boundaries for physics simulation:

- **Geometry**: Similar to visual elements but often simplified for performance
- **Origin**: Position and orientation of the collision element relative to the link frame
- **Simplified shapes**: Often use primitive shapes or convex hulls instead of complex meshes

Collision models should be conservative (enveloping the actual geometry) to prevent phantom collisions while maintaining simulation accuracy.

## Advanced URDF Features for Humanoid Robotics

### Xacro Macros

Xacro (XML Macros) allows for parameterization and reusability in URDF models:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Define properties -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="robot_name" value="autonomous_humanoid" />

  <!-- Macro for creating a humanoid joint -->
  <xacro:macro name="humanoid_joint" params="name type parent child origin_xyz axis_xyz lower upper effort velocity damping friction">
    <joint name="${name}_joint" type="${type}">
      <parent link="${parent}_link"/>
      <child link="${child}_link"/>
      <origin xyz="${origin_xyz}" rpy="0 0 0"/>
      <axis xyz="${axis_xyz}"/>
      <limit lower="${lower}" upper="${upper}" effort="${effort}" velocity="${velocity}"/>
      <dynamics damping="${damping}" friction="${friction}"/>
    </joint>
  </xacro:macro>

  <!-- Macro for creating a humanoid link -->
  <xacro:macro name="humanoid_link" params="name mass ixx ixy ixz iyy iyz izz visual_geometry collision_geometry origin_xyz">
    <link name="${name}_link">
      <visual>
        <origin xyz="${origin_xyz}" rpy="0 0 0"/>
        <geometry>
          ${visual_geometry}
        </geometry>
        <material name="light_grey"/>
      </visual>
      <collision>
        <origin xyz="${origin_xyz}" rpy="0 0 0"/>
        <geometry>
          ${collision_geometry}
        </geometry>
      </collision>
      <inertial>
        <origin xyz="${origin_xyz}" rpy="0 0 0"/>
        <mass value="${mass}"/>
        <inertia ixx="${ixx}" ixy="${ixy}" ixz="${ixz}" iyy="${iyy}" iyz="${iyz}" izz="${izz}"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Use the macros to create robot structure -->
  <xacro:humanoid_link name="base" mass="10.0" ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.2"
                      visual_geometry="<capsule length='0.8' radius='0.15'/>"
                      collision_geometry="<capsule length='0.8' radius='0.15'/>"
                      origin_xyz="0 0 0.5"/>

  <!-- Define joints using the macro -->
  <xacro:humanoid_joint name="neck" type="revolute" parent="base" child="head"
                       origin_xyz="0 0 0.95" axis_xyz="0 1 0"
                       lower="-0.785" upper="0.785" effort="50" velocity="3.0"
                       damping="0.1" friction="0.0"/>

</robot>
```

### Transmission Elements

Transmissions define how actuators connect to joints, which is important for simulation:

```xml
<transmission name="left_hip_pitch_transmission">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_hip_pitch_joint">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_hip_pitch_motor">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Integration with Simulation Environments

### Gazebo-Specific Elements

Gazebo simulation requires additional elements in the URDF:

```xml
<gazebo reference="base_link">
  <material>Gazebo/Grey</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
</gazebo>

<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/autonomous_humanoid</robotNamespace>
  </plugin>
</gazebo>

<!-- Sensor definitions -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <visualize>true</visualize>
  </sensor>
</gazebo>
```

### NVIDIA Isaac Sim Integration

For NVIDIA Isaac Sim, the URDF needs to be compatible with the Omniverse platform:

```json
{
  "isaac_sim_config": {
    "robot_name": "autonomous_humanoid",
    "urdf_path": "/path/to/robot.urdf",
    "scale": [1.0, 1.0, 1.0],
    "position": [0.0, 0.0, 1.0],
    "orientation": [0.0, 0.0, 0.0, 1.0],
    "articulation": {
      "joints": {
        "left_hip_pitch_joint": {
          "drive_type": "force",
          "stiffness": 1e7,
          "damping": 1e5,
          "max_force": 1000.0,
          "max_velocity": 10.0
        }
      }
    },
    "collision_approximation": "convexDecomposition",
    "visual_materials": {
      "default": {
        "albedo": [0.5, 0.5, 0.5],
        "metallic": 0.0,
        "roughness": 0.9
      }
    }
  }
}
```

## Best Practices for Humanoid URDF Models

### 1. Proper Inertial Properties
- Calculate realistic mass and inertia tensors for each link
- Use CAD tools to compute accurate inertial properties
- Verify that the center of mass is correctly positioned
- Use consistent units (SI units: kg, m, kg*m²)

### 2. Appropriate Joint Limits
- Define realistic joint limits based on mechanical constraints
- Include safety margins in the limits
- Consider the effect of soft limits in controllers
- Validate that joint limits prevent self-collision

### 3. Collision Detection
- Use simplified collision geometries for performance
- Ensure collision elements fully encompass the visual model
- Use convex hulls for complex shapes
- Test collision detection thoroughly in simulation

### 4. Visualization
- Use lightweight visual meshes for real-time rendering
- Organize mesh files in a logical directory structure
- Use appropriate textures and materials
- Consider using primitive shapes for simple parts

### 5. Performance Optimization
- Minimize the number of links and joints where possible
- Use fixed joints to combine parts that don't move relative to each other
- Simplify collision models for dynamic simulation
- Organize URDF into modular xacro files

## Validation and Testing

### URDF Validation Tools

Use ROS tools to validate your URDF:

```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Generate and view the kinematic tree
urdf_to_graphiz /path/to/robot.urdf

# Visualize in Rviz
roslaunch urdf_tutorial display.launch model:=/path/to/robot.urdf
```

### Kinematic Validation

Verify that your humanoid robot model has the expected kinematic properties:
- Check that joint limits are appropriate
- Verify that the kinematic chain is correctly defined
- Test that inverse kinematics solvers can work with the model
- Ensure that the model can achieve expected poses

### Dynamic Validation

Test dynamic properties:
- Verify that mass properties are realistic
- Check that the center of mass is in the expected location
- Test that the robot is stable in simulation
- Validate that joint torques are appropriate for the intended motion

## Summary

URDF modeling for humanoid robots requires careful attention to kinematic structure, dynamic properties, and simulation compatibility. A well-designed URDF model serves as the foundation for simulation, visualization, motion planning, and control. The examples provided demonstrate advanced techniques for creating realistic humanoid robot models with proper joint structures, visual and collision elements, and integration with simulation environments.

Proper URDF modeling is essential for humanoid robot simulation and control, enabling the development and testing of complex behaviors before deployment on physical hardware. The use of advanced features like Xacro macros, proper inertial properties, and simulation-specific elements ensures that the model will perform well in both simulation and real-world applications.