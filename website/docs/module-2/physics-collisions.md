---
sidebar_position: 1
---

# Physics and Collisions: Advanced Simulation for Digital Twins of Humanoid Robots

## Introduction to Physics Simulation for Digital Twins

Physics simulation in digital twin environments is fundamental for creating realistic representations of humanoid robots. The digital twin must accurately reflect the physical behavior of the real robot, including its interactions with the environment, to enable effective testing, validation, and development of control algorithms. For humanoid robots, which have complex multi-link structures with numerous degrees of freedom, physics simulation becomes particularly challenging due to the need to accurately model contacts, collisions, and dynamic interactions.

Digital twin physics simulation serves multiple purposes:
- **Validation**: Testing control algorithms in a realistic simulated environment before deployment
- **Safety**: Identifying potential issues without risking physical hardware
- **Optimization**: Tuning parameters and improving performance in a controlled environment
- **Training**: Developing AI models and machine learning algorithms

## Theoretical Foundation of Physics Simulation

### Physics Engine Fundamentals

Physics engines for robotics simulation typically implement the following core components:

#### Rigid Body Dynamics
The simulation of rigid bodies is governed by Newton-Euler equations of motion. For each link in a humanoid robot, the physics engine calculates:
- Translational motion: F = ma
- Rotational motion: τ = Iα + ω × (Iω)

Where F is force, m is mass, a is acceleration, τ is torque, I is the inertia tensor, α is angular acceleration, and ω is angular velocity.

#### Constraint Solving
Joints in humanoid robots impose constraints between links. The physics engine must solve these constraints while maintaining:
- Joint limits
- Closed-loop kinematic chains
- Contact constraints
- Actuator forces and torques

#### Time Integration
Physics engines use numerical integration methods to advance the simulation through time. Common approaches include:
- Explicit Euler (fast but unstable)
- Implicit Euler (stable but computationally expensive)
- Runge-Kutta methods (balanced accuracy and stability)
- Symplectic integrators (energy-conserving)

### Collision Detection Theory

Collision detection in physics engines typically involves a multi-stage process:

#### Broad Phase
Quickly identifies pairs of objects that might be colliding using spatial partitioning techniques:
- Axis-Aligned Bounding Box (AABB) trees
- Spatial hashing
- Grid-based partitioning

#### Narrow Phase
Precisely determines if and where collisions occur between potential pairs:
- GJK algorithm for convex shapes
- SAT (Separating Axis Theorem) for polyhedra
- Ray casting for specific intersection tests

#### Continuous Collision Detection (CCD)
Prevents tunneling effects by detecting collisions that might occur between discrete time steps:
- Conservative advancement
- Speculative advancement
- Posteriori collision detection

## Advanced Physics Engine Implementation

### NVIDIA PhysX Configuration for Humanoid Simulation

```cpp
#include <PxPhysicsAPI.h>
#include <extensions/PxExtensions.h>
#include <vehicle/PxVehicleAPI.h>
#include <iostream>

using namespace physx;

class HumanoidPhysicsSimulator {
private:
    PxFoundation* foundation;
    PxPhysics* physics;
    PxScene* scene;
    PxMaterial* material;

    // Humanoid robot actors
    std::vector<PxRigidDynamic*> humanoid_links;
    std::vector<PxJoint*> humanoid_joints;

    // Simulation parameters
    PxReal timestep;
    PxU32 substeps;
    PxVec3 gravity;

public:
    HumanoidPhysicsSimulator() : timestep(0.001f), substeps(1), gravity(PxVec3(0.0f, -9.81f, 0.0f)) {
        // Foundation setup
        foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, error_callback);

        // Physics creation
        physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale(), true, nullptr);

        // Extensions initialization
        PxInitExtensions(*physics);

        // Material for contacts
        material = physics->createMaterial(0.5f, 0.5f, 0.1f); // static, dynamic, restitution

        // Scene creation with advanced parameters
        PxSceneDesc scene_desc(physics->getTolerancesScale());
        scene_desc.gravity = gravity;

        // Advanced solver settings for humanoid simulation
        scene_desc.solverType = PxSolverType::eTGS;  // Temporal Gauss-Seidel solver
        scene_desc.broadPhaseType = PxBroadPhaseType::eABP;  // Accelerated Broad Phase
        scene_desc.simulationEventCallback = nullptr;  // Event callbacks for contacts
        scene_desc.flags |= PxSceneFlag::eENABLE_CCD;  // Enable Continuous Collision Detection
        scene_desc.flags |= PxSceneFlag::eENABLE_PCM;  // Enable Projection Contact Model

        // Solver parameters optimized for humanoid robots
        scene_desc.solverIterationCounts.minPositionIters = 4;   // Position iterations
        scene_desc.solverIterationCounts.minVelocityIters = 1;   // Velocity iterations
        scene_desc.solverOffsetSlop = 0.001f;                   // Contact offset slop

        scene = physics->createScene(scene_desc);

        // CPU dispatcher for multithreading
        cpu_dispatcher = PxDefaultCpuDispatcherCreate(4); // 4 threads
        scene->setCpuDispatcher(cpu_dispatcher);

        std::cout << "Humanoid Physics Simulator initialized with advanced parameters" << std::endl;
    }

    ~HumanoidPhysicsSimulator() {
        if (scene) scene->release();
        if (cpu_dispatcher) cpu_dispatcher->release();
        if (material) material->release();
        if (physics) PxCloseExtensions();
        if (physics) physics->release();
        if (foundation) foundation->release();
    }

    void createHumanoidRobot() {
        // Create base/torso link
        PxTransform base_transform(PxVec3(0.0f, 1.0f, 0.0f)); // Start slightly above ground
        PxRigidDynamic* base_link = createCapsuleActor(base_transform, 0.15f, 0.4f, 10.0f); // Torso
        humanoid_links.push_back(base_link);

        // Create head link
        PxTransform head_transform(PxVec3(0.0f, 0.0f, 0.5f)); // Relative to torso
        PxRigidDynamic* head_link = createSphereActor(head_transform.transform(base_transform), 0.12f, 2.0f);
        humanoid_links.push_back(head_link);

        // Create left arm
        createArmChain(true, base_transform);

        // Create right arm (mirrored)
        createArmChain(false, base_transform);

        // Create left leg
        createLegChain(true, base_transform);

        // Create right leg (mirrored)
        createLegChain(false, base_transform);

        // Add all actors to scene
        for (auto& link : humanoid_links) {
            scene->addActor(*link);
        }

        std::cout << "Humanoid robot model created with " << humanoid_links.size() << " links" << std::endl;
    }

    PxRigidDynamic* createCapsuleActor(const PxTransform& transform, PxReal radius, PxReal half_height, PxReal mass) {
        PxRigidDynamic* actor = PxCreateDynamic(*physics, transform, PxSphereGeometry(radius), *material, 1.0f);

        // Set capsule geometry (approximate with sphere for now, in real implementation would use proper capsule)
        PxShape* shape;
        PxU32 nb_shapes = actor->getNbShapes();
        actor->getShapes(&shape, nb_shapes);

        // Create proper capsule geometry
        PxTransform local_transform(PxIdentity);
        PxReal density = mass / (PxPi * radius * radius * 2.0f * half_height + 4.0f/3.0f * PxPi * radius * radius * radius); // Approximate density

        // Recreate with proper mass and geometry
        actor->setMass(mass);
        actor->setCMassLocalPose(PxTransform(PxVec3(0.0f, 0.0f, 0.0f)));

        // Set appropriate moment of inertia for capsule
        PxVec3 moi(0.25f * mass * (radius*radius + half_height*half_height/3.0f),
                   0.25f * mass * (radius*radius + half_height*half_height/3.0f),
                   0.5f * mass * radius * radius);
        actor->setMassSpaceInertiaTensor(moi);

        return actor;
    }

    PxRigidDynamic* createSphereActor(const PxTransform& transform, PxReal radius, PxReal mass) {
        PxRigidDynamic* actor = PxCreateDynamic(*physics, transform, PxSphereGeometry(radius), *material, 1.0f);
        actor->setMass(mass);

        // Moment of inertia for sphere: (2/5) * m * r²
        PxReal moi_value = 0.4f * mass * radius * radius;
        actor->setMassSpaceInertiaTensor(PxVec3(moi_value, moi_value, moi_value));

        return actor;
    }

    void createArmChain(bool is_left, const PxTransform& base_transform) {
        PxReal sign = is_left ? 1.0f : -1.0f;
        PxVec3 offset(0.0f, 0.15f * sign, 0.7f); // Shoulder position relative to base

        // Create shoulder joints and links
        PxTransform shoulder_transform = base_transform.transform(offset);
        PxRigidDynamic* shoulder_link = createCapsuleActor(shoulder_transform, 0.06f, 0.15f, 1.5f);
        humanoid_links.push_back(shoulder_link);

        // Upper arm
        PxTransform upper_arm_transform = shoulder_transform.transform(PxVec3(0.0f, 0.0f, -0.3f));
        PxRigidDynamic* upper_arm_link = createCapsuleActor(upper_arm_transform, 0.06f, 0.15f, 1.5f);
        humanoid_links.push_back(upper_arm_link);

        // Lower arm
        PxTransform lower_arm_transform = upper_arm_transform.transform(PxVec3(0.0f, 0.0f, -0.3f));
        PxRigidDynamic* lower_arm_link = createCapsuleActor(lower_arm_transform, 0.05f, 0.15f, 1.0f);
        humanoid_links.push_back(lower_arm_link);

        // Hand
        PxTransform hand_transform = lower_arm_transform.transform(PxVec3(0.0f, 0.0f, -0.3f));
        PxRigidDynamic* hand_link = createSphereActor(hand_transform, 0.05f, 0.3f);
        humanoid_links.push_back(hand_link);

        // Create joints between arm segments (in real implementation would use proper joint constraints)
        // This is a simplified representation
    }

    void createLegChain(bool is_left, const PxTransform& base_transform) {
        PxReal sign = is_left ? 1.0f : -1.0f;
        PxVec3 offset(0.0f, 0.08f * sign, 0.0f); // Hip position relative to base

        // Create hip joints and links
        PxTransform hip_transform = base_transform.transform(offset);
        PxRigidDynamic* hip_link = createCapsuleActor(hip_transform, 0.08f, 0.05f, 0.5f);
        humanoid_links.push_back(hip_link);

        // Thigh
        PxTransform thigh_transform = hip_transform.transform(PxVec3(0.0f, 0.0f, -0.4f));
        PxRigidDynamic* thigh_link = createCapsuleActor(thigh_transform, 0.08f, 0.2f, 3.0f);
        humanoid_links.push_back(thigh_link);

        // Shin
        PxTransform shin_transform = thigh_transform.transform(PxVec3(0.0f, 0.0f, -0.4f));
        PxRigidDynamic* shin_link = createCapsuleActor(shin_transform, 0.07f, 0.2f, 2.5f);
        humanoid_links.push_back(shin_link);

        // Foot
        PxTransform foot_transform = shin_transform.transform(PxVec3(0.05f, 0.0f, -0.4f));
        PxRigidDynamic* foot_link = createBoxActor(foot_transform, PxVec3(0.09f, 0.05f, 0.02f), 1.0f);
        humanoid_links.push_back(foot_link);
    }

    PxRigidDynamic* createBoxActor(const PxTransform& transform, const PxVec3& dimensions, PxReal mass) {
        PxRigidDynamic* actor = PxCreateDynamic(*physics, transform, PxBoxGeometry(dimensions), *material, 1.0f);
        actor->setMass(mass);

        // Moment of inertia for box
        PxReal dx = 2.0f * dimensions.x, dy = 2.0f * dimensions.y, dz = 2.0f * dimensions.z;
        PxVec3 moi(mass/12.0f * (dy*dy + dz*dz),
                   mass/12.0f * (dx*dx + dz*dz),
                   mass/12.0f * (dx*dx + dy*dy));
        actor->setMassSpaceInertiaTensor(moi);

        return actor;
    }

    void simulateStep() {
        scene->simulate(timestep / substeps);
        scene->fetchResults(true); // Block until simulation completes
    }

    void setRobotConfiguration(const std::vector<PxReal>& joint_angles) {
        // In a real implementation, this would update joint positions
        // For now, this is a placeholder for setting robot pose
    }

    void applyControlInputs(const std::vector<PxVec3>& forces, const std::vector<PxVec3>& torques) {
        // Apply control forces and torques to the robot links
        for (size_t i = 0; i < humanoid_links.size() && i < forces.size(); ++i) {
            humanoid_links[i]->addForce(forces[i], PxForceMode::eFORCE);
            humanoid_links[i]->addTorque(torques[i]);
        }
    }

    std::vector<PxTransform> getRobotStates() {
        std::vector<PxTransform> states;
        for (auto& link : humanoid_links) {
            states.push_back(link->getGlobalPose());
        }
        return states;
    }

    void printSimulationInfo() {
        std::cout << "=== Physics Simulation Info ===" << std::endl;
        std::cout << "Timestep: " << timestep << "s" << std::endl;
        std::cout << "Substeps: " << substeps << std::endl;
        std::cout << "Gravity: [" << gravity.x << ", " << gravity.y << ", " << gravity.z << "]" << std::endl;
        std::cout << "Number of actors in scene: " << scene->getNbActors(PxActorTypeFlag::eRIGID_DYNAMIC | PxActorTypeFlag::eRIGID_STATIC) << std::endl;
        std::cout << "Number of humanoid links: " << humanoid_links.size() << std::endl;
        std::cout << "=============================" << std::endl;
    }

private:
    PxDefaultAllocator allocator;
    PxDefaultErrorCallback error_callback;
    PxCpuDispatcher* cpu_dispatcher;
};

// Example usage
int main() {
    HumanoidPhysicsSimulator simulator;
    simulator.createHumanoidRobot();
    simulator.printSimulationInfo();

    // Main simulation loop
    for (int i = 0; i < 1000; ++i) {
        simulator.simulateStep();

        if (i % 100 == 0) {
            std::cout << "Simulation step: " << i << std::endl;
        }
    }

    return 0;
}
```

## Collision Detection Algorithms and Implementation

### Advanced Collision Detection for Humanoid Robots

```python
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class CollisionType(Enum):
    """Types of collisions that can occur in humanoid simulation."""
    SELF_COLLISION = "self_collision"
    ENVIRONMENT_COLLISION = "environment_collision"
    GROUND_COLLISION = "ground_collision"
    CONTACT_POINT = "contact_point"

@dataclass
class CollisionInfo:
    """Information about a detected collision."""
    link_a: str
    link_b: str
    position: np.ndarray  # Collision point in world coordinates
    normal: np.ndarray    # Normal vector pointing from A to B
    depth: float          # Penetration depth
    collision_type: CollisionType

class CollisionDetector:
    """Advanced collision detection system for humanoid robots."""

    def __init__(self, robot_links: List[str]):
        self.robot_links = robot_links
        self.collision_pairs = self._generate_collision_pairs()
        self.contact_points = []

        # Spatial partitioning for broad phase collision detection
        self.spatial_grid = {}
        self.grid_cell_size = 0.5  # meters

        # Collision detection parameters
        self.collision_margin = 0.001  # meters
        self.contact_distance = 0.02   # Distance to consider contact

        print(f"Initialized collision detector for {len(robot_links)} links")
        print(f"Generated {len(self.collision_pairs)} collision pairs")

    def _generate_collision_pairs(self) -> List[Tuple[str, str]]:
        """Generate all possible collision pairs, excluding impossible ones."""
        pairs = []

        # Generate all combinations of links
        for i, link_a in enumerate(self.robot_links):
            for j, link_b in enumerate(self.robot_links):
                if i >= j:  # Avoid duplicates and self-collisions
                    continue

                # Skip adjacent joints that are physically connected
                # This is a simplified check - in reality, you'd have a kinematic tree
                if self._are_adjacent_links(link_a, link_b):
                    continue

                pairs.append((link_a, link_b))

        return pairs

    def _are_adjacent_links(self, link_a: str, link_b: str) -> bool:
        """Check if two links are directly connected by a joint."""
        # Simplified adjacency check based on naming convention
        # In a real implementation, this would use the kinematic tree
        adjacent_pairs = [
            ('base_link', 'head_link'),
            ('base_link', 'left_shoulder_link'), ('base_link', 'right_shoulder_link'),
            ('left_shoulder_link', 'left_upper_arm_link'), ('right_shoulder_link', 'right_upper_arm_link'),
            ('left_upper_arm_link', 'left_lower_arm_link'), ('right_upper_arm_link', 'right_lower_arm_link'),
            ('left_lower_arm_link', 'left_hand_link'), ('right_lower_arm_link', 'right_hand_link'),
            ('base_link', 'left_hip_link'), ('base_link', 'right_hip_link'),
            ('left_hip_link', 'left_thigh_link'), ('right_hip_link', 'right_thigh_link'),
            ('left_thigh_link', 'left_shin_link'), ('right_thigh_link', 'right_shin_link'),
            ('left_shin_link', 'left_foot_link'), ('right_shin_link', 'right_foot_link')
        ]

        return (link_a, link_b) in adjacent_pairs or (link_b, link_a) in adjacent_pairs

    def update_robot_pose(self, link_poses: dict, link_geometries: dict):
        """Update the collision detection system with current robot pose."""
        # Update spatial grid with current link positions
        self._update_spatial_grid(link_poses)

        # Detect collisions
        collisions = self._detect_collisions(link_poses, link_geometries)

        return collisions

    def _update_spatial_grid(self, link_poses: dict):
        """Update the spatial partitioning grid."""
        self.spatial_grid.clear()

        for link_name, pose in link_poses.items():
            # Calculate which grid cells this link occupies
            position = pose['position']
            bounding_radius = self._estimate_bounding_radius(link_poses[link_name]['geometry'])

            min_cell = self._world_to_grid(position - bounding_radius)
            max_cell = self._world_to_grid(position + bounding_radius)

            # Add link to all occupied cells
            for x in range(min_cell[0], max_cell[0] + 1):
                for y in range(min_cell[1], max_cell[1] + 1):
                    for z in range(min_cell[2], max_cell[2] + 1):
                        cell_key = (x, y, z)
                        if cell_key not in self.spatial_grid:
                            self.spatial_grid[cell_key] = []
                        self.spatial_grid[cell_key].append(link_name)

    def _world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int, int]:
        """Convert world coordinates to grid cell coordinates."""
        x = int(world_pos[0] / self.grid_cell_size)
        y = int(world_pos[1] / self.grid_cell_size)
        z = int(world_pos[2] / self.grid_cell_size)
        return (x, y, z)

    def _estimate_bounding_radius(self, geometry: dict) -> float:
        """Estimate bounding radius for a geometry."""
        if geometry['type'] == 'sphere':
            return geometry['radius']
        elif geometry['type'] == 'capsule':
            return max(geometry['radius'], geometry['length'] / 2.0)
        elif geometry['type'] == 'box':
            return np.linalg.norm(geometry['dimensions']) / 2.0
        else:
            return 0.1  # Default radius

    def _detect_collisions(self, link_poses: dict, link_geometries: dict) -> List[CollisionInfo]:
        """Detect collisions between links using broad and narrow phase."""
        potential_collisions = self._broad_phase_collision_detection()
        confirmed_collisions = []

        for link_a, link_b in potential_collisions:
            collision_info = self._narrow_phase_collision_detection(
                link_a, link_b, link_poses, link_geometries
            )

            if collision_info:
                confirmed_collisions.append(collision_info)

        return confirmed_collisions

    def _broad_phase_collision_detection(self) -> List[Tuple[str, str]]:
        """Broad phase collision detection using spatial partitioning."""
        overlapping_pairs = []

        # Check each cell for potential overlaps
        for cell_key, cell_links in self.spatial_grid.items():
            if len(cell_links) < 2:
                continue

            # Check all combinations within this cell
            for i, link_a in enumerate(cell_links):
                for j, link_b in enumerate(cell_links):
                    if i >= j:
                        continue

                    # Verify this pair should be checked
                    if (link_a, link_b) in self.collision_pairs or (link_b, link_a) in self.collision_pairs:
                        overlapping_pairs.append((link_a, link_b))

        return overlapping_pairs

    def _narrow_phase_collision_detection(
        self,
        link_a: str,
        link_b: str,
        link_poses: dict,
        link_geometries: dict
    ) -> Optional[CollisionInfo]:
        """Narrow phase collision detection between two specific links."""
        pose_a = link_poses[link_a]
        pose_b = link_poses[link_b]
        geom_a = link_geometries[link_a]
        geom_b = link_geometries[link_b]

        # Perform collision detection based on geometry types
        if geom_a['type'] == 'sphere' and geom_b['type'] == 'sphere':
            return self._sphere_sphere_collision(pose_a, geom_a, pose_b, geom_b, link_a, link_b)
        elif geom_a['type'] == 'capsule' and geom_b['type'] == 'capsule':
            return self._capsule_capsule_collision(pose_a, geom_a, pose_b, geom_b, link_a, link_b)
        elif geom_a['type'] == 'box' and geom_b['type'] == 'box':
            return self._box_box_collision(pose_a, geom_a, pose_b, geom_b, link_a, link_b)
        elif (geom_a['type'] == 'capsule' and geom_b['type'] == 'sphere') or \
             (geom_a['type'] == 'sphere' and geom_b['type'] == 'capsule'):
            return self._capsule_sphere_collision(pose_a, geom_a, pose_b, geom_b, link_a, link_b)
        else:
            # For mixed types, approximate with spheres
            return self._sphere_sphere_collision(pose_a, geom_a, pose_b, geom_b, link_a, link_b)

    def _sphere_sphere_collision(
        self,
        pose_a: dict,
        geom_a: dict,
        pose_b: dict,
        geom_b: dict,
        link_a: str,
        link_b: str
    ) -> Optional[CollisionInfo]:
        """Detect collision between two spheres."""
        pos_a = pose_a['position']
        pos_b = pose_b['position']
        radius_a = geom_a['radius']
        radius_b = geom_b['radius']

        distance = np.linalg.norm(pos_b - pos_a)
        min_distance = radius_a + radius_b + self.collision_margin

        if distance < min_distance:
            # Calculate collision point and normal
            normal = (pos_b - pos_a) / distance if distance > 0 else np.array([0, 0, 1])
            penetration_depth = min_distance - distance

            # Collision point is along the line connecting centers
            collision_point = pos_a + normal * radius_a

            return CollisionInfo(
                link_a=link_a,
                link_b=link_b,
                position=collision_point,
                normal=normal,
                depth=penetration_depth,
                collision_type=CollisionType.SELF_COLLISION
            )

        return None

    def _capsule_capsule_collision(
        self,
        pose_a: dict,
        geom_a: dict,
        pose_b: dict,
        geom_b: dict,
        link_a: str,
        link_b: str
    ) -> Optional[CollisionInfo]:
        """Detect collision between two capsules."""
        # Get capsule parameters
        pos_a = pose_a['position']
        rot_a = pose_a['orientation']  # As quaternion [w, x, y, z]
        radius_a = geom_a['radius']
        length_a = geom_a['length']

        pos_b = pose_b['position']
        rot_b = pose_b['orientation']
        radius_b = geom_b['radius']
        length_b = geom_b['length']

        # Convert quaternions to rotation matrices
        rot_mat_a = self._quaternion_to_matrix(rot_a)
        rot_mat_b = self._quaternion_to_matrix(rot_b)

        # Calculate capsule axis directions
        axis_a = rot_mat_a @ np.array([0, 0, 1])  # Capsule oriented along Z-axis in local frame
        axis_b = rot_mat_b @ np.array([0, 0, 1])

        # Calculate capsule endpoints
        half_len_a = length_a / 2.0
        half_len_b = length_b / 2.0

        a_start = pos_a - axis_a * half_len_a
        a_end = pos_a + axis_a * half_len_a

        b_start = pos_b - axis_b * half_len_b
        b_end = pos_b + axis_b * half_len_b

        # Find closest points on both line segments
        closest_a, closest_b = self._closest_points_on_segments(
            a_start, a_end, b_start, b_end
        )

        # Calculate distance between closest points
        distance = np.linalg.norm(closest_b - closest_a)
        min_distance = radius_a + radius_b + self.collision_margin

        if distance < min_distance:
            normal = (closest_b - closest_a) / distance if distance > 0 else np.array([0, 0, 1])
            penetration_depth = min_distance - distance

            # Collision point is the midpoint between closest points
            collision_point = (closest_a + closest_b) / 2.0

            return CollisionInfo(
                link_a=link_a,
                link_b=link_b,
                position=collision_point,
                normal=normal,
                depth=penetration_depth,
                collision_type=CollisionType.SELF_COLLISION
            )

        return None

    def _closest_points_on_segments(
        self,
        a_start: np.ndarray,
        a_end: np.ndarray,
        b_start: np.ndarray,
        b_end: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find closest points between two line segments."""
        # Vector representations of segments
        d1 = a_end - a_start
        d2 = b_end - b_start
        r = a_start - b_start

        # Calculate dot products
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)

        # Check if segments are parallel
        if a <= 1e-10 and e <= 1e-10:
            return a_start, b_start

        if a <= 1e-10:
            t = np.clip(f / e, 0, 1)
            s = 0
        elif e <= 1e-10:
            s = 0
            t = np.clip(f / e, 0, 1)
        else:
            c = np.dot(d1, r)

            # Calculate parameters
            b = np.dot(d1, d2)
            denom = a * e - b * b

            if abs(denom) > 1e-10:
                s = np.clip((b * f - c * e) / denom, 0, 1)
                t = np.clip((a * f - b * c) / denom, 0, 1)
            else:
                s = 0
                t = np.clip(f / e, 0, 1)

        # Calculate closest points
        closest_a = a_start + s * d1
        closest_b = b_start + t * d2

        return closest_a, closest_b

    def _quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q

        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        if norm > 0:
            w, x, y, z = w/norm, x/norm, y/norm, z/norm

        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

    def _capsule_sphere_collision(
        self,
        pose_capsule: dict,
        geom_capsule: dict,
        pose_sphere: dict,
        geom_sphere: dict,
        link_capsule: str,
        link_sphere: str
    ) -> Optional[CollisionInfo]:
        """Detect collision between a capsule and a sphere."""
        # Get parameters
        pos_capsule = pose_capsule['position']
        rot_capsule = pose_capsule['orientation']
        radius_capsule = geom_capsule['radius']
        length_capsule = geom_capsule['length']

        pos_sphere = pose_sphere['position']
        radius_sphere = geom_sphere['radius']

        # Calculate capsule axis
        rot_mat = self._quaternion_to_matrix(rot_capsule)
        axis = rot_mat @ np.array([0, 0, 1])
        half_len = length_capsule / 2.0

        # Capsule endpoints
        start = pos_capsule - axis * half_len
        end = pos_capsule + axis * half_len

        # Find closest point on capsule segment to sphere center
        closest_point = self._closest_point_on_segment(start, end, pos_sphere)

        # Calculate distance
        distance = np.linalg.norm(pos_sphere - closest_point)
        min_distance = radius_capsule + radius_sphere + self.collision_margin

        if distance < min_distance:
            normal = (pos_sphere - closest_point) / distance if distance > 0 else np.array([0, 0, 1])
            penetration_depth = min_distance - distance

            # Collision point is along the surface
            collision_point = closest_point + normal * radius_capsule

            return CollisionInfo(
                link_a=link_capsule,
                link_b=link_sphere,
                position=collision_point,
                normal=normal,
                depth=penetration_depth,
                collision_type=CollisionType.SELF_COLLISION
            )

        return None

    def _closest_point_on_segment(self, start: np.ndarray, end: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Find the closest point on a line segment to a given point."""
        segment_vec = end - start
        point_vec = point - start

        segment_len_sq = np.dot(segment_vec, segment_vec)

        if segment_len_sq < 1e-10:
            return start  # Degenerate segment

        t = np.dot(point_vec, segment_vec) / segment_len_sq
        t = np.clip(t, 0, 1)  # Clamp to segment

        return start + t * segment_vec

    def check_ground_collision(self, link_poses: dict, link_geometries: dict) -> List[CollisionInfo]:
        """Check for collisions with the ground plane."""
        ground_collisions = []
        ground_level = 0.0  # Ground at z = 0

        for link_name, pose in link_poses.items():
            geometry = link_geometries[link_name]

            # Calculate minimum Z extent of the link
            pos_z = pose['position'][2]

            if geometry['type'] == 'sphere':
                min_extent = pos_z - geometry['radius']
            elif geometry['type'] == 'capsule':
                # Account for orientation
                rot_mat = self._quaternion_to_matrix(pose['orientation'])
                axis = rot_mat @ np.array([0, 0, 1])
                half_len = geometry['length'] / 2.0

                # Project capsule extent along world Z
                extent_projection = abs(axis[2]) * half_len
                min_extent = pos_z - extent_projection - geometry['radius']
            elif geometry['type'] == 'box':
                # Calculate minimum Z extent considering orientation
                rot_mat = self._quaternion_to_matrix(pose['orientation'])
                dimensions = geometry['dimensions']

                # Find minimum Z extent of oriented box
                corners = self._get_box_corners(dimensions)
                transformed_corners = [pose['position'] + rot_mat @ corner for corner in corners]
                min_extent = min(corner[2] for corner in transformed_corners)
            else:
                min_extent = pos_z  # Default case

            # Check if link is below or penetrating ground
            if min_extent < ground_level + self.collision_margin:
                penetration_depth = (ground_level + self.collision_margin) - min_extent

                # Collision point is at ground level directly below the link
                collision_point = np.array([pose['position'][0], pose['position'][1], ground_level])
                normal = np.array([0, 0, 1])  # Normal pointing up

                ground_collisions.append(CollisionInfo(
                    link_a=link_name,
                    link_b='ground',
                    position=collision_point,
                    normal=normal,
                    depth=penetration_depth,
                    collision_type=CollisionType.GROUND_COLLISION
                ))

        return ground_collisions

    def _get_box_corners(self, dimensions: np.ndarray) -> List[np.ndarray]:
        """Get all 8 corners of an axis-aligned box."""
        dx, dy, dz = dimensions

        return [
            np.array([-dx, -dy, -dz]) / 2,
            np.array([dx, -dy, -dz]) / 2,
            np.array([-dx, dy, -dz]) / 2,
            np.array([dx, dy, -dz]) / 2,
            np.array([-dx, -dy, dz]) / 2,
            np.array([dx, -dy, dz]) / 2,
            np.array([-dx, dy, dz]) / 2,
            np.array([dx, dy, dz]) / 2
        ]

# Example usage
def example_usage():
    """Example of using the collision detection system."""
    # Define humanoid robot links
    robot_links = [
        'base_link', 'head_link',
        'left_shoulder_link', 'left_upper_arm_link', 'left_lower_arm_link', 'left_hand_link',
        'right_shoulder_link', 'right_upper_arm_link', 'right_lower_arm_link', 'right_hand_link',
        'left_hip_link', 'left_thigh_link', 'left_shin_link', 'left_foot_link',
        'right_hip_link', 'right_thigh_link', 'right_shin_link', 'right_foot_link'
    ]

    # Initialize collision detector
    collision_detector = CollisionDetector(robot_links)

    # Example robot pose (simplified)
    link_poses = {
        'base_link': {'position': np.array([0.0, 0.0, 1.0]), 'orientation': np.array([1.0, 0.0, 0.0, 0.0])},
        'head_link': {'position': np.array([0.0, 0.0, 1.5]), 'orientation': np.array([1.0, 0.0, 0.0, 0.0])},
        'left_foot_link': {'position': np.array([-0.1, 0.1, 0.05]), 'orientation': np.array([1.0, 0.0, 0.0, 0.0])},
        'right_foot_link': {'position': np.array([0.1, -0.1, 0.05]), 'orientation': np.array([1.0, 0.0, 0.0, 0.0])}
    }

    # Example link geometries
    link_geometries = {
        'base_link': {'type': 'capsule', 'radius': 0.15, 'length': 0.8},
        'head_link': {'type': 'sphere', 'radius': 0.12},
        'left_foot_link': {'type': 'box', 'dimensions': np.array([0.18, 0.1, 0.04])},
        'right_foot_link': {'type': 'box', 'dimensions': np.array([0.18, 0.1, 0.04])}
    }

    # Update collision detector with current pose
    self_collisions = collision_detector.update_robot_pose(link_poses, link_geometries)
    ground_collisions = collision_detector.check_ground_collision(link_poses, link_geometries)

    print(f"Detected {len(self_collisions)} self-collisions")
    print(f"Detected {len(ground_collisions)} ground collisions")

    for collision in ground_collisions:
        print(f"Ground collision: {collision.link_a} at {collision.position}")

if __name__ == "__main__":
    example_usage()
```

## Contact Simulation and Friction Modeling

Contact simulation is critical for realistic humanoid robot behavior, especially for locomotion and manipulation tasks. The contact model determines how forces are transmitted between the robot and its environment.

### Contact Models for Humanoid Robots

```python
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class ContactPoint:
    """Represents a single contact point between two bodies."""
    position: np.ndarray      # Contact position in world coordinates
    normal: np.ndarray        # Contact normal (points from first body to second)
    penetration_depth: float  # Depth of penetration
    body_indices: Tuple[int, int]  # Indices of contacting bodies
    force: np.ndarray = None  # Accumulated contact force

class ContactSolver:
    """Advanced contact solver for humanoid robot simulation."""

    def __init__(self, num_bodies: int, dt: float = 0.001):
        self.num_bodies = num_bodies
        self.dt = dt  # Time step

        # Physical parameters
        self.restitution = 0.2  # Coefficient of restitution
        self.static_friction = 0.5  # Static friction coefficient
        self.dynamic_friction = 0.3  # Dynamic friction coefficient
        self.safety_factor = 0.9    # For constraint stability

        # Cache for performance
        self.mass_inv_cache = np.zeros(num_bodies)
        self.inertia_inv_cache = np.zeros((num_bodies, 3, 3))

        print(f"Initialized contact solver for {num_bodies} bodies with dt={dt}")

    def solve_contacts(self, bodies_state: List[dict], contact_points: List[ContactPoint]) -> List[np.ndarray]:
        """Solve contact constraints and return corrective forces."""
        if not contact_points:
            return [np.zeros(6) for _ in range(self.num_bodies)]  # No contacts, no forces

        # Prepare system matrices
        num_contacts = len(contact_points)
        forces = np.zeros((num_contacts, 3))  # Normal and tangential forces

        # Solve each contact individually (simplified approach)
        # In practice, you'd solve the full LCP (Linear Complementarity Problem)
        for i, contact in enumerate(contact_points):
            body1_idx, body2_idx = contact.body_indices

            # Get body properties
            body1 = bodies_state[body1_idx]
            body2 = bodies_state[body2_idx]

            # Calculate relative velocity at contact point
            rel_vel = self._calculate_relative_velocity(
                body1, body2, contact.position, contact.normal
            )

            # Solve normal contact constraint
            normal_impulse = self._solve_normal_contact(
                body1, body2, contact, rel_vel
            )

            # Solve friction constraints
            tangent1, tangent2 = self._compute_tangent_vectors(contact.normal)
            friction_impulses = self._solve_friction_contacts(
                body1, body2, contact, rel_vel, tangent1, tangent2
            )

            # Combine impulses into force
            total_impulse = normal_impulse * contact.normal + friction_impulses[0] * tangent1 + friction_impulses[1] * tangent2
            forces[i] = total_impulse / self.dt  # Convert impulse to force

        # Distribute forces to bodies
        body_forces = [np.zeros(6) for _ in range(self.num_bodies)]  # [Fx, Fy, Fz, Tx, Ty, Tz]

        for i, contact in enumerate(contact_points):
            body1_idx, body2_idx = contact.body_indices
            force = forces[i]

            # Apply equal and opposite forces
            body_forces[body1_idx][:3] -= force[:3]  # Linear forces
            body_forces[body2_idx][:3] += force[:3]

            # Calculate torques (r × F)
            r1 = contact.position - bodies_state[body1_idx]['position']
            r2 = contact.position - bodies_state[body2_idx]['position']

            torque1 = np.cross(r1, force[:3])
            torque2 = np.cross(r2, force[:3])

            body_forces[body1_idx][3:] -= torque1  # Torque components
            body_forces[body2_idx][3:] -= torque2

        return body_forces

    def _calculate_relative_velocity(self, body1: dict, body2: dict, contact_pos: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Calculate relative velocity of contact points."""
        # Velocity of contact point on body 1
        r1 = contact_pos - body1['position']
        vel1 = body1['linear_velocity'] + np.cross(body1['angular_velocity'], r1)

        # Velocity of contact point on body 2
        r2 = contact_pos - body2['position']
        vel2 = body2['linear_velocity'] + np.cross(body2['angular_velocity'], r2)

        # Relative velocity
        rel_vel = vel1 - vel2

        return rel_vel

    def _solve_normal_contact(self, body1: dict, body2: dict, contact: ContactPoint, rel_vel: np.ndarray) -> float:
        """Solve normal contact constraint."""
        # Normal velocity component
        vn = np.dot(rel_vel, contact.normal)

        # Calculate effective mass
        r1 = contact.position - body1['position']
        r2 = contact.position - body2['position']

        # Effective inverse mass for normal direction
        eff_mass_inv = (
            body1['inv_mass'] + body2['inv_mass'] +
            np.dot(contact.normal, np.cross(body1['inv_inertia'] @ np.cross(r1, contact.normal), r1)) +
            np.dot(contact.normal, np.cross(body2['inv_inertia'] @ np.cross(r2, contact.normal), r2))
        )

        if eff_mass_inv <= 0:
            return 0.0

        eff_mass = 1.0 / eff_mass_inv

        # Calculate impulse to stop penetration velocity
        impulse_mag = -(1.0 + self.restitution) * vn
        impulse_mag /= eff_mass_inv

        # Apply penetration correction
        penetration_correction = max(0.0, contact.penetration_depth) * self.safety_factor / (self.dt * self.dt)
        impulse_mag += penetration_correction * eff_mass

        # Clamp impulse to prevent pulling apart
        impulse_mag = max(0.0, impulse_mag)

        return impulse_mag

    def _compute_tangent_vectors(self, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute two tangent vectors perpendicular to the normal."""
        # Find a vector not parallel to normal
        if abs(normal[2]) < 0.9:
            tangent1 = np.cross(normal, np.array([0, 0, 1]))
        else:
            tangent1 = np.cross(normal, np.array([1, 0, 0]))

        tangent1 = tangent1 / np.linalg.norm(tangent1)
        tangent2 = np.cross(normal, tangent1)
        tangent2 = tangent2 / np.linalg.norm(tangent2)

        return tangent1, tangent2

    def _solve_friction_contacts(self, body1: dict, body2: dict, contact: ContactPoint,
                                rel_vel: np.ndarray, tangent1: np.ndarray, tangent2: np.ndarray) -> Tuple[float, float]:
        """Solve friction constraints using the Coulomb friction model."""
        # Normal force magnitude (from normal contact)
        normal_vel = np.dot(rel_vel, contact.normal)
        normal_impulse = max(0.0, -(1.0 + self.restitution) * normal_vel)  # Approximate normal impulse

        # Tangential velocity components
        vt1 = np.dot(rel_vel, tangent1)
        vt2 = np.dot(rel_vel, tangent2)

        # Effective masses for tangential directions
        r1 = contact.position - body1['position']
        r2 = contact.position - body2['position']

        eff_mass_inv_t1 = (
            body1['inv_mass'] + body2['inv_mass'] +
            np.dot(tangent1, np.cross(body1['inv_inertia'] @ np.cross(r1, tangent1), r1)) +
            np.dot(tangent1, np.cross(body2['inv_inertia'] @ np.cross(r2, tangent1), r2))
        )

        eff_mass_inv_t2 = (
            body1['inv_mass'] + body2['inv_mass'] +
            np.dot(tangent2, np.cross(body1['inv_inertia'] @ np.cross(r1, tangent2), r1)) +
            np.dot(tangent2, np.cross(body2['inv_inertia'] @ np.cross(r2, tangent2), r2))
        )

        if eff_mass_inv_t1 <= 0 or eff_mass_inv_t2 <= 0:
            return 0.0, 0.0

        # Calculate tangential impulses
        impulse_t1 = -vt1 / eff_mass_inv_t1
        impulse_t2 = -vt2 / eff_mass_inv_t2

        # Apply friction cone constraint (Coulomb friction)
        max_friction_impulse = self.static_friction * normal_impulse

        # Magnitude of tangential impulse
        impulse_mag = np.sqrt(impulse_t1**2 + impulse_t2**2)

        if impulse_mag > max_friction_impulse:
            # Sliding friction
            scale = max_friction_impulse / impulse_mag
            impulse_t1 *= scale
            impulse_t2 *= scale

        return impulse_t1, impulse_t2

# Integration with physics simulation
class HumanoidPhysicsEngine:
    """Complete physics engine for humanoid robot simulation."""

    def __init__(self, num_links: int):
        self.num_links = num_links
        self.dt = 0.001  # 1ms time step
        self.gravity = np.array([0, 0, -9.81])

        # Initialize contact solver
        self.contact_solver = ContactSolver(num_links, self.dt)

        # Initialize collision detector
        self.collision_detector = None

        print(f"Initialized humanoid physics engine for {num_links} links")

    def integrate_motion(self, bodies_state: List[dict], external_forces: List[np.ndarray], dt: float = None):
        """Integrate equations of motion for all bodies."""
        if dt is None:
            dt = self.dt

        for i, body in enumerate(bodies_state):
            # Apply external forces
            linear_force = external_forces[i][:3]
            torque = external_forces[i][3:]

            # Update linear motion (F = ma => a = F/m)
            linear_acc = linear_force * body['inv_mass'] + self.gravity
            body['linear_velocity'] += linear_acc * dt
            body['position'] += body['linear_velocity'] * dt

            # Update angular motion (τ = Iα => α = I⁻¹τ)
            angular_acc = body['inv_inertia'] @ torque
            body['angular_velocity'] += angular_acc * dt

            # Update orientation using quaternion integration
            omega_quat = np.array([0, *body['angular_velocity']])
            quat_deriv = 0.5 * self._quat_multiply(omega_quat, body['orientation'])
            body['orientation'] += quat_deriv * dt
            # Normalize quaternion
            body['orientation'] /= np.linalg.norm(body['orientation'])

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

    def simulate_step(self, bodies_state: List[dict], link_geometries: dict):
        """Perform one simulation step including collision detection and contact resolution."""
        # Detect collisions
        collisions = self.collision_detector.update_robot_pose(
            {i: {'position': bodies_state[i]['position'], 'orientation': bodies_state[i]['orientation'], 'geometry': link_geometries[f'link_{i}']}
             for i in range(len(bodies_state))},
            link_geometries
        )

        # Add ground collisions
        ground_collisions = self.collision_detector.check_ground_collision(
            {i: {'position': bodies_state[i]['position'], 'orientation': bodies_state[i]['orientation'], 'geometry': link_geometries[f'link_{i}']}
             for i in range(len(bodies_state))},
            link_geometries
        )

        all_collisions = collisions + ground_collisions

        # Convert collisions to contact points
        contact_points = []
        for collision in all_collisions:
            # This is a simplified mapping - in reality you'd have more detailed collision info
            contact_points.append(ContactPoint(
                position=collision.position,
                normal=collision.normal,
                penetration_depth=collision.depth,
                body_indices=(self._get_link_index(collision.link_a), self._get_link_index(collision.link_b))
            ))

        # Solve contacts to get forces
        contact_forces = self.contact_solver.solve_contacts(bodies_state, contact_points)

        # Integrate motion with contact forces
        self.integrate_motion(bodies_state, contact_forces)

        return bodies_state

    def _get_link_index(self, link_name: str) -> int:
        """Convert link name to index."""
        # Simplified mapping - in reality you'd have a proper mapping
        if link_name.startswith('link_'):
            return int(link_name.split('_')[1])
        return 0  # Default

# Example usage
def run_physics_simulation():
    """Run a complete physics simulation example."""
    # Initialize physics engine
    engine = HumanoidPhysicsEngine(num_links=18)  # Example humanoid with 18 links

    # Initialize body states (simplified)
    bodies_state = []
    for i in range(18):
        body_state = {
            'position': np.random.rand(3) * 0.1,  # Small random positions
            'orientation': np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
            'linear_velocity': np.zeros(3),
            'angular_velocity': np.zeros(3),
            'inv_mass': 1.0,  # 1kg mass
            'inv_inertia': np.eye(3)  # Identity inertia matrix
        }
        bodies_state.append(body_state)

    # Initialize collision detector
    link_names = [f'link_{i}' for i in range(18)]
    engine.collision_detector = CollisionDetector(link_names)

    # Initialize link geometries
    link_geometries = {}
    for i in range(18):
        if i < 3:  # Torso/head links
            link_geometries[f'link_{i}'] = {'type': 'capsule', 'radius': 0.1, 'length': 0.3}
        elif i < 9:  # Arm links
            link_geometries[f'link_{i}'] = {'type': 'capsule', 'radius': 0.05, 'length': 0.2}
        else:  # Leg links
            link_geometries[f'link_{i}'] = {'type': 'capsule', 'radius': 0.07, 'length': 0.3}

    # Run simulation
    print("Starting physics simulation...")
    for step in range(1000):
        engine.simulate_step(bodies_state, link_geometries)

        if step % 100 == 0:
            print(f"Simulation step: {step}")

    print("Physics simulation completed!")

if __name__ == "__main__":
    run_physics_simulation()
```

## Performance Optimization for Real-time Simulation

```mermaid
graph TD
    A[Physics Simulation Pipeline] --> B[Broad Phase Collision Detection]
    A --> C[Narrow Phase Collision Detection]
    A --> D[Constraint Solving]
    A --> E[Integration]

    B --> F[Spatial Hashing]
    B --> G[AABB Trees]
    B --> H[Grid-based Partitioning]

    C --> I[GJK Algorithm]
    C --> J[SAT Algorithm]
    C --> K[Ray Casting]

    D --> L[Projected Gauss-Seidel]
    D --> M[Sequential Impulses]
    D --> N[MLCP Solvers]

    E --> O[Symplectic Integrators]
    E --> P[Runge-Kutta Methods]
    E --> Q[Implicit Methods]

    F --> R[O(1) Query Time]
    G --> S[Log n Complexity]
    H --> T[Simple Implementation]

    style A fill:#e1f5fe
    style D fill:#f3e5f5
    style E fill:#e8f5e8
</mermaid>

### Optimization Techniques for Humanoid Physics Simulation

```python
import numpy as np
import numba
from typing import List, Dict
import time

class OptimizedPhysicsEngine:
    """High-performance physics engine optimized for humanoid robot simulation."""

    def __init__(self, max_bodies: int = 100, dt: float = 0.001):
        self.max_bodies = max_bodies
        self.dt = dt
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float32)

        # Pre-allocate arrays for performance
        self.positions = np.zeros((max_bodies, 3), dtype=np.float32)
        self.velocities = np.zeros((max_bodies, 3), dtype=np.float32)
        self.orientations = np.zeros((max_bodies, 4), dtype=np.float32)  # Quaternions
        self.angular_velocities = np.zeros((max_bodies, 3), dtype=np.float32)
        self.inv_masses = np.ones(max_bodies, dtype=np.float32)
        self.inv_inertias = np.zeros((max_bodies, 3, 3), dtype=np.float32)

        # Initialize identity inertias
        for i in range(max_bodies):
            self.inv_inertias[i] = np.eye(3, dtype=np.float32)

        # Contact constraint data
        self.max_contacts = 1000
        self.contact_positions = np.zeros((self.max_contacts, 3), dtype=np.float32)
        self.contact_normals = np.zeros((self.max_contacts, 3), dtype=np.float32)
        self.contact_depths = np.zeros(self.max_contacts, dtype=np.float32)
        self.contact_body_indices = np.zeros((self.max_contacts, 2), dtype=np.int32)

        # Performance counters
        self.broad_phase_time = 0.0
        self.narrow_phase_time = 0.0
        self.constraint_solve_time = 0.0
        self.integration_time = 0.0

        print(f"Initialized optimized physics engine with {max_bodies} max bodies, dt={dt}")

    def integrate_motion_batch(self, num_bodies: int):
        """High-performance batch integration of motion equations."""
        start_time = time.time()

        # Vectorized integration of linear motion
        linear_acc = np.zeros_like(self.velocities[:num_bodies])

        # Add gravity to acceleration
        linear_acc[:, :] = self.gravity * self.inv_masses[:num_bodies, np.newaxis]

        # Update velocities and positions
        self.velocities[:num_bodies] += linear_acc * self.dt
        self.positions[:num_bodies] += self.velocities[:num_bodies] * self.dt

        # Vectorized integration of angular motion
        # For simplicity, using a basic angular velocity update
        self.angular_velocities[:num_bodies] *= 0.99  # Damping for stability

        # Update orientations using quaternion integration
        self._integrate_orientations(num_bodies)

        self.integration_time += time.time() - start_time

    @staticmethod
    @numba.jit(nopython=True)
    def _integrate_orientations_numba(orientations, angular_velocities, dt, num_bodies):
        """Numba-optimized quaternion integration."""
        for i in range(num_bodies):
            # Create quaternion from angular velocity
            omega_quat = np.array([0.0, angular_velocities[i, 0],
                                  angular_velocities[i, 1], angular_velocities[i, 2]], dtype=np.float32)

            # Multiply: dq/dt = 0.5 * ω * q
            temp = np.zeros(4, dtype=np.float32)
            temp[0] = -omega_quat[1]*orientations[i, 1] - omega_quat[2]*orientations[i, 2] - omega_quat[3]*orientations[i, 3]
            temp[1] = omega_quat[0]*orientations[i, 1] + omega_quat[3]*orientations[i, 2] - omega_quat[2]*orientations[i, 3]
            temp[2] = omega_quat[0]*orientations[i, 2] - omega_quat[3]*orientations[i, 1] + omega_quat[1]*orientations[i, 3]
            temp[3] = omega_quat[0]*orientations[i, 3] + omega_quat[2]*orientations[i, 1] - omega_quat[1]*orientations[i, 2]

            temp *= 0.5 * dt

            # Update orientation
            orientations[i] += temp

            # Normalize quaternion
            norm = np.sqrt(orientations[i, 0]**2 + orientations[i, 1]**2 +
                          orientations[i, 2]**2 + orientations[i, 3]**2)
            if norm > 0:
                orientations[i] /= norm

    def _integrate_orientations(self, num_bodies: int):
        """Integrate orientations using optimized method."""
        self._integrate_orientations_numba(
            self.orientations, self.angular_velocities, self.dt, num_bodies
        )

    def solve_constraints_batch(self, num_contacts: int):
        """Batch solve contact constraints with optimized algorithms."""
        start_time = time.time()

        if num_contacts == 0:
            return

        # Vectorized contact solving
        for i in range(num_contacts):
            body1_idx = self.contact_body_indices[i, 0]
            body2_idx = self.contact_body_indices[i, 1]

            # Get contact data
            contact_pos = self.contact_positions[i]
            contact_normal = self.contact_normals[i]
            penetration_depth = self.contact_depths[i]

            # Calculate relative velocity at contact
            r1 = contact_pos - self.positions[body1_idx]
            r2 = contact_pos - self.positions[body2_idx]

            v1 = self.velocities[body1_idx] + np.cross(self.angular_velocities[body1_idx], r1)
            v2 = self.velocities[body2_idx] + np.cross(self.angular_velocities[body2_idx], r2)
            rel_vel = v1 - v2

            # Normal velocity
            vn = np.dot(rel_vel, contact_normal)

            # Calculate effective mass
            k_normal = (self.inv_masses[body1_idx] + self.inv_masses[body2_idx] +
                       np.dot(contact_normal, np.cross(
                           self.inv_inertias[body1_idx] @ np.cross(r1, contact_normal), r1)) +
                       np.dot(contact_normal, np.cross(
                           self.inv_inertias[body2_idx] @ np.cross(r2, contact_normal), r2)))

            if k_normal > 0:
                # Normal impulse
                inv_k = 1.0 / k_normal
                rest_impulse = max(0.0, -(1.0 + 0.2) * vn)  # 0.2 = restitution
                pen_impulse = max(0.0, penetration_depth * 0.9 / (self.dt * self.dt))  # Penetration correction

                normal_impulse = (rest_impulse + pen_impulse) * inv_k

                # Apply impulse
                impulse_vec = normal_impulse * contact_normal
                self.velocities[body1_idx] -= self.inv_masses[body1_idx] * impulse_vec
                self.velocities[body2_idx] += self.inv_masses[body2_idx] * impulse_vec

                # Angular components
                self.angular_velocities[body1_idx] -= self.inv_inertias[body1_idx] @ np.cross(r1, impulse_vec)
                self.angular_velocities[body2_idx] -= self.inv_inertias[body2_idx] @ np.cross(r2, impulse_vec)

        self.constraint_solve_time += time.time() - start_time

    def reset_performance_counters(self):
        """Reset all performance counters."""
        self.broad_phase_time = 0.0
        self.narrow_phase_time = 0.0
        self.constraint_solve_time = 0.0
        self.integration_time = 0.0

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        total_time = (self.broad_phase_time + self.narrow_phase_time +
                     self.constraint_solve_time + self.integration_time)

        return {
            'total_time': total_time,
            'integration_time': self.integration_time,
            'constraint_solve_time': self.constraint_solve_time,
            'broad_phase_time': self.broad_phase_time,
            'narrow_phase_time': self.narrow_phase_time,
            'integration_percentage': (self.integration_time / total_time * 100) if total_time > 0 else 0,
            'constraint_percentage': (self.constraint_solve_time / total_time * 100) if total_time > 0 else 0
        }

# Example performance test
def performance_test():
    """Test the performance of the optimized physics engine."""
    print("Starting performance test...")

    # Initialize engine
    engine = OptimizedPhysicsEngine(max_bodies=50, dt=0.001)

    # Initialize some bodies
    num_bodies = 20
    for i in range(num_bodies):
        engine.positions[i] = np.random.rand(3).astype(np.float32) * 2.0
        engine.velocities[i] = (np.random.rand(3).astype(np.float32) - 0.5) * 0.1
        engine.orientations[i, 0] = 1.0  # Identity quaternion
        engine.inv_masses[i] = 1.0 + np.random.rand().astype(np.float32) * 0.5  # 1.0-1.5 kg

    # Simulate for a number of steps
    num_steps = 1000
    start_time = time.time()

    for step in range(num_steps):
        # Add some fake contacts for testing
        num_contacts = min(10, step % 20)  # Varying number of contacts
        for i in range(num_contacts):
            engine.contact_positions[i] = engine.positions[i % num_bodies].copy()
            engine.contact_normals[i] = np.array([0, 0, 1], dtype=np.float32)
            engine.contact_depths[i] = 0.001
            engine.contact_body_indices[i] = [i % num_bodies, (i+1) % num_bodies]

        # Run simulation step
        engine.solve_constraints_batch(num_contacts)
        engine.integrate_motion_batch(num_bodies)

    total_sim_time = time.time() - start_time

    # Print performance stats
    stats = engine.get_performance_stats()
    print(f"\nPerformance Results:")
    print(f"Total simulation time: {total_sim_time:.4f}s for {num_steps} steps")
    print(f"Average time per step: {total_sim_time/num_steps*1000:.3f}ms")
    print(f"Steps per second: {num_steps/total_sim_time:.1f}")
    print(f"Integration: {stats['integration_time']:.4f}s ({stats['integration_percentage']:.1f}%)")
    print(f"Constraint solving: {stats['constraint_solve_time']:.4f}s ({stats['constraint_percentage']:.1f}%)")

    return stats

if __name__ == "__main__":
    performance_test()
```

## Integration with NVIDIA Isaac Sim

When integrating with NVIDIA Isaac Sim, specific parameters and configurations are required to ensure optimal physics simulation performance:

```json
{
  "physics_sim_config": {
    "engine": "PhysX",
    "gravity": [0.0, 0.0, -9.81],
    "timestep": 0.001,
    "substeps": 1,
    "solver_type": "TGS",
    "solver_iterations": 4,
    "collision_margin": 0.001,
    "contact_offset": 0.02,
    "gpu": {
      "enabled": true,
      "use_gpu_sim": true,
      "gpu_feedback_mode": "all",
      "gpu_collision_frame_count": 2
    },
    "scene_query_resolution": {
      "default_buffer_size": 1024,
      "max_hits_any_query_size": 128,
      "max_hits_closest_query_size": 128
    },
    "articulation_solver": {
      "position_iteration_count": 4,
      "velocity_iteration_count": 1,
      "projection_angle_tolerance": 0.05236,
      "projection_linear_tolerance": 0.001,
      "max_depenetration_velocity": 100.0
    },
    "humanoid_specific": {
      "enable_character_controller": false,
      "max_angular_speed": 50.0,
      "sleep_threshold": 0.005,
      "stabilization_threshold": 0.0,
      "solver_batch_size": 32,
      "enable_ccd": true,
      "ccd_threshold": 0.05,
      "ccd_max_passes": 2
    }
  }
}
```

## Summary

Physics simulation and collision detection form the foundation of realistic humanoid robot digital twins. The implementation requires careful consideration of multiple factors:

1. **Physics Engine Selection**: Choose appropriate algorithms based on real-time requirements vs. accuracy needs
2. **Collision Detection**: Implement efficient broad and narrow phase algorithms for complex multi-link systems
3. **Contact Resolution**: Use stable constraint solvers that handle multiple simultaneous contacts
4. **Performance Optimization**: Apply batching, vectorization, and GPU acceleration where possible
5. **Integration**: Configure parameters appropriately for the target simulation environment

The examples provided demonstrate advanced techniques for implementing physics simulation that can handle the complex requirements of humanoid robot systems, including real-time performance, stability, and accuracy. Proper implementation of these systems enables effective testing and development of humanoid robot control algorithms in simulation before deployment on physical hardware.

## Advanced Topics in Humanoid Physics Simulation

### Multi-Body Dynamics for Humanoid Robots

For humanoid robots with complex kinematic structures, the equations of motion become significantly more complex. The articulated body algorithm (ABA) is commonly used to efficiently compute the forward dynamics of multi-body systems:

```cpp
// Advanced articulated body algorithm implementation
#include <Eigen/Dense>
#include <vector>

struct ArticulatedBody {
    Eigen::Matrix3d inertia;      // 3x3 inertia matrix
    double mass;                  // Mass of the body
    Eigen::Vector3d com;          // Center of mass
    Eigen::Matrix6d I_articulated; // Articulated body inertia
    Eigen::Vector6d pA;           // Articulated bias force
    Eigen::Matrix6d IA;           // Articulated inertia
    Eigen::VectorXd S;            // Motion subspace vector
    Eigen::MatrixXd U;            // U matrix for ABA
    double d;                     // d scalar for ABA
    Eigen::VectorXd u;            // u vector for ABA
};

class HumanoidDynamicsSolver {
private:
    std::vector<ArticulatedBody> bodies;
    std::vector<int> parent;      // Parent index for each body
    std::vector<Eigen::Matrix4d> transforms;  // Transforms from parent to child
    std::vector<Eigen::Vector6d> v;          // Velocities
    std::vector<Eigen::Vector6d> c;          // Coriolis accelerations
    std::vector<Eigen::Vector6d> a;          // Accelerations
    std::vector<double> q;                   // Joint positions
    std::vector<double> qd;                  // Joint velocities
    std::vector<double> qdd;                 // Joint accelerations
    std::vector<double> tau;                 // Joint torques

public:
    HumanoidDynamicsSolver(int num_bodies) {
        bodies.resize(num_bodies);
        parent.resize(num_bodies);
        transforms.resize(num_bodies);
        v.resize(num_bodies);
        c.resize(num_bodies);
        a.resize(num_bodies);
        q.resize(num_bodies-1);  // Root has 6 DOF, others have 1 each
        qd.resize(num_bodies-1);
        qdd.resize(num_bodies-1);
        tau.resize(num_bodies-1);
    }

    void computeForwardDynamics() {
        // 1. Initialize root
        a[0] = Eigen::Vector6d::Zero();
        IA[0] = bodies[0].I_articulated;
        pA[0] = -bodies[0].pA;

        // 2. Forward recursion to compute articulated inertias
        for (int i = 1; i < bodies.size(); i++) {
            int pa = parent[i];
            Eigen::Matrix6d X_pa_i = transformWrench(transforms[i]);

            // Articulated body inertia
            IA[i] = bodies[i].I_articulated;

            // Propagate articulated inertia
            IA[i] = IA[i] + X_pa_i.transpose() * IA[pa] * X_pa_i;

            // Bias force
            pA[i] = bodies[i].pA + X_pa_i.transpose() * (pA[pa] + IA[i] * c[i]);
        }

        // 3. Backward recursion to compute joint accelerations
        for (int i = bodies.size()-1; i >= 0; i--) {
            Eigen::Matrix6d X_pa_i = transformWrench(transforms[i]);
            Eigen::Vector6d U_i = IA[i] * S[i];
            double d_i = S[i].transpose() * U_i;
            Eigen::Vector6d u_i = tau[i] - S[i].transpose() * pA[i];

            if (i == 0) {
                // Root body - 6 DOF
                Eigen::Matrix6d root_inertia = IA[0];
                Eigen::Vector6d root_bias = pA[0];
                qdd.segment(0, 6) = root_inertia.inverse() * (Eigen::Vector6d::Zero() - root_bias);
            } else {
                qdd[i-1] = (u_i - U_i.transpose() * a[parent[i]]) / d_i;
            }
        }

        // 4. Forward recursion to compute body accelerations
        for (int i = 1; i < bodies.size(); i++) {
            int pa = parent[i];
            Eigen::Matrix6d X_pa_i = transformWrench(transforms[i]);
            a[i] = X_pa_i * a[pa] + c[i] + S[i] * qdd[i-1];
        }
    }

    Eigen::Matrix6d transformWrench(const Eigen::Matrix4d& X) {
        // Transform a 6D spatial force vector
        Eigen::Matrix6d X_transform = Eigen::Matrix6d::Zero();
        X_transform.block<3,3>(0,0) = X.block<3,3>(0,0);
        X_transform.block<3,3>(0,3) = skew(X.block<3,1>(0,3)) * X.block<3,3>(0,0);
        X_transform.block<3,3>(3,0) = Eigen::Matrix3d::Zero();
        X_transform.block<3,3>(3,3) = X.block<3,3>(0,0);

        return X_transform;
    }

    Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
        Eigen::Matrix3d S;
        S << 0, -v(2), v(1),
             v(2), 0, -v(0),
             -v(1), v(0), 0;
        return S;
    }

    void updateKinematics() {
        // Update transforms and velocities based on current joint positions and velocities
        // This is called before dynamics computation
    }
};
```

### Real-time Physics Simulation Considerations

For real-time humanoid simulation, several optimization strategies are essential:

1. **Fixed Time Step Integration**: Ensures deterministic behavior and stability
2. **Parallel Processing**: Utilize multi-threading for collision detection and constraint solving
3. **Approximation Techniques**: Use simplified models when computational budget is tight
4. **Caching**: Pre-compute and cache expensive calculations when possible

```python
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class RealtimePhysicsSimulator:
    """Real-time physics simulator optimized for humanoid robots."""

    def __init__(self, target_frequency=1000):  # 1kHz target
        self.target_frequency = target_frequency
        self.dt = 1.0 / target_frequency
        self.current_time = 0.0

        # Thread pools for different physics tasks
        self.collision_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="collision")
        self.constraint_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="constraint")
        self.integration_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="integration")

        # Fixed-size buffers for real-time performance
        self.max_bodies = 100
        self.max_contacts = 1000

        # Real-time safety margins
        self.computation_budget = 0.8 * self.dt  # Use only 80% of time step

        print(f"Initialized real-time physics simulator at {target_frequency}Hz")

    def simulate_frame(self, bodies_state, contact_constraints):
        """Simulate one physics frame with real-time guarantees."""
        start_time = time.perf_counter()

        # Perform collision detection in parallel
        collision_future = self.collision_pool.submit(
            self.parallel_collision_detection, bodies_state
        )

        # Process constraints while collision detection runs
        constraint_result = self.solve_constraints(contact_constraints)

        # Wait for collision detection to complete
        new_collisions = collision_future.result()

        # Integrate motion
        integrated_state = self.integrate_motion(bodies_state, self.dt)

        # Check real-time performance
        frame_time = time.perf_counter() - start_time
        if frame_time > self.computation_budget:
            print(f"Warning: Physics frame took {frame_time*1000:.2f}ms, budget was {self.computation_budget*1000:.2f}ms")

        return integrated_state, new_collisions

    def parallel_collision_detection(self, bodies_state):
        """Perform collision detection using parallel processing."""
        # Split bodies into chunks for parallel processing
        num_threads = 4
        chunk_size = len(bodies_state) // num_threads
        futures = []

        for i in range(num_threads):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_threads - 1 else len(bodies_state)
            chunk = bodies_state[start_idx:end_idx]

            future = self.collision_pool.submit(
                self.collision_detection_worker, chunk, bodies_state, start_idx
            )
            futures.append(future)

        # Collect results
        all_collisions = []
        for future in futures:
            collisions = future.result()
            all_collisions.extend(collisions)

        return all_collisions

    def collision_detection_worker(self, local_bodies, all_bodies, start_idx):
        """Worker function for parallel collision detection."""
        collisions = []

        for i, body_a in enumerate(local_bodies):
            for j, body_b in enumerate(all_bodies):
                if i + start_idx == j:  # Skip self-collision
                    continue

                # Perform collision check
                if self.broad_phase_check(body_a, body_b):
                    collision = self.narrow_phase_check(body_a, body_b)
                    if collision:
                        collisions.append(collision)

        return collisions

    def broad_phase_check(self, body_a, body_b):
        """Fast broad-phase collision check."""
        # Simple sphere-based broad phase
        pos_a = body_a['position']
        pos_b = body_b['position']
        dist_sq = np.sum((pos_a - pos_b)**2)

        # Estimate bounding sphere radius
        radius_a = body_a.get('bounding_radius', 0.1)
        radius_b = body_b.get('bounding_radius', 0.1)
        threshold_sq = (radius_a + radius_b + 0.01)**2

        return dist_sq < threshold_sq

    def narrow_phase_check(self, body_a, body_b):
        """Detailed narrow-phase collision check."""
        # Placeholder for actual narrow-phase algorithm
        # In practice, this would use GJK, SAT, or other algorithms
        return None

    def solve_constraints(self, constraints):
        """Solve physics constraints."""
        # Parallel constraint solving
        return self.constraint_pool.submit(
            self.constraint_solver_worker, constraints
        ).result()

    def constraint_solver_worker(self, constraints):
        """Worker for constraint solving."""
        # Solve constraints using iterative methods
        max_iterations = 10
        for iteration in range(max_iterations):
            for constraint in constraints:
                self.solve_single_constraint(constraint)

        return constraints

    def solve_single_constraint(self, constraint):
        """Solve a single physics constraint."""
        # Implementation of constraint solving algorithm
        pass

    def integrate_motion(self, bodies_state, dt):
        """Integrate equations of motion."""
        # Use fixed-coefficient integration for real-time stability
        for body in bodies_state:
            # Semi-implicit Euler integration
            body['velocity'] += body['acceleration'] * dt
            body['position'] += body['velocity'] * dt

            # Integrate orientation (simplified)
            omega = body.get('angular_velocity', np.zeros(3))
            orientation = body.get('orientation', np.array([1, 0, 0, 0]))

            # Quaternion integration
            q_dot = 0.5 * self.quat_multiply(
                np.concatenate([[0], omega]),
                orientation
            )
            orientation += q_dot * dt
            orientation = orientation / np.linalg.norm(orientation)
            body['orientation'] = orientation

        return bodies_state

    def quat_multiply(self, q1, q2):
        """Quaternion multiplication."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

    def shutdown(self):
        """Clean up resources."""
        self.collision_pool.shutdown(wait=True)
        self.constraint_pool.shutdown(wait=True)
        self.integration_pool.shutdown(wait=True)

# Real-time simulation loop
def run_realtime_simulation():
    """Run a real-time physics simulation loop."""
    simulator = RealtimePhysicsSimulator(target_frequency=1000)

    # Initialize humanoid state
    bodies_state = initialize_humanoid_state()

    # Timing control
    import time
    next_frame_time = time.time()

    for frame in range(10000):  # Run for 10 seconds at 1kHz
        current_time = time.time()

        if current_time < next_frame_time:
            # Sleep to maintain timing
            time.sleep(next_frame_time - current_time)

        # Perform physics simulation
        bodies_state, collisions = simulator.simulate_frame(bodies_state, [])

        # Update timing
        next_frame_time += simulator.dt

        if frame % 1000 == 0:
            print(f"Frame {frame}, real-time factor: {(time.time() - current_time) / simulator.dt:.2f}x")

def initialize_humanoid_state():
    """Initialize a humanoid robot state for simulation."""
    # Placeholder for humanoid initialization
    return []
```

### Isaac Sim Performance Tuning

For optimal performance in Isaac Sim, specific configurations are required:

```json
{
  "isaac_sim_physics_config": {
    "physics_properties": {
      "solver_type": "TGS",
      "solver_position_iteration_count": 8,
      "solver_velocity_iteration_count": 1,
      "sleep_threshold": 0.005,
      "stabilization_threshold": 0.0,
      "max_depenetration_velocity": 100.0,
      "use_enhanced_determinism": false,
      "gpu_max_particles": 1048576,
      "gpu_max_particle_contacts": 1048576
    },
    "scene_properties": {
      "gravity": [0.0, 0.0, -9.81],
      "enable_gpu_physics": true,
      "broadphase_type": "MBP",
      "collision_stack_size": 64000000,
      "use_gpu_dynamic_scene": true,
      "gpu_max_rigid_contact_count": 524288,
      "gpu_max_rigid_patch_count": 33554432,
      "gpu_found_lost_pairs_capacity": 1048576,
      "gpu_found_lost_aggregate_pairs_capacity": 1048576,
      "gpu_total_aggregate_pairs_capacity": 1048576
    },
    "humanoid_robot_properties": {
      "enable_character_controller": false,
      "max_angular_speed": 50.0,
      "solver_batch_size": 32,
      "enable_ccd": true,
      "ccd_threshold": 0.05,
      "ccd_max_passes": 2,
      "solver_frequency": 240,
      "use_fabric": true,
      "enable_kinematic_pairs": true,
      "enable_gpu_feedback": true,
      "gpu_feedback_mode": "all"
    },
    "optimization_settings": {
      "max_linear_velocity": 100.0,
      "max_angular_velocity": 50.0,
      "max_depenetration_velocity": 100.0,
      "bounce_threshold_velocity": 2.0,
      "friction_correlation_model": "patch",
      "gpu_sim_frame_timeout": 2000,
      "thread_count": 4,
      "enable_pcm": true,
      "pcm_max_pairs": 1048576
    }
  }
}
```

### Advanced Contact Modeling for Humanoid Locomotion

For humanoid robots, special attention must be paid to foot-ground contact modeling, which is critical for stable locomotion:

```python
import numpy as np
from scipy.spatial import ConvexHull
from typing import List, Tuple

class FootContactModel:
    """Advanced foot contact model for humanoid locomotion."""

    def __init__(self, foot_geometry: dict):
        self.foot_geometry = foot_geometry
        self.contact_points = self._generate_contact_points(foot_geometry)
        self.pressure_distribution = self._calculate_pressure_distribution()

    def _generate_contact_points(self, geometry: dict) -> np.ndarray:
        """Generate contact points based on foot geometry."""
        if geometry['type'] == 'rectangular':
            # Generate points along the perimeter and center
            length = geometry['length']
            width = geometry['width']

            # Perimeter points
            points = []
            # Front edge
            for i in range(5):
                x = -length/2 + i * length/4
                points.append([x, width/2, 0])

            # Back edge
            for i in range(5):
                x = -length/2 + i * length/4
                points.append([x, -width/2, 0])

            # Side edges (excluding corners already covered)
            for i in range(3):
                y = -width/2 + (i+1) * width/4
                points.append([length/2, y, 0])
                points.append([-length/2, y, 0])

            # Center point
            points.append([0, 0, 0])

            return np.array(points)

        elif geometry['type'] == 'elliptical':
            # Generate points in elliptical pattern
            a = geometry['semi_major_axis']
            b = geometry['semi_minor_axis']

            points = []
            # Create grid and keep only points inside ellipse
            for x in np.linspace(-a, a, 7):
                for y in np.linspace(-b, b, 5):
                    if (x**2)/(a**2) + (y**2)/(b**2) <= 1:
                        points.append([x, y, 0])

            return np.array(points)

    def _calculate_pressure_distribution(self) -> np.ndarray:
        """Calculate default pressure distribution across contact points."""
        # For a foot, pressure is typically higher at heel and forefoot
        n_points = len(self.contact_points)
        pressures = np.zeros(n_points)

        for i, point in enumerate(self.contact_points):
            x, y, z = point

            # Pressure based on position (heuristic model)
            if x < -0.1:  # Heel area
                pressures[i] = 0.8
            elif x > 0.1:  # Toe area
                pressures[i] = 0.6
            else:  # Arch area
                pressures[i] = 0.3

        # Normalize
        pressures = pressures / np.sum(pressures)
        return pressures

    def calculate_contact_forces(self, foot_pose: dict, ground_normal: np.ndarray,
                                total_force: float) -> List[Tuple[np.ndarray, float]]:
        """Calculate distributed contact forces based on foot pose and loading."""
        contact_forces = []

        # Transform contact points to world coordinates
        world_points = self._transform_to_world(self.contact_points, foot_pose)

        # Calculate pressure at each point based on orientation
        for i, (world_point, pressure) in enumerate(zip(world_points, self.pressure_distribution)):
            # Adjust pressure based on local ground contact
            local_normal = self._adjust_normal_for_local_geometry(world_point, ground_normal)

            # Calculate force magnitude based on pressure and total load
            force_magnitude = total_force * pressure

            # Apply force in direction of adjusted normal
            force_vector = force_magnitude * local_normal
            contact_forces.append((world_point, force_vector))

        return contact_forces

    def _transform_to_world(self, local_points: np.ndarray, pose: dict) -> np.ndarray:
        """Transform points from local foot frame to world frame."""
        position = np.array(pose['position'])
        orientation = np.array(pose['orientation'])  # Quaternion [w, x, y, z]

        # Convert quaternion to rotation matrix
        rotation_matrix = self._quaternion_to_rotation_matrix(orientation)

        # Transform points
        world_points = []
        for point in local_points:
            world_point = position + rotation_matrix @ point
            world_points.append(world_point)

        return np.array(world_points)

    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q

        # Normalize quaternion
        norm = np.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm

        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

    def _adjust_normal_for_local_geometry(self, point: np.ndarray, ground_normal: np.ndarray) -> np.ndarray:
        """Adjust contact normal based on local foot geometry."""
        # For now, return ground normal; in a full implementation,
        # this would consider the local surface normal of the foot
        return ground_normal

class AdvancedLocomotionPhysics:
    """Physics system for advanced humanoid locomotion."""

    def __init__(self, robot_mass: float = 80.0):
        self.robot_mass = robot_mass
        self.gravity = 9.81
        self.total_weight = robot_mass * self.gravity

        # Initialize foot contact models
        self.left_foot_model = FootContactModel({
            'type': 'rectangular',
            'length': 0.25,
            'width': 0.10
        })

        self.right_foot_model = FootContactModel({
            'type': 'rectangular',
            'length': 0.25,
            'width': 0.10
        })

    def calculate_ground_reactions(self, robot_state: dict) -> dict:
        """Calculate ground reaction forces for both feet."""
        left_foot_pose = robot_state.get('left_foot_pose', {'position': [0,0,0], 'orientation': [1,0,0,0]})
        right_foot_pose = robot_state.get('right_foot_pose', {'position': [0,0,0], 'orientation': [1,0,0,0]})

        # Determine weight distribution based on center of mass position
        com_position = np.array(robot_state.get('com_position', [0,0,0]))
        left_foot_pos = np.array(left_foot_pose['position'])
        right_foot_pos = np.array(right_foot_pose['position'])

        # Simple weight distribution based on distance to each foot
        dist_to_left = np.linalg.norm(com_position[:2] - left_foot_pos[:2])
        dist_to_right = np.linalg.norm(com_position[:2] - right_foot_pos[:2])

        # Weight distribution (inverse to distance, normalized)
        left_weight_ratio = dist_to_right / (dist_to_left + dist_to_right)
        right_weight_ratio = dist_to_left / (dist_to_left + dist_to_right)

        left_force = self.total_weight * left_weight_ratio
        right_force = self.total_weight * right_weight_ratio

        # Calculate contact forces for each foot
        left_contacts = self.left_foot_model.calculate_contact_forces(
            left_foot_pose, np.array([0, 0, 1]), left_force
        )

        right_contacts = self.right_foot_model.calculate_contact_forces(
            right_foot_pose, np.array([0, 0, 1]), right_force
        )

        return {
            'left_foot_contacts': left_contacts,
            'right_foot_contacts': right_contacts,
            'total_left_force': left_force,
            'total_right_force': right_force
        }

    def update_balance_control(self, robot_state: dict) -> dict:
        """Update balance control based on physics calculations."""
        ground_reactions = self.calculate_ground_reactions(robot_state)

        # Calculate center of pressure
        cop_left = self._calculate_center_of_pressure(ground_reactions['left_foot_contacts'])
        cop_right = self._calculate_center_of_pressure(ground_reactions['right_foot_contacts'])

        # Calculate zero moment point (ZMP)
        zmp = self._calculate_zmp(robot_state, ground_reactions)

        return {
            'center_of_pressure': {'left': cop_left, 'right': cop_right},
            'zero_moment_point': zmp,
            'ground_reactions': ground_reactions
        }

    def _calculate_center_of_pressure(self, contacts: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """Calculate center of pressure from contact forces."""
        if not contacts:
            return np.array([0, 0, 0])

        total_force = np.array([0.0, 0.0, 0.0])
        moment = np.array([0.0, 0.0, 0.0])

        for point, force in contacts:
            total_force += force
            moment += np.cross(point[:3], force)  # Only use first 3 components (position)

        if abs(total_force[2]) > 1e-6:  # Avoid division by zero
            cop_x = moment[1] / total_force[2]  # M_y / F_z
            cop_y = -moment[0] / total_force[2]  # -M_x / F_z
            return np.array([cop_x, cop_y, 0])
        else:
            return np.array([0, 0, 0])

    def _calculate_zmp(self, robot_state: dict, ground_reactions: dict) -> np.ndarray:
        """Calculate Zero Moment Point for balance control."""
        # Simplified ZMP calculation
        # ZMP = (Σ(Fi * xi) + Σ(Ti)) / Σ(Fi_z)
        # where Fi is force at contact i, xi is position, Ti is torque

        all_contacts = (
            ground_reactions['left_foot_contacts'] +
            ground_reactions['right_foot_contacts']
        )

        if not all_contacts:
            return np.array([0, 0, 0])

        total_moment_x = 0.0
        total_moment_y = 0.0
        total_force_z = 0.0

        com_height = robot_state.get('com_position', [0, 0, 0.8])[2]

        for point, force in all_contacts:
            # Moments about x and y axes
            total_moment_x += point[1] * force[2] - point[2] * force[1]  # y*Fz - z*Fy
            total_moment_y += point[2] * force[0] - point[0] * force[2]  # z*Fx - x*Fz
            total_force_z += force[2]  # Fz

        if abs(total_force_z) > 1e-6:
            zmp_x = -total_moment_y / total_force_z
            zmp_y = total_moment_x / total_force_z
            return np.array([zmp_x, zmp_y, 0])
        else:
            return np.array([0, 0, 0])

# Example usage
def example_locomotion_physics():
    """Example of advanced locomotion physics in action."""
    locomotion_physics = AdvancedLocomotionPhysics(robot_mass=80.0)

    # Sample robot state
    robot_state = {
        'com_position': [0.0, 0.0, 0.85],  # Center of mass at 85cm height
        'left_foot_pose': {
            'position': [-0.1, 0.1, 0.0],
            'orientation': [1.0, 0.0, 0.0, 0.0]
        },
        'right_foot_pose': {
            'position': [0.1, -0.1, 0.0],
            'orientation': [1.0, 0.0, 0.0, 0.0]
        }
    }

    balance_info = locomotion_physics.update_balance_control(robot_state)

    print("Balance Control Information:")
    print(f"Center of Pressure (Left): {balance_info['center_of_pressure']['left']}")
    print(f"Center of Pressure (Right): {balance_info['center_of_pressure']['right']}")
    print(f"Zero Moment Point: {balance_info['zero_moment_point']}")
    print(f"Total Left Force: {balance_info['ground_reactions']['total_left_force']:.2f}N")
    print(f"Total Right Force: {balance_info['ground_reactions']['total_right_force']:.2f}N")

if __name__ == "__main__":
    example_locomotion_physics()
```

## Conclusion

Physics simulation for humanoid robots in digital twin environments requires sophisticated implementations that balance computational efficiency with physical accuracy. The systems described in this chapter provide the foundation for realistic simulation of humanoid robot dynamics, including collision detection, contact resolution, and real-time performance optimization. Proper implementation of these physics systems enables safe and effective development of humanoid robot control algorithms in simulation before deployment on physical hardware.