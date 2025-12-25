import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Sidebar for the Autonomous Humanoid curriculum
  curriculumSidebar: [
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      items: [
        {
          type: 'category',
          label: 'Chapters',
          items: [
            'module-1/nodes-topics-services',
            'module-1/rclpy-bridging',
            'module-1/urdf-humanoids'
          ]
        }
      ],
      link: {
        type: 'generated-index',
        description: 'Introduction to ROS 2 fundamentals for autonomous humanoid robotics'
      }
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        {
          type: 'category',
          label: 'Chapters',
          items: [
            'module-2/physics-collisions',
            'module-2/unity-rendering',
            'module-2/sensor-simulation'
          ]
        }
      ],
      link: {
        type: 'generated-index',
        description: 'Creating digital twins with Gazebo and Unity for humanoid robotics'
      }
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
      items: [
        {
          type: 'category',
          label: 'Chapters',
          items: [
            'module-3/synthetic-data',
            'module-3/isaac-ros-vslam',
            'module-3/nav2-path-planning'
          ]
        }
      ],
      link: {
        type: 'generated-index',
        description: 'Using NVIDIA Isaac for perception and planning in humanoid robots'
      }
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      items: [
        {
          type: 'category',
          label: 'Chapters',
          items: [
            'module-4/whisper-integration',
            'module-4/llm-cognitive-planning',
            'module-4/capstone-project'
          ]
        }
      ],
      link: {
        type: 'generated-index',
        description: 'Implementing Vision-Language-Action models for humanoid robots'
      }
    }
  ],
};

export default sidebars;
