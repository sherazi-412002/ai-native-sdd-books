import React from 'react';
import clsx from 'clsx';
import styles from './CurriculumNavigation.module.css';

type CurriculumModule = {
  title: string;
  description: string;
  chapters: string[];
  position: number;
};

const FeatureList: CurriculumModule[] = [
  {
    title: 'Module 1: The Robotic Nervous System (ROS 2)',
    description: 'Introduction to ROS 2 fundamentals for autonomous humanoid robotics',
    chapters: ['Nodes/Topics/Services', 'rclpy bridging', 'URDF humanoids'],
    position: 1,
  },
  {
    title: 'Module 2: The Digital Twin (Gazebo & Unity)',
    description: 'Creating digital twins with Gazebo and Unity for humanoid robotics',
    chapters: ['Physics/Collisions', 'Unity Rendering', 'Sensor Simulation'],
    position: 2,
  },
  {
    title: 'Module 3: The AI-Robot Brain (NVIDIA Isaac™)',
    description: 'Using NVIDIA Isaac for perception and planning in humanoid robots',
    chapters: ['Synthetic Data', 'Isaac ROS VSLAM', 'Nav2 Path Planning'],
    position: 3,
  },
  {
    title: 'Module 4: Vision-Language-Action (VLA)',
    description: 'Implementing Vision-Language-Action models for humanoid robots',
    chapters: ['Whisper Integration', 'LLM Cognitive Planning', 'Capstone Project'],
    position: 4,
  },
];

function Feature({title, description, chapters, position}: CurriculumModule) {
  return (
    <div className={clsx('col col--6')}>
      <div className="text--center padding-horiz--md">
        <h2>{title}</h2>
        <p>{description}</p>
        <ul>
          {chapters.map((chapter, index) => (
            <li key={index}>{chapter}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default function CurriculumNavigation(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}