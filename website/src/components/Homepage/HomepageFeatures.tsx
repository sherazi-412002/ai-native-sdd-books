import React from 'react';
import clsx from 'clsx';
import styles from './HomepageFeatures.module.css';

type FeatureItem = {
  title: string;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'ROS 2 Fundamentals',
    description: (
      <>
        Learn the core concepts of ROS 2 that form the nervous system of autonomous humanoid robots.
      </>
    ),
  },
  {
    title: 'Digital Twins',
    description: (
      <>
        Create realistic simulations with Gazebo and Unity for testing humanoid robot behaviors.
      </>
    ),
  },
  {
    title: 'AI Integration',
    description: (
      <>
        Integrate NVIDIA Isaac, LLMs, and VLA models for advanced humanoid robot capabilities.
      </>
    ),
  },
];

function Feature({title, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
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