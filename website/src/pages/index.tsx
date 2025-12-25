import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/Homepage/HomepageFeatures';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className="badge badge--primary margin-bottom--sm">The Future of Robotics</div>
        <Heading as="h1" className="hero__title">
          <span>Physical AI</span>
          <span>& Humanoid Robotics</span>
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            View Full Curriculum
          </Link>
        </div>
      </div>
    </header>
  );
}

function CurriculumOverview() {
  return (
    <section className={styles.features}>
      <div className="container padding-vert--xl">
        <div className="row">
          <div className="col col--3 padding-horiz--md">
            <div className={clsx('card', styles.card)}>
              <div className="card__header">
                <h3>🤖 Module 1: The Robotic Nervous System</h3>
              </div>
              <div className="card__body">
                <p>Master ROS 2 fundamentals for autonomous humanoid robotics</p>
              </div>
              <div className="card__footer">
                <Link className="button button--primary" to="/docs/module-1/intro">
                  Start Module
                </Link>
              </div>
            </div>
          </div>
          <div className="col col--3 padding-horiz--md">
            <div className={clsx('card', styles.card)}>
              <div className="card__header">
                <h3>🌐 Module 2: The Digital Twin</h3>
              </div>
              <div className="card__body">
                <p>Create realistic simulations with Gazebo and Unity</p>
              </div>
              <div className="card__footer">
                <Link className="button button--primary" to="/docs/module-2/intro">
                  Start Module
                </Link>
              </div>
            </div>
          </div>
          <div className="col col--3 padding-horiz--md">
            <div className={clsx('card', styles.card)}>
              <div className="card__header">
                <h3>🧠 Module 3: The AI-Robot Brain</h3>
              </div>
              <div className="card__body">
                <p>Implement perception and planning with NVIDIA Isaac</p>
              </div>
              <div className="card__footer">
                <Link className="button button--primary" to="/docs/module-3/intro">
                  Start Module
                </Link>
              </div>
            </div>
          </div>
          <div className="col col--3 padding-horiz--md">
            <div className={clsx('card', styles.card)}>
              <div className="card__header">
                <h3>⚡ Module 4: Vision-Language-Action</h3>
              </div>
              <div className="card__body">
                <p>Integrate multimodal AI with real-world robotics</p>
              </div>
              <div className="card__footer">
                <Link className="button button--primary" to="/docs/module-4/intro">
                  Start Module
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="A Comprehensive Guide to Autonomous Humanoid Robotics">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <CurriculumOverview />
      </main>
    </Layout>
  );
}
