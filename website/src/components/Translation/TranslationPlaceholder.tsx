import React from 'react';
import clsx from 'clsx';
import styles from './TranslationPlaceholder.module.css';

type TranslationPlaceholderProps = {
  position?: 'top' | 'bottom' | 'sidebar';
};

export default function TranslationPlaceholder({
  position = 'top'
}: TranslationPlaceholderProps): JSX.Element {
  return (
    <div className={clsx(styles.translationPlaceholder, styles[position])}>
      <div className={styles.translationContainer}>
        <span className={styles.translationIcon}>🌐</span>
        <select className={styles.translationSelect} disabled>
          <option>English (en)</option>
          <option disabled>Urdu (ur) - Coming Soon</option>
        </select>
        <span className={styles.translationBadge}>Phase 3 Feature</span>
      </div>
    </div>
  );
}