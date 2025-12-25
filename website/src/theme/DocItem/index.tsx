import React from 'react';
import DocItem from '@theme-original/DocItem';
import TranslationPlaceholder from '@site/src/components/Translation/TranslationPlaceholder';

export default function DocItemWrapper(props) {
  return (
    <>
      <TranslationPlaceholder position="top" />
      <DocItem {...props} />
      <TranslationPlaceholder position="bottom" />
    </>
  );
}