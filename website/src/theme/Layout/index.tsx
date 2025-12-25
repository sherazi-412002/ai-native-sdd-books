import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatWidget from '@site/src/components/ChatWidget';
import type {Props} from '@theme/Layout';

export default function Layout(props: Props): JSX.Element {
  return (
    <>
      <OriginalLayout {...props} />
      <ChatWidget />
    </>
  );
}