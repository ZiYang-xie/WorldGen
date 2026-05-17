import { useState } from 'react';
import type { CapitalistView } from './types';
import { CapitalistHeader } from './components/CapitalistHeader';
import { CapitalistNav } from './components/CapitalistNav';
import { CapitalistTerminal } from './components/CapitalistTerminal';
import { CapitalistReserve } from './components/CapitalistReserve';
import { CapitalistDispatch } from './components/CapitalistDispatch';
import { CapitalistFleet } from './components/CapitalistFleet';
import { CapitalistCore } from './components/CapitalistCore';

const VIEWS: Record<CapitalistView, React.ReactNode> = {
  terminal:  <CapitalistTerminal />,
  reserve:   <CapitalistReserve />,
  dispatch:  <CapitalistDispatch />,
  fleet:     <CapitalistFleet />,
  core:      <CapitalistCore />,
};

export function CapitalistApp() {
  const [activeView, setActiveView] = useState<CapitalistView>('terminal');

  return (
    <div className="ac-app">
      <CapitalistHeader coreActive latency={12} />
      <div className="ac-body">
        <CapitalistNav active={activeView} onSelect={setActiveView} />
        {VIEWS[activeView]}
      </div>
    </div>
  );
}
