import { invoke } from "@tauri-apps/api/core";

interface Props {
  coreActive?: boolean;
  latency?: number;
}

export function CapitalistHeader({ coreActive = true, latency = 12 }: Props) {
  return (
    <header className="ac-header">
      <div className="ac-logo">
        <div className="ac-logo-mark" />
        <span className="ac-logo-text">Auroch Capitalist</span>
      </div>

      <div className="ac-header-spacer" />

      <div className="ac-core-status">
        <span className={`ac-core-dot ${coreActive ? '' : 'amber'}`} />
        <span className="ac-core-label">
          {coreActive ? 'Argent Core Active' : 'Standby'}
        </span>
        <span className="ac-core-metric">│ {latency}ms</span>
      </div>

      <div className="ac-header-controls">
        <button className="titlebar-btn min"    onClick={() => invoke("window_minimize")} aria-label="Minimize" />
        <button className="titlebar-btn max"    onClick={() => invoke("window_maximize")} aria-label="Maximize" />
        <button className="titlebar-btn close"  onClick={() => invoke("window_close")}   aria-label="Close"    />
      </div>
    </header>
  );
}
