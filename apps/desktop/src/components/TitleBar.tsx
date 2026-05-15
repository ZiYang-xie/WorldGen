import { invoke } from "@tauri-apps/api/core";

export function TitleBar() {
  return (
    <div className="titlebar">
      <div className="titlebar-controls">
        <button className="titlebar-btn close"  onClick={() => invoke("window_close")}   aria-label="Close"    />
        <button className="titlebar-btn min"    onClick={() => invoke("window_minimize")} aria-label="Minimize" />
        <button className="titlebar-btn max"    onClick={() => invoke("window_maximize")} aria-label="Maximize" />
      </div>
      <span className="titlebar-title">Auroch Syna</span>
    </div>
  );
}
