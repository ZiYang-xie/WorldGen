import { useState } from "react";
import { useDaemon } from "./hooks/useDaemon";
import { TitleBar } from "./components/TitleBar";
import { DaemonStatus } from "./components/DaemonStatus";
import { GeneratePanel } from "./components/GeneratePanel";
import { SceneViewer } from "./components/SceneViewer";
import { CapitalistApp } from "./capitalist";

type Theme = "syna" | "capitalist";

export function App() {
  const daemon = useDaemon();
  const [log, setLog] = useState<string[]>([]);
  const [theme, setTheme] = useState<Theme>("capitalist");

  function addLog(msg: string) {
    setLog((prev) => [...prev.slice(-200), msg]);
  }

  if (theme === "capitalist") {
    return <CapitalistApp />;
  }

  return (
    <div className="app">
      <TitleBar />
      <div className="app-body">
        <aside className="sidebar">
          <div style={{ display: 'flex', gap: 4, marginBottom: 8 }}>
            <button className="btn secondary" style={{ flex: 1, fontSize: 11 }} onClick={() => setTheme("capitalist")}>
              Capitalist
            </button>
          </div>
          <DaemonStatus
            status={daemon.status}
            onStart={daemon.start}
            onStop={daemon.stop}
            lastError={daemon.lastError}
          />
          <GeneratePanel daemon={daemon} onResult={addLog} />
          {log.length > 0 && (
            <div className="card" style={{ flex: 1 }}>
              <div className="card-title">Log</div>
              <div className="log">
                {log.map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
              </div>
            </div>
          )}
        </aside>
        <main className="main-panel">
          <SceneViewer daemonRunning={daemon.status === "running"} />
        </main>
      </div>
    </div>
  );
}
