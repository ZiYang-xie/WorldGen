import type { DaemonStatus } from "../hooks/useDaemon";

interface Props {
  status: DaemonStatus;
  onStart: () => void;
  onStop: () => void;
  lastError: string | null;
}

const DOT: Record<DaemonStatus, string> = {
  running: "green",
  starting: "amber",
  stopped: "red",
  error: "red",
};

const LABEL: Record<DaemonStatus, string> = {
  running: "Daemon running",
  starting: "Starting…",
  stopped: "Daemon stopped",
  error: "Daemon error",
};

export function DaemonStatus({ status, onStart, onStop, lastError }: Props) {
  return (
    <div className="card">
      <div className="card-title">Engine</div>
      <div className="status-row" style={{ marginBottom: 10 }}>
        <span className={`dot ${DOT[status]}`} />
        <span style={{ fontSize: 13 }}>{LABEL[status]}</span>
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <button className="btn secondary" onClick={onStart}
          disabled={status === "running" || status === "starting"}
          style={{ flex: 1, fontSize: 12 }}>
          Start
        </button>
        <button className="btn secondary" onClick={onStop}
          disabled={status !== "running"}
          style={{ flex: 1, fontSize: 12 }}>
          Stop
        </button>
      </div>
      {lastError && (
        <div className="log" style={{ marginTop: 8, color: "#f87171" }}>{lastError}</div>
      )}
    </div>
  );
}
