const SYSTEM_METRICS = [
  { label: 'Core Temperature', value: '42.3°C', status: 'normal' },
  { label: 'Mercury Spin Rate', value: '14,400 RPM', status: 'normal' },
  { label: 'Power Draw', value: '2.4 kW', status: 'normal' },
  { label: 'Throughput', value: '847 TFLOPS', status: 'normal' },
  { label: 'Latency (p99)', value: '12ms', status: 'normal' },
  { label: 'Memory', value: '64 / 128 GB', status: 'normal' },
  { label: 'Uptime', value: '47d 12h 33m', status: 'normal' },
  { label: 'Last Sync', value: '2s ago', status: 'normal' },
];

export function CapitalistCore() {
  return (
    <div className="ac-content">
      <div className="ac-content-header">
        <h1 className="ac-content-title">Core</h1>
        <p className="ac-content-subtitle">Argent Core diagnostics — system status</p>
      </div>

      <div className="ac-content-body">
        {/* Core Visualization */}
        <div className="ac-card" style={{ textAlign: 'center', padding: '48px 24px', marginBottom: 32 }}>
          <div className="ac-core-viz" style={{ width: 160, height: 160 }}>
            <div className="ac-core-ring" style={{ animationDuration: '6s' }} />
            <div className="ac-core-ring-inner" style={{ animationDuration: '10s' }} />
            <div className="ac-core-center" style={{ inset: 40 }} />
          </div>
          <div style={{ marginTop: 24 }}>
            <div className="ac-card-label">Argent Core III</div>
            <div style={{ fontSize: 16, fontWeight: 600, color: 'var(--ac-blue-deep)' }}>
              All Systems Nominal
            </div>
            <div style={{ marginTop: 8 }}>
              <span className="ac-status ac-status-active">
                <span className="ac-status-dot" />
                Active
              </span>
            </div>
          </div>
        </div>

        {/* System Metrics */}
        <div className="ac-grid ac-grid-2">
          <div className="ac-card">
            <div className="ac-card-label">Performance</div>
            {SYSTEM_METRICS.slice(0, 4).map((m) => (
              <div key={m.label} className="ac-metric-row">
                <span className="ac-metric-label">{m.label}</span>
                <span className="ac-metric-value">{m.value}</span>
              </div>
            ))}
          </div>

          <div className="ac-card">
            <div className="ac-card-label">System</div>
            {SYSTEM_METRICS.slice(4).map((m) => (
              <div key={m.label} className="ac-metric-row">
                <span className="ac-metric-label">{m.label}</span>
                <span className="ac-metric-value">{m.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Actions */}
        <div style={{ marginTop: 32, display: 'flex', gap: 12 }}>
          <button className="ac-btn ac-btn-secondary">Run Diagnostics</button>
          <button className="ac-btn ac-btn-secondary">Export Logs</button>
          <button className="ac-btn ac-btn-secondary" style={{ marginLeft: 'auto' }}>
            Configure Core
          </button>
        </div>
      </div>
    </div>
  );
}
