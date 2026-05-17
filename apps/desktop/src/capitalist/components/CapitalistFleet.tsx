import { useState } from 'react';

const VEHICLES = [
  { id: 'AC-001', name: 'Capitalist One',    status: 'active',   route: 'NYC → London',   eta: '4h 12m', core: 'Hermes III',  range: '12,400 km' },
  { id: 'AC-002', name: 'Reserve Wing',      status: 'active',   route: 'Tokyo → Dubai',  eta: '8h 45m', core: 'Hermes II',   range: '9,800 km' },
  { id: 'AC-003', name: 'Dispatch Runner',   status: 'standby',  route: '—',              eta: '—',      core: 'Hermes III',  range: '12,400 km' },
  { id: 'AC-004', name: 'Terminal Express',  status: 'active',   route: 'SF → Singapore', eta: '11h 20m', core: 'Hermes II',   range: '9,800 km' },
];

export function CapitalistFleet() {
  const [selected, setSelected] = useState(0);
  const v = VEHICLES[selected];

  return (
    <div className="ac-content">
      <div className="ac-content-header">
        <h1 className="ac-content-title">Fleet</h1>
        <p className="ac-content-subtitle">Hermes Drive — private mobility platform</p>
      </div>

      <div className="ac-content-body">
        {/* Fleet Overview */}
        <div className="ac-grid ac-grid-4" style={{ marginBottom: 32 }}>
          <div className="ac-card">
            <div className="ac-card-label">Active Vessels</div>
            <div className="ac-card-value">3</div>
            <div className="ac-card-change" style={{ color: 'var(--ac-silver)' }}>of 4 total</div>
          </div>
          <div className="ac-card">
            <div className="ac-card-label">Combined Range</div>
            <div className="ac-card-value">44.6K</div>
            <div className="ac-card-change" style={{ color: 'var(--ac-silver)' }}>kilometers</div>
          </div>
          <div className="ac-card">
            <div className="ac-card-label">Next Departure</div>
            <div className="ac-card-value" style={{ fontSize: 22 }}>16:30</div>
            <div className="ac-card-change" style={{ color: 'var(--ac-silver)' }}>UTC — AC-003</div>
          </div>
          <div className="ac-card">
            <div className="ac-card-label">Core Health</div>
            <div className="ac-card-value" style={{ color: 'var(--ac-green)' }}>98%</div>
            <div className="ac-card-change positive">All nominal</div>
          </div>
        </div>

        <div className="ac-grid ac-grid-2">
          {/* Vehicle List */}
          <div>
            <div className="ac-card-label" style={{ marginBottom: 16 }}>Vessels</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {VEHICLES.map((vehicle, i) => (
                <div
                  key={vehicle.id}
                  className={`ac-vehicle-card ${selected === i ? 'active' : ''}`}
                  onClick={() => setSelected(i)}
                >
                  <div className="ac-vehicle-icon">△</div>
                  <div className="ac-vehicle-info">
                    <div className="ac-vehicle-name">{vehicle.name}</div>
                    <div className="ac-vehicle-detail">
                      {vehicle.status === 'active' ? `${vehicle.route} · ETA ${vehicle.eta}` : 'Standby — Ready'}
                    </div>
                  </div>
                  <span className={`ac-status ${vehicle.status === 'active' ? 'ac-status-active' : 'ac-status-standby'}`}>
                    <span className="ac-status-dot" />
                    {vehicle.status === 'active' ? 'En Route' : 'Standby'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Selected Vehicle Detail */}
          <div className="ac-card">
            <div className="ac-card-label">Vessel Detail</div>
            <div style={{ textAlign: 'center', padding: '16px 0' }}>
              <div className="ac-core-viz">
                <div className="ac-core-ring" />
                <div className="ac-core-ring-inner" />
                <div className="ac-core-center" />
              </div>
            </div>

            <div style={{ marginTop: 16 }}>
              <div className="ac-metric-row">
                <span className="ac-metric-label">ID</span>
                <span className="ac-metric-value">{v.id}</span>
              </div>
              <div className="ac-metric-row">
                <span className="ac-metric-label">Core</span>
                <span className="ac-metric-value">{v.core}</span>
              </div>
              <div className="ac-metric-row">
                <span className="ac-metric-label">Route</span>
                <span className="ac-metric-value">{v.route}</span>
              </div>
              <div className="ac-metric-row">
                <span className="ac-metric-label">ETA</span>
                <span className="ac-metric-value">{v.eta}</span>
              </div>
              <div className="ac-metric-row">
                <span className="ac-metric-label">Max Range</span>
                <span className="ac-metric-value">{v.range}</span>
              </div>
            </div>

            <div style={{ marginTop: 24, display: 'flex', gap: 8 }}>
              <button className="ac-btn ac-btn-primary" style={{ flex: 1 }}>
                Request Vessel
              </button>
              <button className="ac-btn ac-btn-secondary">
                Configure
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
