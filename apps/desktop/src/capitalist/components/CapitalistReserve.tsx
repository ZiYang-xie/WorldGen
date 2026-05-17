const POSITIONS = [
  { asset: 'US Treasury 10Y',   value: 2450000,  pnl: 12400,  pnlPct: 0.51,  allocation: 32 },
  { asset: 'S&P 500 Index',     value: 1820000,  pnl: 48200,  pnlPct: 2.72,  allocation: 24 },
  { asset: 'Private Equity A',  value: 1200000,  pnl: 0,      pnlPct: 0,     allocation: 16 },
  { asset: 'Gold Reserve',      value: 680000,   pnl: 18500,  pnlPct: 2.80,  allocation: 9 },
  { asset: 'Crypto Basket',     value: 420000,   pnl: -12300, pnlPct: -2.84, allocation: 6 },
  { asset: 'Cash (USD)',        value: 980000,   pnl: 4200,   pnlPct: 0.43,  allocation: 13 },
];

export function CapitalistReserve() {
  const totalAUM = POSITIONS.reduce((sum, p) => sum + p.value, 0);
  const totalPnl = POSITIONS.reduce((sum, p) => sum + p.pnl, 0);
  const dailyChange = 1.42;

  return (
    <div className="ac-content">
      <div className="ac-content-header">
        <h1 className="ac-content-title">Reserve</h1>
        <p className="ac-content-subtitle">Private finance OS — capital allocation</p>
      </div>

      <div className="ac-content-body">
        {/* AUM Header */}
        <div className="ac-card" style={{ marginBottom: 32, background: 'var(--ac-ivory)' }}>
          <div className="ac-card-label">Total Assets Under Management</div>
          <div className="ac-card-value">${(totalAUM / 1000000).toFixed(2)}M</div>
          <div className="ac-card-change positive">
            +${(totalPnl / 1000).toFixed(0)}K ({dailyChange >= 0 ? '+' : ''}{dailyChange}%) today
          </div>
        </div>

        <div className="ac-grid ac-grid-2">
          {/* Positions */}
          <div className="ac-card" style={{ padding: 0, overflow: 'hidden' }}>
            <div style={{ padding: '16px 20px', borderBottom: '1px solid var(--ac-pearl)' }}>
              <div className="ac-card-label" style={{ margin: 0 }}>Positions</div>
            </div>
            <table className="ac-table">
              <thead>
                <tr>
                  <th>Asset</th>
                  <th>Value</th>
                  <th>P&L</th>
                  <th>Alloc</th>
                </tr>
              </thead>
              <tbody>
                {POSITIONS.map((pos) => (
                  <tr key={pos.asset}>
                    <td style={{ fontWeight: 500, color: 'var(--ac-blue-deep)' }}>{pos.asset}</td>
                    <td className="mono">${(pos.value / 1000).toFixed(0)}K</td>
                    <td className={`mono ${pos.pnl >= 0 ? 'positive' : 'negative'}`}>
                      {pos.pnl >= 0 ? '+' : ''}{pos.pnlPct.toFixed(2)}%
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{
                          width: 60, height: 4, background: 'var(--ac-pearl)', borderRadius: 2, overflow: 'hidden'
                        }}>
                          <div style={{
                            width: `${pos.allocation}%`, height: '100%',
                            background: 'var(--ac-gold)', borderRadius: 2
                          }} />
                        </div>
                        <span className="mono" style={{ fontSize: 11 }}>{pos.allocation}%</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Capital Flow */}
          <div className="ac-card">
            <div className="ac-card-label">Capital Flow — 30 Days</div>
            <div style={{ marginTop: 16 }}>
              <div className="ac-metric-row">
                <span className="ac-metric-label">Inflows</span>
                <span className="ac-metric-value" style={{ color: 'var(--ac-green)' }}>+$342K</span>
              </div>
              <div className="ac-metric-row">
                <span className="ac-metric-label">Outflows</span>
                <span className="ac-metric-value" style={{ color: 'var(--ac-red)' }}>-$128K</span>
              </div>
              <div className="ac-metric-row">
                <span className="ac-metric-label">Net Flow</span>
                <span className="ac-metric-value" style={{ color: 'var(--ac-green)' }}>+$214K</span>
              </div>
              <div className="ac-metric-row">
                <span className="ac-metric-label">Yield (30d)</span>
                <span className="ac-metric-value">4.82%</span>
              </div>
              <div className="ac-metric-row">
                <span className="ac-metric-label">Sharpe Ratio</span>
                <span className="ac-metric-value">1.84</span>
              </div>
            </div>

            <div style={{ marginTop: 24 }}>
              <button className="ac-btn ac-btn-primary" style={{ width: '100%' }}>
                Rebalance Portfolio
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
