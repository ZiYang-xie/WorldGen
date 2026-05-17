import { useState } from 'react';

const WATCHLIST = [
  { symbol: 'AAPL',  price: 198.42, change: 1.24,  volume: '52.3M' },
  { symbol: 'TSLA',  price: 247.18, change: -0.83, volume: '89.1M' },
  { symbol: 'NVDA',  price: 892.50, change: 3.41,  volume: '41.7M' },
  { symbol: 'MSFT',  price: 417.88, change: 0.67,  volume: '28.4M' },
  { symbol: 'AMZN',  price: 186.51, change: -0.32, volume: '45.9M' },
  { symbol: 'GOOGL', price: 174.13, change: 1.89,  volume: '31.2M' },
];

const TICKER = [
  { symbol: 'AAPL', price: 198.42, change: 1.24 },
  { symbol: 'TSLA', price: 247.18, change: -0.83 },
  { symbol: 'NVDA', price: 892.50, change: 3.41 },
  { symbol: 'MSFT', price: 417.88, change: 0.67 },
  { symbol: 'BTC',  price: 67420,  change: 2.15 },
  { symbol: 'ETH',  price: 3580,   change: -1.02 },
  { symbol: 'GLD',  price: 234.80, change: 0.44 },
];

export function CapitalistTerminal() {
  const [query, setQuery] = useState('');

  return (
    <div className="ac-content">
      {/* Ticker Strip */}
      <div className="ac-ticker">
        {TICKER.map((item, i) => (
          <span key={item.symbol} className={`ac-ticker-item ${i === 0 ? 'watched' : ''}`}>
            {item.symbol}
            <span className="mono">{item.price.toLocaleString()}</span>
            <span className={item.change >= 0 ? 'positive' : 'negative'}>
              {item.change >= 0 ? '▲' : '▼'}{Math.abs(item.change)}%
            </span>
            {i < TICKER.length - 1 && <span className="ac-ticker-sep">│</span>}
          </span>
        ))}
      </div>

      {/* Header */}
      <div className="ac-content-header">
        <h1 className="ac-content-title">Terminal</h1>
        <p className="ac-content-subtitle">Market intelligence — real-time</p>
      </div>

      {/* Body */}
      <div className="ac-content-body">
        {/* Metrics Row */}
        <div className="ac-grid ac-grid-4" style={{ marginBottom: 32 }}>
          <div className="ac-card">
            <div className="ac-card-label">S&P 500</div>
            <div className="ac-card-value">5,234.18</div>
            <div className="ac-card-change positive">+1.24% today</div>
          </div>
          <div className="ac-card">
            <div className="ac-card-label">NASDAQ</div>
            <div className="ac-card-value">16,742.39</div>
            <div className="ac-card-change positive">+1.87% today</div>
          </div>
          <div className="ac-card">
            <div className="ac-card-label">VIX</div>
            <div className="ac-card-value">13.42</div>
            <div className="ac-card-change negative">-4.2% today</div>
          </div>
          <div className="ac-card">
            <div className="ac-card-label">10Y Yield</div>
            <div className="ac-card-value">4.28%</div>
            <div className="ac-card-change positive">+0.03 today</div>
          </div>
        </div>

        {/* Watchlist Table */}
        <div className="ac-card" style={{ padding: 0, overflow: 'hidden' }}>
          <table className="ac-table">
            <thead>
              <tr>
                <th>Asset</th>
                <th>Price</th>
                <th>Change</th>
                <th>Volume</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {WATCHLIST.map((row) => (
                <tr key={row.symbol}>
                  <td style={{ fontWeight: 600, color: 'var(--ac-blue-deep)' }}>{row.symbol}</td>
                  <td className="mono">{row.price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                  <td className={`mono ${row.change >= 0 ? 'positive' : 'negative'}`}>
                    {row.change >= 0 ? '+' : ''}{row.change}%
                  </td>
                  <td className="mono">{row.volume}</td>
                  <td>
                    <button className="ac-btn ac-btn-ghost" style={{ fontSize: 11 }}>
                      Details →
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Query Input */}
        <div style={{ marginTop: 32 }}>
          <div className="ac-input-group">
            <label className="ac-input-label">Intelligence Query</label>
            <input
              className="ac-input"
              placeholder="Show me tech momentum this week…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="ac-btn ac-btn-primary">Execute</button>
            <button className="ac-btn ac-btn-secondary">Save Query</button>
          </div>
        </div>
      </div>
    </div>
  );
}
