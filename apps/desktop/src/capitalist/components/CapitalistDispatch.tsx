const DISPATCH_ITEMS = [
  {
    time: '14:32 UTC',
    headline: 'Fed signals potential rate adjustment in September meeting',
    body: 'Markets responded with measured optimism as Chair Powell indicated data-dependent flexibility. Treasury yields compressed 4bps on the news, while rate futures priced in a 68% probability of a 25bp cut. The dollar index softened to 104.2.',
    tags: ['macro', 'fed', 'rates'],
  },
  {
    time: '13:18 UTC',
    headline: 'Semiconductor supply chain consolidation accelerates',
    body: 'Three major foundry partnerships announced this week signal a structural shift in chip manufacturing geography. NVDA and TSMC joint capacity commitments extend through 2028. Capital expenditure in the sector now exceeds $180B annually.',
    tags: ['sector', 'semiconductors', 'capex'],
  },
  {
    time: '11:45 UTC',
    headline: 'Sovereign wealth funds increase allocation to private infrastructure',
    body: 'Middle Eastern and Asian SWFs deployed $42B into infrastructure assets in Q2, a 34% increase year-over-year. Transport, energy transition, and digital infrastructure remain the primary targets. Auroch Fleet mobility assets are among the tracked benchmarks.',
    tags: ['geopolitical', 'infrastructure', 'swf'],
  },
  {
    time: '09:20 UTC',
    headline: 'Bitcoin ETF flows reverse after three-day outflow streak',
    body: 'Institutional inflows returned to BTC ETFs with $280M net positive yesterday. The move coincides with traditional market strength and suggests risk-on sentiment is broadening beyond equities. ETH ETFs saw modest inflows of $42M.',
    tags: ['crypto', 'etf', 'flows'],
  },
];

export function CapitalistDispatch() {
  return (
    <div className="ac-content">
      <div className="ac-content-header">
        <h1 className="ac-content-title">Dispatch</h1>
        <p className="ac-content-subtitle">Market General — narrated intelligence</p>
      </div>

      <div className="ac-content-body" style={{ maxWidth: 720 }}>
        {/* Filter Bar */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 32 }}>
          {['All', 'Macro', 'Sector', 'Geopolitical', 'Crypto'].map((filter, i) => (
            <button
              key={filter}
              className={`ac-btn ${i === 0 ? 'ac-btn-primary' : 'ac-btn-secondary'}`}
              style={{ fontSize: 12, padding: '6px 14px' }}
            >
              {filter}
            </button>
          ))}
        </div>

        {/* Feed */}
        <div>
          {DISPATCH_ITEMS.map((item) => (
            <div key={item.time} className="ac-dispatch-item">
              <div className="ac-dispatch-time">{item.time}</div>
              <h3 className="ac-dispatch-headline">{item.headline}</h3>
              <p className="ac-dispatch-body">{item.body}</p>
              <div className="ac-dispatch-tags">
                {item.tags.map((tag) => (
                  <span key={tag} className="ac-dispatch-tag">{tag}</span>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Narration Toggle */}
        <div className="ac-card" style={{ marginTop: 32, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <div className="ac-card-label" style={{ marginBottom: 4 }}>Audio Narration</div>
            <div style={{ fontSize: 13, color: 'var(--ac-blue)' }}>
              Market General voice briefing — 4 items, ~6 min
            </div>
          </div>
          <button className="ac-btn ac-btn-primary">
            ▶ Play Briefing
          </button>
        </div>
      </div>
    </div>
  );
}
