import type { CapitalistView } from '../types';
import { NAV_ITEMS } from '../types';

interface Props {
  active: CapitalistView;
  onSelect: (view: CapitalistView) => void;
}

export function CapitalistNav({ active, onSelect }: Props) {
  return (
    <nav className="ac-sidebar">
      <div className="ac-nav-section">
        <div className="ac-nav-label">Command</div>
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`ac-nav-item ${active === item.id ? 'active' : ''}`}
            onClick={() => onSelect(item.id)}
          >
            <span className="ac-nav-icon">{item.icon}</span>
            {item.label}
            {item.badge && <span className="ac-nav-badge">{item.badge}</span>}
          </button>
        ))}
      </div>

      <div className="ac-nav-section" style={{ marginTop: 'auto', paddingTop: 16 }}>
        <div className="ac-tagline">
          "The world moves below you."
        </div>
      </div>
    </nav>
  );
}
