export type CapitalistView = 'terminal' | 'reserve' | 'dispatch' | 'fleet' | 'core';

export interface NavItem {
  id: CapitalistView;
  label: string;
  icon: string;
  badge?: string;
}

export const NAV_ITEMS: NavItem[] = [
  { id: 'terminal',  label: 'Terminal',  icon: '◈', badge: 'LIVE' },
  { id: 'reserve',   label: 'Reserve',   icon: '◉' },
  { id: 'dispatch',  label: 'Dispatch',  icon: '◇', badge: '3' },
  { id: 'fleet',     label: 'Fleet',     icon: '△' },
  { id: 'core',      label: 'Core',      icon: '◎' },
];
