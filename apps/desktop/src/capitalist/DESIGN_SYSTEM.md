# Auroch Capitalist — UX Design System

> "Wealth, movement, and intelligence are one system."

---

## 1. Philosophy

Auroch Capitalist is the executive mobility and finance intelligence line of Auroch. The UX must communicate:

- **Institutional trust** — calm, authoritative, unhurried
- **Capital as infrastructure** — wealth as power plant, not destination
- **Motion as sovereignty** — the world moves below you
- **Beauty as discipline** — every pixel is a structural decision

Not a dashboard. Not a trading terminal. A **command vessel**.

---

## 2. Visual Identity

### 2.1 Palette

| Token | Value | Role |
|---|---|---|
| `--ac-white` | `#FAFBFC` | Primary surface, institutional canvas |
| `--ac-ivory` | `#F2F4F6` | Secondary surface, card backgrounds |
| `--ac-pearl` | `#E8EAED` | Borders, dividers, subtle structure |
| `--ac-gold` | `#C9A84C` | Primary accent, authority, capital |
| `--ac-gold-light` | `#E8D48B` | Hover states, secondary gold |
| `--ac-gold-dim` | `rgba(201,168,76,0.12)` | Gold wash, active indicators |
| `--ac-blue-deep` | `#1B3A5C` | Primary text, deep intelligence |
| `--ac-blue` | `#2B5B84` | Secondary text, links, navigation |
| `--ac-blue-light` | `#4A8BB5` | Tertiary text, muted states |
| `--ac-blue-glass` | `rgba(27,58,92,0.06)` | Glass overlay, depth layers |
| `--ac-mercury` | `#B8C0C8` | Structural chrome, separators |
| `--ac-mercury-light` | `#D4D8DC` | Subtle chrome, inactive states |
| `--ac-silver` | `#8A9199` | Muted data, secondary metrics |
| `--ac-green` | `#2D9B6E` | Positive motion, growth |
| `--ac-red` | `#B8433E` | Caution, decline, friction |
| `--ac-amber` | `#C49A3C` | Warning, pending, transitional |

### 2.2 Typography

| Role | Font | Weight | Size | Letter-spacing |
|---|---|---|---|---|
| Display | SF Pro Display / Inter | 600 | 28-36px | -0.02em |
| Heading | SF Pro Text / Inter | 600 | 16-20px | -0.01em |
| Body | SF Pro Text / Inter | 400 | 14px | 0 |
| Label | SF Pro Text / Inter | 500 | 11-12px | 0.06em (uppercase) |
| Mono | SF Mono / JetBrains Mono | 400 | 12-13px | 0 |
| Ticker | SF Mono / JetBrains Mono | 500 | 11px | 0.04em |

### 2.3 Spacing

Base unit: **4px**

| Token | Value |
|---|---|
| `--ac-space-1` | 4px |
| `--ac-space-2` | 8px |
| `--ac-space-3` | 12px |
| `--ac-space-4` | 16px |
| `--ac-space-5` | 24px |
| `--ac-space-6` | 32px |
| `--ac-space-7` | 48px |
| `--ac-space-8` | 64px |

### 2.4 Radius

| Token | Value | Use |
|---|---|---|
| `--ac-radius-sm` | 4px | Buttons, inputs, tags |
| `--ac-radius` | 8px | Cards, panels |
| `--ac-radius-lg` | 12px | Modals, large surfaces |
| `--ac-radius-xl` | 16px | Hero surfaces |

### 2.5 Shadows

```
--ac-shadow-sm:  0 1px 3px rgba(27,58,92,0.08)
--ac-shadow:     0 4px 12px rgba(27,58,92,0.10)
--ac-shadow-lg:  0 8px 32px rgba(27,58,92,0.14)
--ac-shadow-gold: 0 2px 12px rgba(201,168,76,0.18)
```

---

## 3. Layout Philosophy

### 3.1 Principles

1. **Sunlit, not dark** — white surfaces with blue depth, never black voids
2. **Chrome structure** — mercury-silver lines create architecture, not decoration
3. **Gold is earned** — accent used sparingly, only for authority and action
4. **Data breathes** — generous whitespace, information density through hierarchy not crowding
5. **Motion is quiet** — transitions are smooth, deliberate, never flashy

### 3.2 Grid

- Main layout: **sidebar + content** (280px sidebar, fluid content)
- Content grid: **12-column** with 24px gutters
- Card padding: **24px** standard, **16px** compact
- Section spacing: **32px** between major blocks

### 3.3 Navigation

```
┌─────────────────────────────────────────────────────┐
│  AUROCH CAPITALIST                    [● Core Active]│
├────────┬────────────────────────────────────────────┤
│        │                                            │
│  NAV   │              CONTENT AREA                  │
│        │                                            │
│  ◆ Terminal    ┌──────────────────────────────────┐ │
│  ◇ Reserve     │                                  │ │
│  ◇ Dispatch    │         Primary View             │ │
│  ◇ Fleet       │                                  │ │
│  ◇ Core        └──────────────────────────────────┘ │
│        │                                            │
│  ──────│──── Status Bar ─────────────────────────── │
│        │  Hermes Core: Active │ Latency: 12ms      │
└────────┴────────────────────────────────────────────┘
```

Navigation items:
- **Terminal** — Bloomberg-style intelligence interface (default)
- **Reserve** — Private finance OS, portfolio, positions
- **Dispatch** — Markets/news narrator, AVN Market General
- **Fleet** — Hermes Drive mobility platform, vehicle status
- **Core** — System settings, Argent Core diagnostics

---

## 4. Component Patterns

### 4.1 Cards

```
┌─────────────────────────────┐
│  LABEL (uppercase, muted)   │
│                             │
│  Primary content            │
│  in deep blue text          │
│                             │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│  Secondary info  │  Metric  │
└─────────────────────────────┘
```

- White/ivory background
- Pearl border (1px)
- Mercury separator for footer
- Gold left-border on active/selected state

### 4.2 Data Tables (Terminal)

```
┌────────┬──────────┬──────────┬──────────┐
│ ASSET  │ PRICE    │ CHANGE   │ VOLUME   │
├────────┼──────────┼──────────┼──────────┤
│ AAPL   │ 198.42   │ +1.24%   │ 52.3M    │
│ TSLA   │ 247.18   │ -0.83%   │ 89.1M    │
│ NVDA   │ 892.50   │ +3.41%   │ 41.7M    │
└────────┴──────────┴──────────┴──────────┘
```

- Header: label style, uppercase, mercury bottom border
- Rows: alternating ivory/white
- Positive: green text
- Negative: red text
- Hover: gold-dim background wash
- Monospace for all numeric data

### 4.3 Ticker Strip

```
─ AAPL 198.42 ▲1.24% │ TSLA 247.18 ▼0.83% │ NVDA 892.50 ▲3.41% ─
```

- Fixed top or bottom bar
- Mono font, 11px
- Mercury separators between items
- Subtle scroll animation
- Gold accent on user's watched items

### 4.4 Status Indicators

```
● Active    ○ Standby    ◐ Syncing    ✕ Offline
```

- Core status always visible in header
- Mercury ring with colored fill
- Green = active, Amber = transitional, Red = offline
- Pulse animation on state change (subtle, 2s)

### 4.5 Buttons

**Primary (Gold)**
```
┌─────────────────┐
│   Execute       │  ← gold bg, white text
└─────────────────┘
```

**Secondary (Chrome)**
```
┌─────────────────┐
│   Configure     │  ← white bg, pearl border, blue text
└─────────────────┘
```

**Tertiary (Ghost)**
```
  Details →        ← no bg, blue text, hover gold
```

### 4.6 Inputs

```
┌─────────────────────────────┐
│ Label (uppercase, muted)    │
│ ─────────────────────────── │
│ User input text...          │
└─────────────────────────────┘
```

- White background
- Pearl border, 1px
- Focus: gold border + gold-dim shadow
- Mono for numeric/financial inputs
- Deep blue placeholder text

### 4.7 Core Visualization (Argent Core)

```
         ╭─────╮
      ╭──╢     ╠──╮
     ╱    ║  ◉  ║    ╲
    │     ║     ║     │
     ╲    ║     ║    ╱
      ╰──╢     ╠──╯
         ╰─────╯
```

- Mercury-silver ring structure
- Gold inner glow when active
- Slow rotation animation (8s full cycle)
- Pulse intensity based on load
- Located in Core panel, subtle in header

---

## 5. Product Division UX

### 5.1 Terminal (Intelligence Interface)

**Purpose:** Bloomberg-style market intelligence

**Layout:**
- Top: Ticker strip with watched assets
- Left: Watchlist / saved queries
- Center: Main chart + data table
- Right: Dispatch feed (narrated insights)
- Bottom: Command input for queries

**Key interactions:**
- Type queries naturally: "Show me tech momentum this week"
- Charts render in blue/gold palette
- Data tables with sortable columns
- Right-click for context actions (add to watchlist, export)

### 5.2 Reserve (Private Finance OS)

**Purpose:** Portfolio management, positions, capital allocation

**Layout:**
- Top: Total AUM, daily change, performance period selector
- Left: Asset allocation pie / tree map
- Center: Position list with P&L
- Right: Capital flow timeline

**Key interactions:**
- Drag to rebalance allocation
- Click position for detail drawer
- Timeline scrubber for historical view
- Gold highlight on active positions

### 5.3 Dispatch (Markets/News Narrator)

**Purpose:** AVN Market General — narrated market intelligence

**Layout:**
- Full-width narrative feed
- Each entry: timestamp, headline, narrative body, related assets
- Filter by: macro, sector, geopolitical, crypto
- Voice toggle for audio narration

**Key interactions:**
- Scroll for continuous narrative
- Tap asset tags to jump to Terminal
- Bookmark for later review
- Share as structured brief

### 5.4 Fleet (Hermes Drive Mobility)

**Purpose:** Vehicle status, routing, Hermes Core diagnostics

**Layout:**
- Top: Fleet overview (active vehicles, total range, next departure)
- Left: Vehicle list with status
- Center: Route map / 3D vessel view
- Right: Core diagnostics (mercury spin, power, temp)

**Key interactions:**
- Select vehicle for detail view
- Route planning with ETA
- Core health monitoring
- Gold accent on active vessel

### 5.5 Core (System Settings)

**Purpose:** Argent Core configuration, system status

**Layout:**
- Core visualization (center, large)
- System metrics grid around it
- Settings panels below
- Connection status, latency, throughput

---

## 6. Motion Design

### 6.1 Principles

- **Slow is smooth, smooth is fast** — 200-400ms transitions
- **Ease out** — decelerate into final state
- **No bounce** — professional, not playful
- **Gold on action** — gold flash on successful operations

### 6.2 Timings

| Action | Duration | Easing |
|---|---|---|
| Hover state | 150ms | ease-out |
| Panel transition | 250ms | ease-out |
| Modal open | 300ms | ease-out |
| Data refresh | 400ms | ease-in-out |
| Core pulse | 2000ms | ease-in-out infinite |
| Ticker scroll | continuous | linear |

### 6.3 Animations

```css
/* Core rotation */
@keyframes core-spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}

/* Status pulse */
@keyframes status-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(201,168,76,0.4); }
  50%      { box-shadow: 0 0 0 8px rgba(201,168,76,0); }
}

/* Gold flash on success */
@keyframes gold-flash {
  0%   { background: rgba(201,168,76,0.2); }
  100% { background: transparent; }
}

/* Data row enter */
@keyframes row-enter {
  from { opacity: 0; transform: translateY(-4px); }
  to   { opacity: 1; transform: translateY(0); }
}
```

---

## 7. Accessibility

- Minimum contrast ratio: **4.5:1** for body text, **3:1** for large text
- Gold on white: 2.8:1 — **use gold only on deep blue or as accent border**
- All interactive elements: minimum **44x44px** touch target
- Keyboard navigation: full tab order, visible focus rings (gold)
- Screen reader: all data tables have proper headers, status updates announced

---

## 8. Responsive Breakpoints

| Breakpoint | Width | Layout |
|---|---|---|
| Desktop | >1200px | Full sidebar + content |
| Tablet | 768-1200px | Collapsible sidebar |
| Mobile | <768px | Bottom nav, stacked content |

---

## 9. Canon Lines

Use these as copy guidelines:

- "The world moves below you."
- "Capital as engine. Intelligence as leverage."
- "Motion is sovereignty."
- "Wealth is the power plant."
- "Beauty is infrastructure."
- "Auroch Capitalist: the financial engine of motion."

---

## 10. What This Is NOT

- ❌ Dark mode crypto bro aesthetic
- ❌ Wall Street green/red chaos
- ❌ Black/gold throne room
- ❌ Cluttered Bloomberg terminal from 2003
- ❌ Gamified trading app
- ❌ Startup dashboard with too many charts

## What This IS

- ✅ Sunlit empire — white, gold, blue
- ✅ Chrysler Building chrome — structural elegance
- ✅ Private aviation — calm authority
- ✅ Apple hardware — precision, restraint
- ✅ Ocean liner first-class — space, comfort, permanence
- ✅ 1930s financial district — institutional gravity
