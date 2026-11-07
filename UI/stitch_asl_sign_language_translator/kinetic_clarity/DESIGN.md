---
name: Kinetic Clarity
colors:
  surface: '#0b1326'
  surface-dim: '#0b1326'
  surface-bright: '#31394d'
  surface-container-lowest: '#060e20'
  surface-container-low: '#131b2e'
  surface-container: '#171f33'
  surface-container-high: '#222a3d'
  surface-container-highest: '#2d3449'
  on-surface: '#dae2fd'
  on-surface-variant: '#c7c4d7'
  inverse-surface: '#dae2fd'
  inverse-on-surface: '#283044'
  outline: '#908fa0'
  outline-variant: '#464554'
  surface-tint: '#c0c1ff'
  primary: '#c0c1ff'
  on-primary: '#1000a9'
  primary-container: '#8083ff'
  on-primary-container: '#0d0096'
  inverse-primary: '#494bd6'
  secondary: '#4edea3'
  on-secondary: '#003824'
  secondary-container: '#00a572'
  on-secondary-container: '#00311f'
  tertiary: '#ffb95f'
  on-tertiary: '#472a00'
  tertiary-container: '#ca8100'
  on-tertiary-container: '#3e2400'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e1e0ff'
  primary-fixed-dim: '#c0c1ff'
  on-primary-fixed: '#07006c'
  on-primary-fixed-variant: '#2f2ebe'
  secondary-fixed: '#6ffbbe'
  secondary-fixed-dim: '#4edea3'
  on-secondary-fixed: '#002113'
  on-secondary-fixed-variant: '#005236'
  tertiary-fixed: '#ffddb8'
  tertiary-fixed-dim: '#ffb95f'
  on-tertiary-fixed: '#2a1700'
  on-tertiary-fixed-variant: '#653e00'
  background: '#0b1326'
  on-background: '#dae2fd'
  surface-variant: '#2d3449'
typography:
  headline-xl:
    fontFamily: Inter
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Inter
    fontSize: 22px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 22px
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  label-lg:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: 0.02em
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.04em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '500'
    lineHeight: 14px
    letterSpacing: 0.06em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-lg: 1.5rem
  margin: 1.5rem
  margin-sm: 1rem
  margin-lg: 2rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1.25rem
  space-xl: 2rem
---

## Brand & Style

This design system establishes an assistive, high-precision computer vision workspace designed for real-time American Sign Language (ASL) transcription and bi-directional communication. The design aesthetic fuses technical precision with human-centric accessibility, avoiding clinical sterility in favor of an empowering, responsive digital interpreter suite. 

The aesthetic is grounded in modern dark workspace ergonomics: deep slate canvas surfaces paired with luminous indigo focal accents, crisp monospaced telemetry, and deliberate semantic feedback channels. Micro-interactions utilize soft, calibrated luminous glows to denote machine learning confidence tiers without triggering visual fatigue or cognitive overload. Every interactive element conforms to strict WCAG 2.2 AAA readability standards, high-contrast visual signifiers, and instant tactile feedback for screen readers and keyboard navigation.

## Colors

The palette operates on a strict functional hierarchy built on low-glare dark slate surfaces, ensuring continuous camera-feed monitoring without eye strain while maximizing chromatic contrast for real-time semantic states.

- **Workspace Base (Neutral - `#0F172A`)**: Deep slate foundation. Scaled into sub-surface steps: `#0A0F1D` (canvas floor), `#1E293B` (panel elevation), and `#334155` (structural partitions).
- **Core Action & Focus (Primary - `#6366F1`)**: Luminous Indigo. Denotes interactive controls, primary application commands, system navigations, and global states.
- **Active Gesture Buffer (Secondary - `#10B981`)**: Vivid Emerald. Dedicated strictly to real-time transient data: live skeletal joint tracking, active spatial buffers, streaming raw token recognition, and high-confidence spatial detection.
- **Formed Lexical Synthesis (Tertiary - `#F59E0B`)**: Radiant Amber. Dictates finalized gesture commitments, assembled grammatical clauses, word completions, and linguistic boundary confirmations.
- **Accessibility Safeguards**: Contrast ratios against slate surfaces exceed 7:1 for text-critical items and 4.5:1 for structural boundary lines. Color is never used as the sole state conveyer; distinct geometric iconography and token badges accompany all streaming states.

## Typography

The typography pairs **Inter** for readable, neutral UI messaging and transcription prose with **JetBrains Mono** for hardware metrics, spatial tracking coordinates, ML confidence scores, and token streams.

- **Inter**: Drives all communicative, translated outputs, sentence construction windows, dialogs, and navigation. Tight tracking at headline scales provides a structured, modern software feel, while regular body weights are scaled slightly up (15px/18px) to guarantee effortless legibility during split-attention video monitoring.
- **JetBrains Mono**: Strictly applied to telemetry, frame latency readouts, token buffers, confidence percentages, keyboard shortcuts, and capture mode indicators. The monospaced tabular numbers prevent layout jitter during dynamic inference updates.

## Layout & Spacing

The layout is architected as an integrated, multi-pane desktop application workspace using a modular 12-column adaptive fluid grid. The interface prioritizes continuous video viewpoint retention alongside live lexical streaming.

- **Application Framing**: The viewport centers on the primary camera feed / skeleton viewport (spanning 7–8 columns on desktop), bordered by a fixed collapsible telemetry dock (2 columns) and an interactive transcription stream / translation output pane (3–4 columns).
- **Breakpoint Behavior**:
  - **Desktop (>= 1200px)**: Side-by-side triad layout with persistent capture stage, live confidence analytics, and multi-sentence translation logs.
  - **Tablet / Split-Screen (768px – 1199px)**: 2-column stacked configuration; video capture anchors the upper viewport, transcription tokens flow below with collapsible HUD metrics.
  - **Compact / Picture-in-Picture (< 768px)**: Single-column priority view; video canvas features overlay HUD controls with transcription pinned as an anchored captions lower-third.
- **Rhythm**: Internal spacing relies strictly on an 8pt architectural rhythm (with a 4pt sub-scale for dense telemetry). Panels maintain a minimum internal padding of `space-lg` to create clear cognitive separation between visual tracking controls and linguistic readouts.

## Elevation & Depth

Visual depth is achieved through structured tonal layering and razor-thin boundaries rather than heavy drop shadows, keeping video performance high and visual noise minimal.

- **Base Layer (Elevation 0 - `#0A0F1D`)**: Main background canvas hosting low-priority viewport borders.
- **Workspace Surfaces (Elevation 1 - `#0F172A`)**: Camera container panels, side drawers, and transcription cards. Defined by a crisp border: `1px solid rgba(255, 255, 255, 0.08)`.
- **Raised Controls & Toolbars (Elevation 2 - `#1E293B`)**: Interactive playback controls, model switchers, and dropdowns. Features subtle ambient diffusion: `0 4px 16px -2px rgba(0, 0, 0, 0.5)`.
- **Popovers & Overlays (Elevation 3 - `#334155`)**: Telemetry overlays, diagnostic tooltips, and modal configs. Accompanied by a gentle indigo-tinted shadow: `0 12px 32px -4px rgba(99, 102, 241, 0.12), 0 8px 16px -2px rgba(0, 0, 0, 0.6)`.
- **Luminescent Confidence Glows**: When active, tracking nodes and dynamic chips emit diffuse, low-spread aura rings:
  - Streaming Gesture: `0 0 12px rgba(16, 185, 129, 0.35)`
  - Completed Lexeme: `0 0 12px rgba(245, 158, 11, 0.35)`
  - Active Target Lock: `0 0 16px rgba(99, 102, 241, 0.4)`

## Shapes

The geometric approach strikes a balance between professional software structure and comfortable readability. Surfaces and functional blocks use rounded corners (`0.5rem` / `8px`), softening the technical density of the workspace without sacrificing information real estate. 

- **Containers & Panes**: Standardized at `rounded-lg` (`1rem` / `16px`) to frame camera feeds and data streams.
- **Controls & Input Fields**: Configured at default roundedness (`0.5rem` / `8px`) for immediate button affordance.
- **Streaming Tokens & Badges**: Fully pill-shaped (`9999px`) to emphasize their transient, sequential nature as linguistic building blocks.

## Components

### Buttons & Interactive Controls
- **Primary Buttons**: Indigo fill (`#6366F1`) with high-contrast white text (`#FFFFFF`), `0.5rem` radius, and 1px inset highlight. Hover induces a gentle luminance boost and subtle indigo halo.
- **Assistive / High-Contrast Action**: Slate border with 2px offset focus rings in `#6366F1` when keyboard focused, never relying on low-contrast browser defaults.
- **Icon Actions**: 40x40px minimum target area for motor accessibility, featuring centered SVG iconography and tooltip labels bound to keyboard shortcuts.

### Gesture Tokens & Live Chips
- **Live Gesture Buffer Chips**: Rendered with an emerald-tinted backdrop (`rgba(16, 185, 129, 0.12)`), an emerald border (`#10B981`), and monospace labels. Features an animated pulsing indicator dot (`#10B981`) denoting dynamic parsing.
- **Completed Word Tokens**: Amber surface tint (`rgba(245, 158, 11, 0.12)`) framed with crisp amber borders (`#F59E0B`). Tokens dynamically slide and merge into the continuous transcription stream upon phrase resolution.

### Video Workspace Viewport
- **Camera Frame**: Inset rounded canvas with live keypoint mesh overlays. Includes corner-anchored HUD brackets that shift dynamically based on recognition state (Neutral Slate -> Tracking Emerald -> Confirmed Amber).
- **Status Indicator**: Pill badge overlay anchored at the upper-left of the feed, displaying real-time FPS, latency (ms), and active ML pipeline version in `label-sm` monospace.

### Input Fields & Search Bars
- Background surface `#1E293B` enclosed by `1px solid rgba(255, 255, 255, 0.12)`.
- Active focus displays a dual-tone border: inner bright indigo stroke (`#6366F1`) with an outer diffuse outline (`rgba(99, 102, 241, 0.25)`).

### Cards & Transcription Panels
- Structured with distinct semantic regions: a pinned header bar with speech synthesis toggle and export triggers, an expandable scrollable body containing translated paragraphs, and a pinned live buffer row visualizing in-flight gestures.
- High-contrast text selection with immediate screen reader aria-live announcements on committed translations.