# Web Chess GUI Design

## Intent

Replace the current Tkinter interface with a polished local web application for
playing against the chess engine and understanding the machine-learning
pipeline. The UI should feel like a focused chess analysis workspace rather
than a debug window.

## Goals

- Provide a responsive board and clear move interaction in a browser.
- Keep the existing Python engine, `python-chess` rules, and threaded search.
- Show engine thinking state, evaluation, principal variation/top moves when
  available, move history, captured material, and game controls.
- Add a **Validation** area covering both runtime health and training metrics:
  checkpoint/device status, dataset availability, board/move self-tests,
  validation loss/accuracy, epoch progress, and checkpoint metadata.
- Make failures visible in the UI with actionable messages.

## Non-goals

- Rewriting the chess engine or neural network.
- Moving training into the browser.
- Adding authentication, remote hosting, or a cloud database.
- Guaranteeing that an absent training log contains metrics.

## Constraints

- The app runs locally on Windows from the existing Python environment.
- The engine can take time to search, so browser requests must not block the
  server thread.
- The existing `engine.py`, `play.py`, `train.py`, and data/checkpoint paths
  remain the source of truth.

## Approaches

1. **Polished Tkinter refresh:** lowest implementation risk, but limited visual
   flexibility and less convenient validation/data presentation.
2. **CustomTkinter:** improved desktop styling, but still constrained by the
   desktop widget layout and adds a dependency.
3. **Local web app (selected):** Flask JSON endpoints plus a dedicated
   HTML/CSS/JavaScript client. It gives the UI responsive layout, tabs/cards,
   accessible controls, and a clear separation between engine state and
   presentation without changing the engine itself.

The local web app is selected because the requested “much more elegant” visual
upgrade benefits most from modern layout and styling, while Flask keeps the
Python integration small and local.

Approved product decisions: the user plays White, the engine replies as Black
automatically, promotion uses a browser picker, search stays at fixed depth 3,
the app owns one local game, and it runs on `127.0.0.1:5000`.

## Editorial visual correction

The first implementation is too dashboard-like and uses platform-dependent
Unicode chess glyphs. The approved visual direction is an editorial chess
studio: consistent local SVG Staunton pieces, a warm parchment-and-forest board,
one restrained display headline, fewer status labels, and a quiet analysis rail.
Decorative build numbers, telemetry-style microcopy, excessive all-caps labels,
and artificial card density are removed. The board remains the dominant visual
and the Validation area stays available as a secondary inspector.

## Design

### Backend

- Add a small Flask server entry point that owns a game session.
- Expose endpoints for state, legal moves, human moves, undo, new game,
  engine moves, and validation.
- Run engine search in a worker and expose a polling/status response so the
  browser stays responsive.
- Return structured JSON errors rather than silently falling back.

### Frontend

- Use a dark graphite/slate visual system with one green accent and warm board
  colors matching the chess theme.
- Use a two-column desktop layout that collapses to one column on narrow
  screens.
- Main area: board, turn/status badge, move controls, and captured material.
- Side area: move list, engine insight card, and tabs for **Game** and
  **Validation**.
- Validation uses clear pass/warn/fail states and separates:
  - runtime checks (imports, device, model checkpoint, data files);
  - chess checks (legal move generation and board encoding/decoding smoke
    tests);
  - training checks (available metrics and checkpoint metadata).
- Add loading states, disabled controls while the engine searches, and
  keyboard/focus-visible states.

### Validation data

- Read training metrics only when a supported metrics file or checkpoint
  metadata is present.
- If metrics are unavailable, show “not recorded” with the exact source the
  user can create; do not invent values.
- Keep validation checks fast enough to run on demand from the browser.

## Success criteria

- A user can start the app, play legal moves, undo, restart, and see the
  engine respond without freezing the page.
- The interface is visually coherent at desktop and narrow widths.
- The Validation area reports real project state and distinguishes pass,
  warning, and failure.
- Existing engine behavior is preserved and the app can be smoke-tested
  without a trained checkpoint.
