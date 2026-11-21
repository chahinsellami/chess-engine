# Stockfish Opponent Integration

## Intent

Let the local GUI select the neural bot or Stockfish as an opponent, and run
matches where two engines play one another.

## Goals

- Preserve the existing Human vs Neural game behavior.
- Add Human vs Stockfish with Stockfish levels 1 through 8.
- Add Engine vs Engine matches with Neural and Stockfish as selectable sides.
- Keep move history, undo, reset, promotion, board rendering, and validation
  in the existing Flask interface.
- Use the bundled Stockfish executable at
  `engines/stockfish/stockfish19.exe`.

## Non-goals

- Human-vs-human mode.
- Persistent game databases or online matchmaking.
- Training or replacing the neural engine.
- Exact Elo claims or official Stockfish strength certification.

## Approaches

1. **Stockfish-only backend:** Replace the neural engine. This is simpler but
   removes the project's existing bot and match capability.
2. **Separate Stockfish page:** Add a second interface. This avoids backend
   changes but duplicates board and session behavior.
3. **Selectable match controller (selected):** Keep one GameSession and route
   each turn to either ChessEngine or Stockfish. It preserves current behavior
   and supports every requested pairing.

## Design

- Add a small `StockfishOpponent` wrapper around
  `chess.engine.SimpleEngine.popen_uci`.
- Map GUI levels 1-8 to Stockfish skill levels and bounded move times.
- Add `mode` and engine selection state to `GameSession`:
  - `human`: the human plays White; `opponent` is Neural or Stockfish for Black.
  - `engine`: both sides use `white_engine` and `black_engine`, selected as
    Neural or Stockfish.
- Add `POST /api/config` to change opponent settings before or between games.
  Changing configuration must not run during an active search.
- Route engine searches through one background worker; Stockfish and Neural
  moves both push through the same board/history logic.
- Extend snapshots with `mode`, opponent names, Stockfish level, and engine
  readiness.
- Add three compact controls to the Game panel: match mode, White engine,
  and Black/opponent engine. Stockfish level appears when Stockfish is selected.
- Add a Stockfish executable check to validation.

## Constraints

- Stockfish runs locally and never leaves the machine.
- The existing neural model and depth-3 search remain unchanged.
- Level settings are practical strength/speed presets, not exact Elo ratings.
- If Stockfish is unavailable, fallback must be explicit; it must not silently
  make Neural moves.
