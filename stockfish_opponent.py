from __future__ import annotations

from pathlib import Path
from typing import Any

import chess
import chess.engine


ROOT = Path(__file__).resolve().parent
STOCKFISH_PATH = ROOT / "engines" / "stockfish" / "stockfish19.exe"

LEVELS = {
    1: {"skill": 0, "time": 0.08},
    2: {"skill": 3, "time": 0.15},
    3: {"skill": 6, "time": 0.25},
    4: {"skill": 9, "time": 0.40},
    5: {"skill": 12, "time": 0.65},
    6: {"skill": 15, "time": 1.00},
    7: {"skill": 17, "time": 1.50},
    8: {"skill": 20, "time": 2.50},
}


class StockfishOpponent:
    def __init__(self, level: int = 5) -> None:
        if not STOCKFISH_PATH.exists():
            raise FileNotFoundError(
                f"Stockfish executable not found: {STOCKFISH_PATH}"
            )

        self.level = self._valid_level(level)
        self.engine = chess.engine.SimpleEngine.popen_uci(
            str(STOCKFISH_PATH)
        )
        self.configure_level(self.level)

    @staticmethod
    def _valid_level(level: int) -> int:
        level = int(level)
        if level not in LEVELS:
            raise ValueError("Stockfish level must be between 1 and 8.")
        return level

    def configure_level(self, level: int) -> None:
        self.level = self._valid_level(level)
        preset = LEVELS[self.level]
        self.engine.configure({
            "Skill Level": preset["skill"],
            "Threads": 1,
        })

    def get_best_move(self, board: chess.Board) -> chess.Move:
        preset = LEVELS[self.level]
        result = self.engine.play(
            board,
            chess.engine.Limit(time=preset["time"]),
        )
        if result.move is None:
            raise RuntimeError("Stockfish did not return a move.")
        return result.move

    def analyse(self, board: chess.Board) -> dict[str, Any]:
        preset = LEVELS[self.level]
        result = self.engine.analyse(
            board,
            chess.engine.Limit(time=min(preset["time"], 0.25)),
        )
        score = result["score"].pov(chess.WHITE)
        return {
            "depth": result.get("depth"),
            "score_cp": score.score(mate_score=10000),
        }

    def quit(self) -> None:
        try:
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass
