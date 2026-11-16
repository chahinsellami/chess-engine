"""Local web interface for the chess engine.

Run with ``python gui.py`` and open http://127.0.0.1:5000.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import chess
import torch
from flask import Flask, jsonify, render_template, request

from engine import ChessEngine


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints" / "best_chess_model.pt"
DATASET = ROOT / "data" / "chess_training_data.npz"

app = Flask(__name__)


class GameSession:
    """Owns the single local game and its background engine search."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.board = chess.Board()
        self.history: list[dict[str, str]] = []
        self.last_move: chess.Move | None = None
        self.thinking = False
        self.search_error: str | None = None
        self.engine_error: str | None = None
        try:
            self.engine = ChessEngine(depth=3)
            self.engine_ready = True
        except Exception as error:
            self.engine = ChessEngine(depth=3, use_neural_network=False)
            self.engine_ready = False
            self.engine_error = str(error)

    def reset(self) -> None:
        with self.lock:
            self.board = chess.Board()
            self.history.clear()
            self.last_move = None
            self.thinking = False
            self.search_error = None

    def push(self, move: chess.Move) -> str:
        san = self.board.san(move)
        self.board.push(move)
        self.history.append({"san": san, "uci": move.uci()})
        self.last_move = move
        return san

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            evaluation = self.engine.evaluate(self.board)
            return {
                "fen": self.board.fen(),
                "turn": "white" if self.board.turn == chess.WHITE else "black",
                "thinking": self.thinking,
                "game_over": self.board.is_game_over(),
                "check": self.board.is_check(),
                "status": self._status(),
                "evaluation": evaluation / 100,
                "history": self.history,
                "last_move": self.last_move.uci() if self.last_move else None,
                "legal_moves": [move.uci() for move in self.board.legal_moves],
                "engine_ready": self.engine_ready,
                "engine_error": self.engine_error,
                "search_error": self.search_error,
            }

    def _status(self) -> str:
        if self.board.is_checkmate():
            return "Checkmate"
        if self.board.is_stalemate():
            return "Draw by stalemate"
        if self.board.is_insufficient_material():
            return "Draw by insufficient material"
        if self.thinking:
            return "Engine thinking"
        if self.board.is_check():
            return "Check"
        return "Your turn" if self.board.turn == chess.WHITE else "Engine turn"


session = GameSession()


def start_engine_search() -> None:
    with session.lock:
        if session.thinking or session.board.is_game_over():
            return
        session.thinking = True
        session.search_error = None
        board_copy = session.board.copy(stack=True)

    def search() -> None:
        try:
            move = session.engine.get_best_move(board_copy, depth=3)
            with session.lock:
                if session.thinking and move in session.board.legal_moves:
                    session.push(move)
        except Exception as error:
            with session.lock:
                session.search_error = str(error)
        finally:
            with session.lock:
                session.thinking = False

    threading.Thread(target=search, name="chess-engine-search", daemon=True).start()


def validation_report() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    check("Python runtime", True, "Python environment is serving the app")
    check("PyTorch", True, f"{torch.__version__} · {session.engine.device or 'CPU'}")
    check("Neural checkpoint", session.engine_ready, str(CHECKPOINT.relative_to(ROOT)))
    check("Training dataset", DATASET.exists(), str(DATASET.relative_to(ROOT)))
    check(
        "Board rules",
        len(list(chess.Board().legal_moves)) == 20,
        "Starting position exposes 20 legal moves",
    )
    check(
        "Engine fallback",
        session.engine is not None,
        "Search object is available",
    )

    metadata: dict[str, Any] = {}
    if CHECKPOINT.exists():
        try:
            checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
            for key in ("epoch", "val_loss", "val_accuracy", "val_top5_accuracy"):
                if key in checkpoint:
                    metadata[key] = checkpoint[key]
        except Exception as error:
            check("Checkpoint metadata", False, str(error))

    return {
        "checks": checks,
        "passed": sum(item["ok"] for item in checks),
        "total": len(checks),
        "training": {
            "metrics_available": bool(metadata),
            "epoch": metadata.get("epoch"),
            "validation_loss": metadata.get("val_loss"),
            "validation_accuracy": metadata.get("val_accuracy"),
            "validation_top5": metadata.get("val_top5_accuracy"),
            "source": str(CHECKPOINT.relative_to(ROOT)) if metadata else "Not recorded in checkpoint",
        },
    }


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/api/state")
def state() -> Any:
    return jsonify(session.snapshot())


@app.post("/api/new")
def new_game() -> Any:
    session.reset()
    return jsonify(session.snapshot())


@app.post("/api/undo")
def undo() -> Any:
    with session.lock:
        if session.thinking:
            return jsonify({"error": "Wait for the engine to finish thinking."}), 409
        count = min(2, len(session.board.move_stack))
        for _ in range(count):
            session.board.pop()
            if session.history:
                session.history.pop()
        session.last_move = session.board.move_stack[-1] if session.board.move_stack else None
        session.search_error = None
    return jsonify(session.snapshot())


@app.post("/api/move")
def move() -> Any:
    payload = request.get_json(silent=True) or {}
    uci = str(payload.get("uci", ""))
    with session.lock:
        if session.thinking:
            return jsonify({"error": "The engine is thinking."}), 409
        if session.board.turn != chess.WHITE:
            return jsonify({"error": "It is the engine's turn."}), 409
        try:
            candidate = chess.Move.from_uci(uci)
        except ValueError:
            return jsonify({"error": "That move is not valid UCI notation."}), 400
        if candidate not in session.board.legal_moves:
            return jsonify({"error": "That move is not legal in this position."}), 400
        session.push(candidate)
        snapshot = session.snapshot()

    if not session.board.is_game_over():
        start_engine_search()
    return jsonify(snapshot)


@app.get("/api/validation")
def validation() -> Any:
    return jsonify(validation_report())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
