"""Local web interface for the chess engine.

Run with ``python gui.py`` and open http://127.0.0.1:5000.
"""

from __future__ import annotations

import math
import threading
import atexit
from pathlib import Path
from typing import Any

import chess
import torch
from flask import Flask, jsonify, render_template, request

from engine import ChessEngine
from stockfish_opponent import STOCKFISH_PATH, StockfishOpponent


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "checkpoints_1m" / "best_chess_model_1m.pt"
DATASET = ROOT / "data" / "chess_training_data_1m.npz"

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
        self.mode = "human"
        self.opponent = "neural"
        self.white_engine = "stockfish"
        self.black_engine = "neural"
        self.stockfish_level = 5
        self.stockfish: StockfishOpponent | None = None
        self.stockfish_error: str | None = None
        self.evaluation = 0.0
        self.autoplay = False
        try:
            self.engine = ChessEngine(depth=3)
            self.engine_ready = True
        except Exception as error:
            self.engine = ChessEngine(depth=3, use_neural_network=False)
            self.engine_ready = False
            self.engine_error = str(error)
        try:
            self.stockfish = StockfishOpponent(self.stockfish_level)
            self.stockfish_ready = True
        except Exception as error:
            self.stockfish = None
            self.stockfish_ready = False
            self.stockfish_error = str(error)

    def reset(self) -> None:
        with self.lock:
            self.board = chess.Board()
            self.history.clear()
            self.last_move = None
            self.thinking = False
            self.search_error = None
            self.evaluation = 0.0

    def push(self, move: chess.Move) -> str:
        san = self.board.san(move)
        self.board.push(move)
        self.history.append({"san": san, "uci": move.uci()})
        self.last_move = move
        return san

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            if self.mode == "human":
                white_name = "Human"
                black_name = "Stockfish" if self.opponent == "stockfish" else "Neural Bot"
            else:
                white_name = "Stockfish" if self.white_engine == "stockfish" else "Neural Bot"
                black_name = "Stockfish" if self.black_engine == "stockfish" else "Neural Bot"
            return {
                "fen": self.board.fen(),
                "turn": "white" if self.board.turn == chess.WHITE else "black",
                "thinking": self.thinking,
                "game_over": self.board.is_game_over(),
                "check": self.board.is_check(),
                "status": self._status(),
                "evaluation": self.evaluation,
                "history": self.history,
                "last_move": self.last_move.uci() if self.last_move else None,
                "legal_moves": [move.uci() for move in self.board.legal_moves],
                "engine_ready": self.engine_ready,
                "engine_error": self.engine_error,
                "stockfish_ready": self.stockfish_ready,
                "stockfish_error": self.stockfish_error,
                "search_error": self.search_error,
                "mode": self.mode,
                "opponent": self.opponent,
                "white_engine": self.white_engine,
                "black_engine": self.black_engine,
                "stockfish_level": self.stockfish_level,
                "autoplay": self.autoplay,
                "white_name": white_name,
                "black_name": black_name,
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


@atexit.register
def close_stockfish() -> None:
    if session.stockfish is not None:
        session.stockfish.quit()


def start_engine_search() -> None:
    with session.lock:
        if session.thinking or session.board.is_game_over():
            return
        if session.mode == "human":
            if session.board.turn != chess.BLACK:
                return
            engine_name = session.opponent
        else:
            engine_name = (
                session.white_engine
                if session.board.turn == chess.WHITE
                else session.black_engine
            )
        session.thinking = True
        session.search_error = None
        board_copy = session.board.copy(stack=True)

    def search() -> None:
        continue_match = False
        try:
            if engine_name == "stockfish":
                if not session.stockfish_ready or session.stockfish is None:
                    raise RuntimeError(
                        session.stockfish_error or "Stockfish is unavailable."
                    )
                move = session.stockfish.get_best_move(board_copy)
                with session.lock:
                    try:
                        analysis = session.stockfish.analyse(board_copy)
                        session.evaluation = (analysis["score_cp"] or 0) / 100
                    except Exception:
                        pass
            else:
                move = session.engine.get_best_move(board_copy, depth=3)
                with session.lock:
                    session.evaluation = session.engine.evaluate(board_copy) / 100
            with session.lock:
                if session.thinking and move in session.board.legal_moves:
                    session.push(move)
                    continue_match = True
        except Exception as error:
            with session.lock:
                session.search_error = str(error)
        finally:
            with session.lock:
                session.thinking = False
        if session.mode == "engine" and continue_match and session.autoplay:
            start_engine_search()

    threading.Thread(target=search, name="chess-engine-search", daemon=True).start()


# --------------------------------------------------------------------------
# Move predictions (drawn as arrows on the board by the front end)
# --------------------------------------------------------------------------

# If your ChessEngine exposes the policy head under another name, add it here.
# The method may take (board, top_n) or just (board) and may return a dict or a
# list of (move, probability) pairs, where move is a chess.Move or a UCI string.
POLICY_METHODS = (
    "neural_ordering",  # ChessEngine.neural_ordering(board) -> {uci: probability}
    "predict_moves",
    "get_move_probabilities",
    "move_probabilities",
    "get_policy",
    "policy",
    "top_moves",
)

# Set to False if ChessEngine.evaluate() scores from the side to move's view.
EVAL_IS_WHITE_POV = True
SOFTMAX_TEMPERATURE_CP = 150.0

_prediction_cache: dict[str, tuple[str, list[tuple[chess.Move, float]]]] = {}


def _normalise_policy(raw: Any, board: chess.Board) -> list[tuple[chess.Move, float]]:
    items = raw.items() if isinstance(raw, dict) else raw
    parsed: list[tuple[chess.Move, float]] = []
    for item in items:
        move, prob = item[0], float(item[1])
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        if move in board.legal_moves:
            parsed.append((move, prob))
    total = sum(prob for _, prob in parsed)
    if not parsed or total <= 0:
        raise ValueError("Policy output did not contain legal moves.")
    if total > 1.0001:
        parsed = [(move, prob / total) for move, prob in parsed]
    return sorted(parsed, key=lambda pair: -pair[1])


def _policy_predictions(board: chess.Board, top_n: int):
    for name in POLICY_METHODS:
        method = getattr(session.engine, name, None)
        if not callable(method):
            continue
        for args in ((board, top_n), (board,)):
            try:
                return _normalise_policy(method(*args), board)
            except Exception:
                continue
    return None


def _evaluation_predictions(board: chess.Board) -> list[tuple[chess.Move, float]]:
    """Fallback: rank every legal move by the engine's evaluation one ply deep."""
    scored: list[tuple[chess.Move, float]] = []
    mover_is_white = board.turn == chess.WHITE
    for move in list(board.legal_moves):
        board.push(move)
        try:
            if board.is_checkmate():
                score = 10_000.0
            else:
                value = float(session.engine.evaluate(board))
                if not EVAL_IS_WHITE_POV:
                    value = -value  # value was for the side to move, i.e. the opponent
                score = value if mover_is_white else -value
        finally:
            board.pop()
        scored.append((move, score))
    best = max(score for _, score in scored)
    weights = [math.exp((score - best) / SOFTMAX_TEMPERATURE_CP) for _, score in scored]
    total = sum(weights)
    ranked = [(move, weight / total) for (move, _), weight in zip(scored, weights)]
    return sorted(ranked, key=lambda pair: -pair[1])


@app.get("/api/predictions")
def predictions() -> Any:
    try:
        top_n = max(1, min(8, int(request.args.get("n", 5))))
    except ValueError:
        top_n = 5
    with session.lock:
        fen = session.board.fen()
        if session.board.is_game_over():
            return jsonify({"fen": fen, "source": None, "moves": []})
        board = session.board.copy(stack=True)

    cached = _prediction_cache.get(fen)
    if cached is None:
        ranked = _policy_predictions(board, 8)
        source = "policy"
        if ranked is None:
            ranked = _evaluation_predictions(board)
            source = "evaluation"
        if len(_prediction_cache) > 256:
            _prediction_cache.clear()
        cached = _prediction_cache[fen] = (source, ranked)

    source, ranked = cached
    moves = [
        {"uci": move.uci(), "san": board.san(move), "prob": round(prob, 4)}
        for move, prob in ranked[:top_n]
    ]
    return jsonify({"fen": fen, "source": source, "moves": moves})


def validation_report() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    check("Python runtime", True, "Python environment is serving the app")
    check("PyTorch", True, f"{torch.__version__} · {session.engine.device or 'CPU'}")
    check("Neural checkpoint", session.engine_ready, str(CHECKPOINT.relative_to(ROOT)))
    check("Training dataset", DATASET.exists(), str(DATASET.relative_to(ROOT)))
    check(
        "Stockfish executable",
        session.stockfish_ready,
        str(STOCKFISH_PATH.relative_to(ROOT)),
    )
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


@app.post("/api/config")
def configure() -> Any:
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode", "human")).lower()
    opponent = str(payload.get("opponent", "neural")).lower()
    white_engine = str(payload.get("white_engine", "stockfish")).lower()
    black_engine = str(payload.get("black_engine", "neural")).lower()

    if mode not in ("human", "engine"):
        return jsonify({"error": "Mode must be human or engine."}), 400
    if opponent not in ("neural", "stockfish"):
        return jsonify({"error": "Opponent must be neural or stockfish."}), 400
    if white_engine not in ("neural", "stockfish"):
        return jsonify({"error": "White engine must be neural or stockfish."}), 400
    if black_engine not in ("neural", "stockfish"):
        return jsonify({"error": "Black engine must be neural or stockfish."}), 400

    try:
        level = int(payload.get("stockfish_level", 5))
    except (TypeError, ValueError):
        return jsonify({"error": "Stockfish level must be an integer."}), 400
    if level < 1 or level > 8:
        return jsonify({"error": "Stockfish level must be between 1 and 8."}), 400

    with session.lock:
        if session.thinking:
            return jsonify({"error": "Wait for the engine to finish thinking."}), 409
        session.mode = mode
        session.opponent = opponent
        session.white_engine = white_engine
        session.black_engine = black_engine
        session.stockfish_level = level
        if session.stockfish is not None:
            try:
                session.stockfish.configure_level(level)
            except Exception as error:
                session.stockfish_error = str(error)
                session.stockfish_ready = False
        session.reset()

    # Engines never move on their own: use /api/step, or turn autoplay on.
    if mode == "engine" and session.autoplay:
        start_engine_search()
    return jsonify(session.snapshot())


@app.post("/api/step")
def step() -> Any:
    """Let the engine whose turn it is play exactly one move (human mode: Black only)."""
    with session.lock:
        if session.mode == "human" and session.board.turn != chess.BLACK:
            return jsonify({"error": "It is your turn to move."}), 409
        if session.thinking:
            return jsonify({"error": "The engine is already thinking."}), 409
        if session.board.is_game_over():
            return jsonify({"error": "The game is over."}), 409
    start_engine_search()
    return jsonify(session.snapshot())


@app.post("/api/autoplay")
def autoplay() -> Any:
    payload = request.get_json(silent=True) or {}
    on = bool(payload.get("on", False))
    with session.lock:
        session.autoplay = on
        engine_to_move = session.mode == "engine" or session.board.turn == chess.BLACK
        should_start = on and engine_to_move and not session.thinking
    if should_start:
        start_engine_search()
    return jsonify(session.snapshot())


@app.post("/api/undo")
def undo() -> Any:
    with session.lock:
        if session.thinking:
            return jsonify({"error": "Wait for the engine to finish thinking."}), 409
        if session.mode == "engine" or session.board.turn == chess.BLACK:
            count = min(1, len(session.board.move_stack))
        else:
            count = min(2, len(session.board.move_stack))
        for _ in range(count):
            session.board.pop()
            if session.history:
                session.history.pop()
        session.last_move = session.board.move_stack[-1] if session.board.move_stack else None
        session.search_error = None
        session.evaluation = 0.0
    return jsonify(session.snapshot())


@app.post("/api/move")
def move() -> Any:
    payload = request.get_json(silent=True) or {}
    uci = str(payload.get("uci", ""))
    with session.lock:
        if session.thinking:
            return jsonify({"error": "The engine is thinking."}), 409
        if session.mode != "human":
            return jsonify({"error": "Switch to Human match mode to make moves."}), 409
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

    # The bot only replies on its own when auto-play is switched on.
    if session.autoplay and not session.board.is_game_over():
        start_engine_search()
    return jsonify(snapshot)


@app.get("/api/validation")
def validation() -> Any:
    return jsonify(validation_report())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)