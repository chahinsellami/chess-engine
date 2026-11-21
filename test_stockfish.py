
import chess
import chess.engine

from pathlib import Path


# Update this if your executable has a different name.
STOCKFISH_PATH = Path(
    "engines/stockfish/stockfish19.exe"
)


def main():

    if not STOCKFISH_PATH.exists():
        print("ERROR: Stockfish executable not found.")
        print("Expected location:", STOCKFISH_PATH)
        return

    print("Starting Stockfish...")

    engine = chess.engine.SimpleEngine.popen_uci(
        str(STOCKFISH_PATH.resolve())
    )

    try:
        print("Stockfish started successfully!")

        board = chess.Board()

        result = engine.analyse(
            board,
            chess.engine.Limit(time=2.0)
        )

        best_move = result["pv"][0]

        print("\nStockfish analysis:")
        print("Best move:", board.san(best_move))
        print("Evaluation:", result["score"].white())
        print("Depth:", result.get("depth"))

    finally:
        engine.quit()
        print("\nStockfish closed successfully.")


if __name__ == "__main__":
    main()
