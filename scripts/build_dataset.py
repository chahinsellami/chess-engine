
import io
import os
import chess
import chess.pgn
import pandas as pd

from datasets import load_dataset
from tqdm import tqdm

# =====================================
# CONFIGURATION
# =====================================

DATASET_NAME = "Lichess/standard-chess-games"
MAX_GAMES = 1000

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "positions_1000.parquet")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# LOAD DATASET
# =====================================

print("Loading Lichess dataset...")

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

positions = []

games_processed = 0
games_failed = 0
total_positions = 0

print(f"Processing {MAX_GAMES} games...\n")

# =====================================
# PROCESS GAMES
# =====================================

for row in tqdm(dataset, total=MAX_GAMES):

    pgn_text = row.get("movetext")

    if not pgn_text:
        games_failed += 1
        continue

    try:
        game = chess.pgn.read_game(
            io.StringIO(pgn_text)
        )

        if game is None:
            games_failed += 1
            continue

        board = game.board()
        game_positions = []

        for ply, move in enumerate(game.mainline_moves()):

            # Position before the move
            fen = board.fen()

            # Move notation
            uci = move.uci()
            san = board.san(move)

            game_positions.append({
                "game_id": games_processed,
                "ply": ply,
                "fen": fen,
                "move_uci": uci,
                "move_san": san
            })

            board.push(move)

        if not game_positions:
            games_failed += 1
            continue

        positions.extend(game_positions)

        games_processed += 1
        total_positions += len(game_positions)

    except Exception as e:
        games_failed += 1
        print(f"\nError: {e}")

    if games_processed >= MAX_GAMES:
        break

# =====================================
# SAVE DATASET
# =====================================

print("\nProcessing finished!")
print("------------------------------")
print("Games processed:", games_processed)
print("Games failed:", games_failed)
print("Total positions:", total_positions)

if positions:

    df = pd.DataFrame(positions)

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nDataset saved successfully!")
    print("Location:", OUTPUT_FILE)

    print("\nFirst 5 examples:")
    print(df.head().to_string(index=False))

    print("\nDataset information:")
    print(df.info())

else:
    print("\nNo positions generated!")
