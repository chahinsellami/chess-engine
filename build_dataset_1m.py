
import os
import io
import json
import hashlib

import chess
import chess.pgn
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from datasets import load_dataset
from tqdm import tqdm


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_NAME = "Lichess/standard-chess-games"

TARGET_POSITIONS = 1_000_000
MAX_GAMES = 30_000

MIN_GAME_PLIES = 20
MAX_GAME_PLIES = 300

CHUNK_SIZE = 25_000

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "positions_1m.parquet"
)
REPORT_FILE = os.path.join(
    OUTPUT_DIR, "dataset_1m_report.json"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# PROTECT EXISTING RESULTS
# ==========================================

if os.path.exists(OUTPUT_FILE):
    raise FileExistsError(
        f"{OUTPUT_FILE} already exists. "
        "Rename or remove this new output file "
        "before rebuilding. Your original dataset "
        "will not be modified."
    )


# ==========================================
# DATASET INITIALIZATION
# ==========================================

print("=" * 55)
print("BUILDING 1 MILLION CHESS POSITIONS")
print("=" * 55)

print("\nLoading Lichess streaming dataset...")

dataset = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

buffer = []
seen_games = set()

games_processed = 0
games_failed = 0
games_duplicate = 0
games_too_short = 0

total_positions = 0
game_id = 0
writer = None


# ==========================================
# SAVE CHUNK
# ==========================================

def save_chunk(rows):

    global writer

    if not rows:
        return

    df = pd.DataFrame(rows)

    table = pa.Table.from_pandas(
        df,
        preserve_index=False
    )

    if writer is None:
        writer = pq.ParquetWriter(
            OUTPUT_FILE,
            table.schema,
            compression="snappy"
        )

    writer.write_table(table)

    print(
        f"Saved {len(rows):,} positions | "
        f"Total: {total_positions:,}"
    )


# ==========================================
# PROCESS GAMES
# ==========================================

print("\nStarting game collection...\n")

try:

    for row in tqdm(dataset, total=MAX_GAMES):

        if games_processed >= MAX_GAMES:
            break

        if total_positions >= TARGET_POSITIONS:
            break

        pgn_text = row.get("movetext")

        if not pgn_text:
            games_failed += 1
            continue

        try:

            # Detect duplicate PGNs
            game_hash = hashlib.sha256(
                pgn_text.encode("utf-8")
            ).hexdigest()

            if game_hash in seen_games:
                games_duplicate += 1
                continue

            seen_games.add(game_hash)

            # Parse PGN
            game = chess.pgn.read_game(
                io.StringIO(pgn_text)
            )

            if game is None or game.errors:
                games_failed += 1
                continue

            board = game.board()

            if board.chess960 or not board.is_valid():
                games_failed += 1
                continue

            # Extract moves
            game_positions = []

            for ply, move in enumerate(
                game.mainline_moves()
            ):

                if ply >= MAX_GAME_PLIES:
                    break

                if move not in board.legal_moves:
                    raise ValueError("Illegal move")

                fen = board.fen()
                uci = move.uci()
                san = board.san(move)

                game_positions.append({
                    "game_id": game_id,
                    "ply": ply,
                    "fen": fen,
                    "move_uci": uci,
                    "move_san": san
                })

                board.push(move)

            if len(game_positions) < MIN_GAME_PLIES:
                games_too_short += 1
                continue

            # Stop exactly at the target
            remaining = TARGET_POSITIONS - total_positions

            game_positions = game_positions[:remaining]

            buffer.extend(game_positions)

            total_positions += len(game_positions)
            games_processed += 1
            game_id += 1

            # Write progressively
            if len(buffer) >= CHUNK_SIZE:

                save_chunk(buffer)
                buffer.clear()

        except Exception as error:

            games_failed += 1
            print(f"\nSkipped game: {error}")

    # Save remaining positions
    if buffer:
        save_chunk(buffer)
        buffer.clear()

finally:

    if writer is not None:
        writer.close()


# ==========================================
# FINAL REPORT
# ==========================================

report = {
    "dataset": DATASET_NAME,
    "target_positions": TARGET_POSITIONS,
    "actual_positions": total_positions,
    "games_processed": games_processed,
    "games_failed": games_failed,
    "duplicate_games": games_duplicate,
    "games_too_short": games_too_short,
    "minimum_game_plies": MIN_GAME_PLIES,
    "maximum_game_plies": MAX_GAME_PLIES,
    "output_file": OUTPUT_FILE
}

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)


print("\n" + "=" * 55)
print("DATASET COLLECTION FINISHED")
print("=" * 55)

print(f"Positions:       {total_positions:,}")
print(f"Games processed: {games_processed:,}")
print(f"Games failed:    {games_failed:,}")
print(f"Duplicates:      {games_duplicate:,}")
print(f"Too short:       {games_too_short:,}")

print(f"\nDataset: {OUTPUT_FILE}")
print(f"Report:  {REPORT_FILE}")

if total_positions >= TARGET_POSITIONS:
    print("\nSUCCESS: Reached 1 million positions!")
else:
    print("\nWARNING: Target not reached.")
    print("Increase MAX_GAMES and run again with a new output filename.")
