
import chess
import chess.pgn
import io
from datasets import load_dataset

print("Loading dataset...")

dataset = load_dataset(
    "Lichess/standard-chess-games",
    split="train",
    streaming=True
)

total = 0
successful = 0
failed = 0

print("\nTesting the first 100 games...\n")

for game_data in dataset:
    if total >= 100:
        break

    total += 1
    pgn_text = game_data["movetext"]

    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))

        if game is None or game.errors:
            failed += 1
            continue

        moves = list(game.mainline_moves())

        if not moves:
            failed += 1
            continue

        successful += 1

    except Exception as e:
        failed += 1
        print(f"Game {total}: {e}")

print("\n========== RESULTS ==========")
print("Games inspected:", total)
print("Successfully parsed:", successful)
print("Failed games:", failed)

if total:
    print("Success rate:", round(successful / total * 100, 2), "%")