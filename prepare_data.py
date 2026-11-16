
import os
import json
import numpy as np
import chess
import pyarrow.parquet as pq

from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/positions_1m.parquet"
OUTPUT_FILE = "data/chess_training_data_1m.npz"
REPORT_FILE = "data/preparation_1m_report.json"

RANDOM_SEED = 42
BATCH_SIZE = 10_000

INPUT_CHANNELS = 18
NUM_ACTIONS = 5 * 64 * 64

PIECE_CHANNELS = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}

PROMOTION_MAP = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}


# ============================================================
# BOARD ENCODING
# ============================================================

def encode_board(board):

    tensor = np.zeros(
        (INPUT_CHANNELS, 8, 8),
        dtype=np.uint8
    )

    for square, piece in board.piece_map().items():

        channel = PIECE_CHANNELS[
            (piece.color, piece.piece_type)
        ]

        rank = chess.square_rank(square)
        file = chess.square_file(square)

        tensor[channel, rank, file] = 1

    # White to move
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1

    # Castling rights
    rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]

    for i, available in enumerate(rights):
        if available:
            tensor[13 + i, :, :] = 1

    # En passant
    if board.ep_square is not None:

        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)

        tensor[17, rank, file] = 1

    return tensor


# ============================================================
# MOVE ENCODING
# ============================================================

def encode_move(move):

    promotion = PROMOTION_MAP[move.promotion]

    return (
        promotion * 4096
        + move.from_square * 64
        + move.to_square
    )


# ============================================================
# INITIALIZATION
# ============================================================

print("=" * 60)
print("PREPARING 1 MILLION CHESS POSITIONS")
print("=" * 60)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(INPUT_FILE)

if os.path.exists(OUTPUT_FILE):
    raise FileExistsError(
        f"{OUTPUT_FILE} already exists. "
        "Rename it before generating a new dataset."
    )

parquet = pq.ParquetFile(INPUT_FILE)

total_rows = parquet.metadata.num_rows

print("Total Parquet rows:", f"{total_rows:,}")


# ============================================================
# SPLIT COMPLETE GAMES
# ============================================================

print("\nReading game IDs...")

game_table = pq.read_table(
    INPUT_FILE,
    columns=["game_id"]
)

all_game_ids = game_table.column("game_id").to_numpy()

unique_games, positions_per_game = np.unique(
    all_game_ids,
    return_counts=True
)

del game_table
del all_game_ids

rng = np.random.default_rng(RANDOM_SEED)
shuffled_games = unique_games.copy()
rng.shuffle(shuffled_games)

number_of_games = len(shuffled_games)

train_end = int(number_of_games * 0.80)
val_end = int(number_of_games * 0.90)

train_games = shuffled_games[:train_end]
val_games = shuffled_games[train_end:val_end]
test_games = shuffled_games[val_end:]

# Every game belongs to exactly one split.
game_to_split = {}

for game_id in train_games:
    game_to_split[int(game_id)] = 0

for game_id in val_games:
    game_to_split[int(game_id)] = 1

for game_id in test_games:
    game_to_split[int(game_id)] = 2


# Calculate exact allocation sizes.
game_position_counts = {
    int(g): int(c)
    for g, c in zip(unique_games, positions_per_game)
}

split_sizes = [0, 0, 0]

for game_id in train_games:
    split_sizes[0] += game_position_counts[int(game_id)]

for game_id in val_games:
    split_sizes[1] += game_position_counts[int(game_id)]

for game_id in test_games:
    split_sizes[2] += game_position_counts[int(game_id)]

print("\nGame split:")
print("Training games:", len(train_games))
print("Validation games:", len(val_games))
print("Testing games:", len(test_games))

print("\nExpected position allocation:")
print("Training:", f"{split_sizes[0]:,}")
print("Validation:", f"{split_sizes[1]:,}")
print("Testing:", f"{split_sizes[2]:,}")


# ============================================================
# PREALLOCATE ARRAYS
# ============================================================

print("\nAllocating memory...")

X_train = np.empty(
    (split_sizes[0], INPUT_CHANNELS, 8, 8),
    dtype=np.uint8
)
y_train = np.empty(split_sizes[0], dtype=np.int64)

X_val = np.empty(
    (split_sizes[1], INPUT_CHANNELS, 8, 8),
    dtype=np.uint8
)
y_val = np.empty(split_sizes[1], dtype=np.int64)

X_test = np.empty(
    (split_sizes[2], INPUT_CHANNELS, 8, 8),
    dtype=np.uint8
)
y_test = np.empty(split_sizes[2], dtype=np.int64)

X_arrays = [X_train, X_val, X_test]
y_arrays = [y_train, y_val, y_test]

pointers = [0, 0, 0]
invalid_positions = [0, 0, 0]


# ============================================================
# BATCH PROCESSING
# ============================================================

print("\nEncoding positions...")

batch_count = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

batches = parquet.iter_batches(
    batch_size=BATCH_SIZE,
    columns=["game_id", "fen", "move_uci"]
)

for batch in tqdm(batches, total=batch_count):

    ids = batch.column("game_id").to_numpy()
    fens = batch.column("fen").to_pylist()
    moves = batch.column("move_uci").to_pylist()

    for game_id, fen, uci in zip(ids, fens, moves):

        split = game_to_split[int(game_id)]

        try:

            board = chess.Board(fen)

            if not board.is_valid():
                invalid_positions[split] += 1
                continue

            move = chess.Move.from_uci(uci)

            if move not in board.legal_moves:
                invalid_positions[split] += 1
                continue

            X_arrays[split][pointers[split]] = encode_board(board)

            y_arrays[split][pointers[split]] = encode_move(move)

            pointers[split] += 1

        except Exception:
            invalid_positions[split] += 1


# ============================================================
# TRIM ARRAYS
# ============================================================

print("\nEncoding completed.")

X_train = X_train[:pointers[0]]
y_train = y_train[:pointers[0]]

X_val = X_val[:pointers[1]]
y_val = y_val[:pointers[1]]

X_test = X_test[:pointers[2]]
y_test = y_test[:pointers[2]]


# ============================================================
# SAVE DATASET
# ============================================================

print("\nSaving compressed NumPy dataset...")
print("This can take a little while.")

np.savez_compressed(
    OUTPUT_FILE,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test
)


# ============================================================
# VERIFY GAME SPLIT
# ============================================================

train_set = set(map(int, train_games))
val_set = set(map(int, val_games))
test_set = set(map(int, test_games))

assert train_set.isdisjoint(val_set)
assert train_set.isdisjoint(test_set)
assert val_set.isdisjoint(test_set)


# ============================================================
# REPORT
# ============================================================

report = {
    "source": INPUT_FILE,
    "output": OUTPUT_FILE,
    "total_positions": total_rows,
    "total_games": number_of_games,
    "random_seed": RANDOM_SEED,
    "split_method": "game-based",
    "train_games": len(train_games),
    "validation_games": len(val_games),
    "test_games": len(test_games),
    "train_positions": int(len(X_train)),
    "validation_positions": int(len(X_val)),
    "test_positions": int(len(X_test)),
    "invalid_train": invalid_positions[0],
    "invalid_validation": invalid_positions[1],
    "invalid_test": invalid_positions[2],
    "input_shape": list(X_train.shape[1:]),
    "number_of_classes": NUM_ACTIONS
}

with open(REPORT_FILE, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print("Training:", f"{len(X_train):,}")
print("Validation:", f"{len(X_val):,}")
print("Testing:", f"{len(X_test):,}")

print("\nInput shape:", X_train.shape)
print("Output shape:", y_train.shape)

print("\nInvalid positions:", sum(invalid_positions))

print("Possible move classes:", NUM_ACTIONS)

print("\nGame overlap check: PASSED")

print("\nSaved dataset:", OUTPUT_FILE)
print("Saved report:", REPORT_FILE)

print("\nReady for neural network training!")
