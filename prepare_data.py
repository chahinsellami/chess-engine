
import os
import chess
import numpy as np
import pandas as pd

from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================

INPUT_FILE = "data/positions_1000.parquet"
OUTPUT_FILE = "data/chess_training_data.npz"

RANDOM_SEED = 42

# 12 piece planes + 1 turn + 4 castling + 1 en passant
INPUT_CHANNELS = 18

# 4096 normal moves + 4 promotion types per square pair
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


# ==========================================
# BOARD ENCODING
# ==========================================


def encode_board(board):

    tensor = np.zeros(
        (18, 8, 8),
        dtype=np.uint8
    )

    # 12 piece channels
    for square, piece in board.piece_map().items():

        channel = PIECE_CHANNELS[
            (piece.color, piece.piece_type)
        ]

        rank = chess.square_rank(square)
        file = chess.square_file(square)

        tensor[channel, rank, file] = 1

    # Channel 12: Side to move
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1

    # Channels 13-16: Castling rights
    castling_rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK)
    ]

    for i, available in enumerate(castling_rights):

        if available:
            tensor[13 + i, :, :] = 1

    # Channel 17: En passant square
    if board.ep_square is not None:

        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)

        tensor[17, rank, file] = 1

    return tensor



# ==========================================
# MOVE ENCODING
# ==========================================

def encode_move(move):

    promotion = PROMOTION_MAP[move.promotion]

    return (
        promotion * 4096
        + move.from_square * 64
        + move.to_square
    )


# ==========================================
# LOAD DATASET
# ==========================================

print("Loading generated dataset...")

df = pd.read_parquet(INPUT_FILE)

print("Rows loaded:", len(df))
print("Games loaded:", df["game_id"].nunique())


# ==========================================
# VALIDATE AND ENCODE
# ==========================================

X = []
y = []
game_ids = []

invalid_positions = 0

print("\nValidating and encoding positions...")

for row in tqdm(df.itertuples(index=False), total=len(df)):

    try:

        board = chess.Board(row.fen)

        if not board.is_valid():
            invalid_positions += 1
            continue

        move = chess.Move.from_uci(row.move_uci)

        if move not in board.legal_moves:
            invalid_positions += 1
            continue

        encoded_board = encode_board(board)
        encoded_move = encode_move(move)

        X.append(encoded_board)
        y.append(encoded_move)
        game_ids.append(row.game_id)

    except Exception:
        invalid_positions += 1


X = np.asarray(X, dtype=np.uint8)
y = np.asarray(y, dtype=np.int64)
game_ids = np.asarray(game_ids)


# ==========================================
# SPLIT BY GAME
# ==========================================

print("\nSplitting dataset...")

unique_games = np.unique(game_ids)

rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(unique_games)

n = len(unique_games)

train_end = int(n * 0.80)
val_end = int(n * 0.90)

train_games = unique_games[:train_end]
val_games = unique_games[train_end:val_end]
test_games = unique_games[val_end:]

train_mask = np.isin(game_ids, train_games)
val_mask = np.isin(game_ids, val_games)
test_mask = np.isin(game_ids, test_games)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]


# ==========================================
# SAVE
# ==========================================

print("\nSaving training data...")

os.makedirs("data", exist_ok=True)

np.savez_compressed(
    OUTPUT_FILE,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test
)


# ==========================================
# RESULTS
# ==========================================

print("\n========== PREPARATION RESULTS ==========")

print("Valid positions:", len(X))
print("Invalid positions:", invalid_positions)

print("\nTraining:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

print("\nInput shape:", X_train.shape)
print("Output shape:", y_train.shape)

print("Number of possible move classes:", NUM_ACTIONS)

print("\nSaved:", OUTPUT_FILE)

print("\nPreparation completed successfully!")
