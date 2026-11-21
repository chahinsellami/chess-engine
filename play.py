
import os
import numpy as np
import chess
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = "checkpoints_1m/best_chess_model_1m.pt"

INPUT_CHANNELS = 18
NUM_CLASSES = 20480
CHANNELS = 128
NUM_RESIDUAL_BLOCKS = 4

TOP_MOVES = 5


# ============================================================
# BOARD ENCODING
# Must match prepare_data.py exactly
# ============================================================

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


def encode_board(board):

    tensor = np.zeros((18, 8, 8), dtype=np.uint8)

    for square, piece in board.piece_map().items():

        channel = PIECE_CHANNELS[
            (piece.color, piece.piece_type)
        ]

        rank = chess.square_rank(square)
        file = chess.square_file(square)

        tensor[channel, rank, file] = 1

    # Side to move
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1

    # Castling rights
    castling_rights = [
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    ]

    for i, available in enumerate(castling_rights):
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
# Must match prepare_data.py exactly
# ============================================================

def encode_move(move):

    promotion = PROMOTION_MAP[move.promotion]

    return (
        promotion * 4096
        + move.from_square * 64
        + move.to_square
    )


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)

        return out


# ============================================================
# CHESS NEURAL NETWORK
# Must match train.py exactly
# ============================================================

class ChessNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                INPUT_CHANNELS,
                CHANNELS,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(CHANNELS),
            nn.ReLU(inplace=True)
        )

        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(CHANNELS)
                for _ in range(NUM_RESIDUAL_BLOCKS)
            ]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                CHANNELS,
                32,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, NUM_CLASSES)
        )

    def forward(self, x):

        x = self.stem(x)
        x = self.residual_blocks(x)
        x = self.policy_head(x)

        return x


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading chess neural network...")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = ChessNet().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully!")
    print("Best training epoch:", checkpoint["epoch"])
    print(
        "Validation accuracy:",
        f"{checkpoint['val_accuracy']:.2f}%"
    )

    return model, device


# ============================================================
# PREDICT LEGAL MOVES
# ============================================================

def predict_moves(model, board, device):

    if board.is_game_over():
        return []

    encoded_board = encode_board(board)

    inputs = torch.from_numpy(
        encoded_board
    ).unsqueeze(0).to(
        device=device,
        dtype=torch.float32
    )

    with torch.inference_mode():

        logits = model(inputs)[0]

    # Generate legal moves and their corresponding output indices
    legal_moves = list(board.legal_moves)

    move_indices = torch.tensor(
        [encode_move(move) for move in legal_moves],
        dtype=torch.long,
        device=device
    )

    # Keep only scores corresponding to legal moves
    legal_logits = logits[move_indices]

    # Normalize scores across legal moves only
    probabilities = torch.softmax(
        legal_logits,
        dim=0
    )

    count = min(TOP_MOVES, len(legal_moves))

    top_values, top_indices = torch.topk(
        probabilities,
        k=count
    )

    predictions = []

    for probability, index in zip(
        top_values.tolist(),
        top_indices.tolist()
    ):

        move = legal_moves[index]

        predictions.append({
            "move": move,
            "probability": probability
        })

    return predictions


# ============================================================
# DISPLAY PREDICTIONS
# ============================================================

def display_predictions(board, predictions):

    print("\nAI move predictions:")

    for i, prediction in enumerate(predictions, start=1):

        move = prediction["move"]
        probability = prediction["probability"]

        san = board.san(move)

        print(
            f"{i}. {san:<10} "
            f"({move.uci():<6}) "
            f"{probability * 100:.2f}%"
        )


# ============================================================
# HUMAN MOVE INPUT
# Accepts SAN or UCI notation
# ============================================================

def get_human_move(board):

    while True:

        user_input = input(
            "\nYour move (e.g. e4, Nf3, e2e4 or 'quit'): "
        ).strip()

        if user_input.lower() in ("quit", "exit", "q"):
            return None

        try:
            return board.parse_san(user_input)

        except ValueError:
            pass

        try:
            return board.parse_uci(user_input)

        except ValueError:
            print("Invalid or illegal move. Try again.")


# ============================================================
# MAIN GAME
# Human = White
# Neural network = Black
# ============================================================

def play_game(model, device):

    board = chess.Board()

    print("\n" + "=" * 50)
    print("YOU ARE WHITE")
    print("THE NEURAL NETWORK IS BLACK")
    print("=" * 50)

    while not board.is_game_over():

        print("\n", board)
        print("\nPosition:", board.fen())

        if board.turn == chess.WHITE:

            move = get_human_move(board)

            if move is None:
                print("\nGame stopped.")
                return

            san = board.san(move)
            board.push(move)

            print("\nYou played:", san)

        else:

            print("\nAI is thinking...")

            predictions = predict_moves(
                model,
                board,
                device
            )

            if not predictions:
                break

            display_predictions(board, predictions)

            best_move = predictions[0]["move"]
            san = board.san(best_move)

            board.push(best_move)

            print("\nAI played:", san)

    print("\n" + "=" * 50)
    print("GAME OVER")
    print("=" * 50)

    print("Result:", board.result())
    print("Reason:", board.outcome().termination.name)

    print("\nFinal board:")
    print(board)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    model, device = load_model()

    while True:

        print("\n1. Play against the neural network")
        print("2. Predict moves for the starting position")
        print("3. Exit")

        choice = input("\nChoose: ").strip()

        if choice == "1":

            play_game(model, device)

        elif choice == "2":

            board = chess.Board()

            predictions = predict_moves(
                model,
                board,
                device
            )

            display_predictions(board, predictions)

        elif choice == "3":

            print("Goodbye!")
            break

        else:

            print("Invalid choice.")
