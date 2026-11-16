
import chess
import math
import time

from play import load_model, predict_moves


# ==========================================
# ENGINE SETTINGS
# ==========================================

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

MATE_SCORE = 100000
INFINITY = float("inf")


class ChessEngine:

    def __init__(self, depth=3, use_neural_network=True):

        print("Loading chess engine...")

        self.model = None
        self.device = None
        self.use_neural_network = use_neural_network

        if use_neural_network:
            self.model, self.device = load_model()

        self.depth = depth
        self.nodes = 0
        self.start_time = 0
        self.time_limit = None
        self.transposition_table = {}
        self.killers = {}
        self.history = {}

        print("Chess engine ready!")

    # ======================================
    # NEURAL MOVE ORDERING
    # ======================================

    def neural_ordering(self, board):

        scores = {}

        if not self.use_neural_network:
            return scores

        try:
            predictions = predict_moves(
                self.model,
                board,
                self.device
            )

            for item in predictions:
                move = item["move"]
                probability = float(item["probability"])
                scores[move.uci()] = probability

        except Exception as error:
            print("Neural ordering error:", error)

        return scores

    # ======================================
    # POSITION EVALUATION
    # Positive = White advantage
    # ======================================

    def evaluate(self, board):

        if board.is_checkmate():
            return -MATE_SCORE if board.turn else MATE_SCORE

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0

        for square, piece in board.piece_map().items():

            value = PIECE_VALUES[piece.piece_type]

            file = chess.square_file(square)
            rank = chess.square_rank(square)

            center_distance = (
                abs(3.5 - file) + abs(3.5 - rank)
            )

            central_bonus = int(14 - center_distance * 3)

            if piece.piece_type == chess.PAWN:

                advancement = (
                    rank if piece.color == chess.WHITE
                    else 7 - rank
                )

                value += advancement * 8
                value += central_bonus // 2

            elif piece.piece_type == chess.KNIGHT:
                value += central_bonus * 3

            elif piece.piece_type == chess.BISHOP:
                value += central_bonus * 2

            elif piece.piece_type == chess.QUEEN:
                value += central_bonus

            # Reward bishops for staying active.
            if piece.piece_type == chess.BISHOP:
                value += 5

            score += value if piece.color == chess.WHITE else -value

        # Mobility
        mobility = board.legal_moves.count()
        score += mobility * 2 if board.turn == chess.WHITE else -mobility * 2

        # Check bonus
        if board.is_check():
            score += -35 if board.turn == chess.WHITE else 35

        # Rook open-file bonus
        for square, piece in board.piece_map().items():

            if piece.piece_type != chess.ROOK:
                continue

            file = chess.square_file(square)

            if not any(
                p.piece_type == chess.PAWN
                and chess.square_file(s) == file
                and p.color == piece.color
                for s, p in board.piece_map().items()
            ):
                bonus = 20
                score += bonus if piece.color == chess.WHITE else -bonus

        return score

    # ======================================
    # MOVE ORDERING
    # ======================================

    def order_moves(self, board, moves, tt_move=None, ply=0):

        neural_scores = self.neural_ordering(board)

        def move_score(move):

            score = 0

            if move == tt_move:
                return 10_000_000

            if board.is_capture(move):

                victim = board.piece_at(move.to_square)

                if victim is None and board.is_en_passant(move):
                    victim_value = PIECE_VALUES[chess.PAWN]
                else:
                    victim_value = PIECE_VALUES[victim.piece_type]

                attacker = board.piece_at(move.from_square)

                score += 1_000_000 + victim_value * 10
                score -= PIECE_VALUES[attacker.piece_type]

            if move.promotion:
                score += PIECE_VALUES[move.promotion]

            if move.uci() in neural_scores:
                score += int(neural_scores[move.uci()] * 10000)

            if move in self.killers.get(ply, []):
                score += 50000

            score += self.history.get(move.uci(), 0)

            return score

        return sorted(
            moves,
            key=move_score,
            reverse=True
        )

    # ======================================
    # QUIESCENCE SEARCH
    # ======================================

    def quiescence(self, board, alpha, beta, ply=0):

        self.nodes += 1

        if board.is_checkmate():
            return -MATE_SCORE + ply

        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_draw()
        ):
            return 0

        if board.is_check():
            moves = list(board.legal_moves)

            if not moves:
                return -MATE_SCORE + ply

            for move in self.order_moves(board, moves, ply=ply):

                board.push(move)
                score = -self.quiescence(
                    board, -beta, -alpha, ply + 1
                )
                board.pop()

                if score >= beta:
                    return beta

                alpha = max(alpha, score)

            return alpha

        stand_pat = self.evaluate(board)

        if board.turn == chess.BLACK:
            stand_pat = -stand_pat

        if stand_pat >= beta:
            return beta

        alpha = max(alpha, stand_pat)

        captures = [
            move for move in board.legal_moves
            if board.is_capture(move) or move.promotion
        ]

        for move in self.order_moves(board, captures, ply=ply):

            board.push(move)

            score = -self.quiescence(
                board, -beta, -alpha, ply + 1
            )

            board.pop()

            if score >= beta:
                return beta

            alpha = max(alpha, score)

        return alpha

    # ======================================
    # ALPHA-BETA MINIMAX
    # ======================================

    def negamax(
        self,
        board,
        depth,
        alpha,
        beta,
        ply=0
    ):

        self.nodes += 1

        if board.is_checkmate():
            return -MATE_SCORE + ply, None

        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_draw()
        ):
            return 0, None

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply), None

        original_alpha = alpha

        key = board.fen()

        tt_entry = self.transposition_table.get(key)
        tt_move = None

        if tt_entry:
            tt_depth, tt_score, tt_flag, tt_move = tt_entry

            if tt_depth >= depth:

                if tt_flag == "EXACT":
                    return tt_score, tt_move

                if tt_flag == "LOWER":
                    alpha = max(alpha, tt_score)

                elif tt_flag == "UPPER":
                    beta = min(beta, tt_score)

                if alpha >= beta:
                    return tt_score, tt_move

        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return -MATE_SCORE + ply, None

        best_move = None
        best_score = -INFINITY

        ordered_moves = self.order_moves(
            board,
            legal_moves,
            tt_move,
            ply
        )

        for move in ordered_moves:

            board.push(move)

            score, _ = self.negamax(
                board,
                depth - 1,
                -beta,
                -alpha,
                ply + 1
            )

            score = -score

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)

            if alpha >= beta:

                if not board.is_capture(move):

                    killers = self.killers.setdefault(ply, [])

                    if move not in killers:
                        killers.insert(0, move)
                        del killers[2:]

                    self.history[move.uci()] = (
                        self.history.get(move.uci(), 0)
                        + depth * depth
                    )

                break

        if best_score <= original_alpha:
            flag = "UPPER"
        elif best_score >= beta:
            flag = "LOWER"
        else:
            flag = "EXACT"

        self.transposition_table[key] = (
            depth,
            best_score,
            flag,
            best_move
        )

        return best_score, best_move

    # ======================================
    # FIND BEST MOVE
    # ======================================

    def get_best_move(self, board, depth=None, time_limit=None):

        if board.is_game_over():
            return None

        search_depth = depth or self.depth

        self.nodes = 0
        self.start_time = time.time()
        self.time_limit = time_limit

        self.killers.clear()
        self.history.clear()

        best_move = None
        best_score = 0

        # Iterative deepening
        for current_depth in range(1, search_depth + 1):

            try:

                score, move = self.negamax(
                    board,
                    current_depth,
                    -INFINITY,
                    INFINITY
                )

                if move is not None:
                    best_move = move
                    best_score = score

                elapsed = time.time() - self.start_time

                print(
                    f"Depth {current_depth} | "
                    f"Score {best_score / 100:.2f} | "
                    f"Nodes {self.nodes} | "
                    f"Time {elapsed:.2f}s"
                )

                if abs(best_score) > MATE_SCORE - 100:
                    break

                if (
                    time_limit is not None
                    and elapsed >= time_limit
                ):
                    break

            except Exception as error:
                print("Search error:", error)
                break

        return best_move
