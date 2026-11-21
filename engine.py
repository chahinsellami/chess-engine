import math
import time
from collections import OrderedDict

import chess

from play import load_model, predict_moves


# Scores are in centipawns. Positive evaluation means White is better.
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

MATE_SCORE = 100_000
MATE_THRESHOLD = MATE_SCORE - 1_000
INFINITY = float("inf")
MAX_TT_ENTRIES = 500_000
MAX_QUIESCENCE_PLY = 16

# At the root, only search the network's top-K moves (plus any mate in one).
# Set to 0 to search every legal move again.
NEURAL_ROOT_TOP_K = 5


class SearchTimeout(Exception):
    """Raised internally when the search reaches its time limit."""


class ChessEngine:
    """
    Chess engine combining:
      - a trained policy network for root move ordering,
      - iterative-deepening Negamax,
      - Alpha-Beta pruning,
      - quiescence search,
      - transposition table,
      - killer/history move ordering.

    Public interface kept compatible with the project's existing GUI:
        ChessEngine(depth=3, use_neural_network=True)
        get_best_move(board, depth=None, time_limit=None)
    """

    def __init__(self, depth=3, use_neural_network=True, neural_top_k=NEURAL_ROOT_TOP_K):
        print("Loading chess engine...")

        self.depth = max(1, int(depth))
        self.use_neural_network = bool(use_neural_network)
        self.neural_top_k = max(0, int(neural_top_k))
        self.model = None
        self.device = None

        if self.use_neural_network:
            self.model, self.device = load_model()

        self.nodes = 0
        self.qnodes = 0
        self.tt_hits = 0
        self.tt_cutoffs = 0
        self.beta_cutoffs = 0

        self.start_time = 0.0
        self.deadline = None
        self._last_time_check = 0

        self.transposition_table = OrderedDict()
        self.killers = {}
        self.history = {}

        # Neural inference is deliberately restricted to the root position.
        # Running a CNN at every search node makes the engine much slower.
        self.root_neural_scores = {}

        print("Chess engine ready!")

    # ------------------------------------------------------------------
    # TIME CONTROL
    # ------------------------------------------------------------------

    def _check_time(self, force=False):
        # Checking the clock at every node is unnecessarily expensive.
        if not force and (self.nodes + self.qnodes) - self._last_time_check < 512:
            return

        self._last_time_check = self.nodes + self.qnodes

        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise SearchTimeout

    # ------------------------------------------------------------------
    # NEURAL NETWORK: ROOT MOVE ORDERING
    # ------------------------------------------------------------------

    def neural_ordering(self, board):
        """Return {uci: probability} for the network's top legal moves."""
        if not self.use_neural_network or self.model is None:
            return {}

        try:
            predictions = predict_moves(self.model, board, self.device)
            return {
                item["move"].uci(): float(item["probability"])
                for item in predictions
            }
        except Exception as error:
            print(f"Neural move-ordering warning: {error}")
            return {}

    def _root_candidates(self, board, legal_moves):
        """
        Restrict the root search to the network's top-K moves, so the engine
        plays one of the moves it predicted. Any move that gives checkmate is
        always kept, so a mate in one is never missed. Returns None when the
        restriction does not apply (network off, no predictions, K = 0).
        """
        if self.neural_top_k <= 0 or not self.root_neural_scores:
            return None

        ranked = sorted(
            self.root_neural_scores.items(), key=lambda item: item[1], reverse=True
        )
        top = {uci for uci, _ in ranked[: self.neural_top_k]}

        allowed = []
        for move in legal_moves:
            if move.uci() in top:
                allowed.append(move)
                continue
            board.push(move)
            try:
                gives_mate = board.is_checkmate()
            finally:
                board.pop()
            if gives_mate:
                allowed.append(move)

        return allowed or None

    # ------------------------------------------------------------------
    # POSITION EVALUATION
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate(board):
        """
        Static evaluation from White's point of view.
        Mobility is intentionally omitted here: counting only the current
        side's legal moves and signing it creates a perspective inconsistency.
        """
        if board.is_checkmate():
            return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE

        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_fifty_moves()
            or board.is_seventyfive_moves()
        ):
            return 0

        score = 0
        piece_map = board.piece_map()

        for square, piece in piece_map.items():
            piece_type = piece.piece_type
            value = PIECE_VALUES[piece_type]

            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            relative_rank = rank_idx if piece.color == chess.WHITE else 7 - rank_idx

            center_distance = abs(3.5 - file_idx) + abs(3.5 - rank_idx)
            central_bonus = int(14 - center_distance * 3)

            if piece_type == chess.PAWN:
                value += relative_rank * 8
                value += central_bonus // 2

                # Passed pawn: no enemy pawn is ahead on the same
                # file or either adjacent file.
                enemy_pawns = board.pieces(chess.PAWN, not piece.color)
                is_passed = True

                for enemy_square in enemy_pawns:
                    enemy_file = chess.square_file(enemy_square)
                    enemy_rank = chess.square_rank(enemy_square)

                    same_or_adjacent_file = abs(enemy_file - file_idx) <= 1
                    enemy_is_ahead = (
                        enemy_rank > rank_idx
                        if piece.color == chess.WHITE
                        else enemy_rank < rank_idx
                    )

                    if same_or_adjacent_file and enemy_is_ahead:
                        is_passed = False
                        break

                if is_passed:
                    value += 12 + relative_rank * 8

            elif piece_type == chess.KNIGHT:
                value += central_bonus * 3

            elif piece_type == chess.BISHOP:
                value += central_bonus * 2
                value += 5

            elif piece_type == chess.QUEEN:
                value += central_bonus

            elif piece_type == chess.ROOK:
                # Reward rooks on open or semi-open files.
                own_pawns = board.pieces(chess.PAWN, piece.color)
                enemy_pawns = board.pieces(chess.PAWN, not piece.color)
                file_mask = chess.BB_FILES[file_idx]
                if not (own_pawns & file_mask):
                    value += 20 if not (enemy_pawns & file_mask) else 10

            score += value if piece.color == chess.WHITE else -value

        # A small penalty for being in check, from the checked side's view.
        if board.is_check():
            score += -30 if board.turn == chess.WHITE else 30

        return score

    # ------------------------------------------------------------------
    # MOVE ORDERING
    # ------------------------------------------------------------------

    def order_moves(self, board, moves, tt_move=None, ply=0, root=False):
        killers = self.killers.get(ply, [])
        neural_scores = self.root_neural_scores if root else {}

        def move_score(move):
            if move == tt_move:
                return 10_000_000

            score = 0
            attacker = board.piece_at(move.from_square)
            victim = board.piece_at(move.to_square)

            if board.is_capture(move):
                if board.is_en_passant(move):
                    victim_value = PIECE_VALUES[chess.PAWN]
                else:
                    victim_value = PIECE_VALUES[victim.piece_type] if victim else 0

                attacker_value = PIECE_VALUES[attacker.piece_type] if attacker else 0
                # MVV-LVA-style ordering: promising captures first.
                score += 1_000_000 + victim_value * 10 - attacker_value

            if move.promotion:
                score += 800_000 + PIECE_VALUES[move.promotion]

            if move.uci() in neural_scores:
                score += int(neural_scores[move.uci()] * 100_000)

            if move in killers:
                score += 50_000

            score += self.history.get((board.turn, move.from_square, move.to_square), 0)
            return score

        return sorted(moves, key=move_score, reverse=True)

    # ------------------------------------------------------------------
    # TRANSPOSITION TABLE
    # ------------------------------------------------------------------

    @staticmethod
    def _tt_key(board):
        # Include the halfmove clock. En-passant is included only when legal.
        return board.fen(en_passant="legal")

    @staticmethod
    def _score_to_tt(score, ply):
        if score > MATE_THRESHOLD:
            return score + ply
        if score < -MATE_THRESHOLD:
            return score - ply
        return score

    @staticmethod
    def _score_from_tt(score, ply):
        if score > MATE_THRESHOLD:
            return score - ply
        if score < -MATE_THRESHOLD:
            return score + ply
        return score

    def _store_tt(self, key, depth, score, flag, best_move, ply):
        # Repetition depends on the path to the position. Avoid storing a
        # result from a repetition-sensitive node as a general position result.
        entry = (depth, self._score_to_tt(score, ply), flag, best_move)

        if key in self.transposition_table:
            self.transposition_table.pop(key)
        self.transposition_table[key] = entry

        if len(self.transposition_table) > MAX_TT_ENTRIES:
            self.transposition_table.popitem(last=False)

    # ------------------------------------------------------------------
    # QUIESCENCE SEARCH
    # ------------------------------------------------------------------

    def quiescence(self, board, alpha, beta, ply=0, qply=0):
        self.qnodes += 1
        self._check_time()

        if board.is_checkmate():
            return -MATE_SCORE + ply

        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.can_claim_draw()
        ):
            return 0

        in_check = board.is_check()

        if not in_check:
            stand_pat = self.evaluate(board)
            if board.turn == chess.BLACK:
                stand_pat = -stand_pat

            if stand_pat >= beta:
                return beta

            alpha = max(alpha, stand_pat)

            if qply >= MAX_QUIESCENCE_PLY:
                return alpha

            moves = [
                move for move in board.legal_moves
                if board.is_capture(move) or move.promotion
            ]
        else:
            # When in check, all legal evasions must be considered.
            moves = list(board.legal_moves)
            if not moves:
                return -MATE_SCORE + ply

        moves = self.order_moves(board, moves, ply=ply)

        for move in moves:
            board.push(move)
            try:
                score = -self.quiescence(
                    board, -beta, -alpha, ply + 1, qply + 1
                )
            finally:
                board.pop()

            if score >= beta:
                return beta

            alpha = max(alpha, score)

        return alpha

    # ------------------------------------------------------------------
    # NEGAMAX + ALPHA-BETA
    # ------------------------------------------------------------------

    def negamax(self, board, depth, alpha, beta, ply=0):
        self.nodes += 1
        self._check_time()

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
        original_beta = beta

        # A repeated position can have a path-dependent draw value.
        repetition_sensitive = board.is_repetition(2)
        key = self._tt_key(board)
        tt_move = None

        # Root only: limit the search to the network's top moves. A restricted
        # root must neither read nor write the transposition table, because its
        # result is not valid for the position in general.
        root_candidates = None
        if ply == 0:
            root_candidates = self._root_candidates(board, list(board.legal_moves))

        use_tt = not repetition_sensitive and root_candidates is None
        tt_entry = self.transposition_table.get(key) if use_tt else None

        if tt_entry is not None:
            self.tt_hits += 1
            tt_depth, tt_score, tt_flag, tt_move = tt_entry
            tt_score = self._score_from_tt(tt_score, ply)

            if tt_depth >= depth:
                if tt_flag == "EXACT":
                    return tt_score, tt_move
                if tt_flag == "LOWER":
                    alpha = max(alpha, tt_score)
                elif tt_flag == "UPPER":
                    beta = min(beta, tt_score)

                if alpha >= beta:
                    self.tt_cutoffs += 1
                    return tt_score, tt_move

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return (-MATE_SCORE + ply, None) if board.is_check() else (0, None)

        if root_candidates is not None:
            legal_moves = root_candidates

        best_move = None
        best_score = -INFINITY

        ordered_moves = self.order_moves(
            board,
            legal_moves,
            tt_move=tt_move,
            ply=ply,
            root=(ply == 0),
        )

        for move in ordered_moves:
            board.push(move)
            try:
                child_score, _ = self.negamax(
                    board, depth - 1, -beta, -alpha, ply + 1
                )
                score = -child_score
            finally:
                board.pop()

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                self.beta_cutoffs += 1

                if not board.is_capture(move) and not move.promotion:
                    killers = self.killers.setdefault(ply, [])
                    if move not in killers:
                        killers.insert(0, move)
                        del killers[2:]

                    history_key = (board.turn, move.from_square, move.to_square)
                    self.history[history_key] = min(
                        1_000_000,
                        self.history.get(history_key, 0) + depth * depth,
                    )
                break

        if use_tt:
            if best_score <= original_alpha:
                flag = "UPPER"
            elif best_score >= original_beta:
                flag = "LOWER"
            else:
                flag = "EXACT"

            self._store_tt(key, depth, best_score, flag, best_move, ply)

        return best_score, best_move

    # ------------------------------------------------------------------
    # PUBLIC SEARCH METHOD
    # ------------------------------------------------------------------

    def get_best_move(self, board, depth=None, time_limit=None):
        if board.is_game_over(claim_draw=True):
            return None

        search_depth = max(1, int(depth if depth is not None else self.depth))

        self.nodes = 0
        self.qnodes = 0
        self.tt_hits = 0
        self.tt_cutoffs = 0
        self.beta_cutoffs = 0

        self.start_time = time.perf_counter()
        self.deadline = (
            self.start_time + max(0.01, float(time_limit))
            if time_limit is not None
            else None
        )
        self._last_time_check = 0

        self.killers.clear()
        self.history.clear()
        self.root_neural_scores = {}

        # A previous search's TT is useful, but cap its size and use safe keys.
        # Calculate the network policy once per move, at the root only.
        self.root_neural_scores = self.neural_ordering(board)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            self.deadline = None
            return None

        best_move = legal_moves[0]
        best_score = 0
        completed_depth = 0

        for current_depth in range(1, search_depth + 1):
            try:
                self._check_time(force=True)
                score, move = self.negamax(
                    board, current_depth, -INFINITY, INFINITY
                )

                if move is not None:
                    best_move = move
                    best_score = score

                completed_depth = current_depth
                elapsed = time.perf_counter() - self.start_time

                print(
                    f"Depth {current_depth} | "
                    f"Score {best_score / 100:.2f} | "
                    f"Nodes {self.nodes:,} | "
                    f"QNodes {self.qnodes:,} | "
                    f"Time {elapsed:.2f}s"
                )

                if abs(best_score) >= MATE_THRESHOLD:
                    break

            except SearchTimeout:
                print(f"Time limit reached; keeping depth {completed_depth} result.")
                break

        elapsed = time.perf_counter() - self.start_time
        total_nodes = self.nodes + self.qnodes
        nps = int(total_nodes / elapsed) if elapsed > 0 else 0

        self.last_search_stats = {
            "best_move": best_move.uci() if best_move else None,
            "score_cp": best_score,
            "completed_depth": completed_depth,
            "nodes": self.nodes,
            "qnodes": self.qnodes,
            "total_nodes": total_nodes,
            "elapsed_seconds": elapsed,
            "nodes_per_second": nps,
            "tt_hits": self.tt_hits,
            "tt_cutoffs": self.tt_cutoffs,
            "beta_cutoffs": self.beta_cutoffs,
        }

        print(
            f"Search complete | Depth {completed_depth} | "
            f"Total nodes {total_nodes:,} | NPS {nps:,} | "
            f"TT hits {self.tt_hits:,}"
        )

        self.deadline = None
        return best_move