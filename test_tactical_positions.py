
import chess

from engine import ChessEngine


# ==========================================
# CONFIGURATION
# ==========================================

SEARCH_DEPTH = 4
TIME_LIMIT = 10

USE_NEURAL_NETWORK = False


# ==========================================
# TEST POSITIONS
# ==========================================

TEST_POSITIONS = {
    "Mate in 1": "7k/6pp/5KQ1/8/8/8/8/8 w - - 0 1",

    "Capture queen": "4k3/8/8/8/8/4q3/4R3/4K3 w - - 0 1",

    "Pawn promotion": "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
}


# ==========================================
# TESTING FUNCTION
# ==========================================

def test_position(engine, name, fen):

    board = chess.Board(fen)

    print("\n" + "=" * 55)
    print(f"TEST: {name}")
    print("=" * 55)

    print("\nInitial position:")
    print(board)

    print("\nSearching for the best move...")

    move = engine.get_best_move(
        board,
        depth=SEARCH_DEPTH,
        time_limit=TIME_LIMIT
    )

    if move is None:
        print("No legal move found.")
        return

    print("\nEngine decision:")
    print("UCI move:", move)
    print("SAN move:", board.san(move))

    print("\nSearch statistics:")

    for key, value in engine.last_search_stats.items():
        print(f"{key}: {value}")


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("TACTICAL CHESS ENGINE TEST")
    print("=" * 55)

    print("Search depth:", SEARCH_DEPTH)
    print("Time limit:", TIME_LIMIT, "seconds")
    print("Neural network:", USE_NEURAL_NETWORK)

    engine = ChessEngine(
        depth=SEARCH_DEPTH,
        use_neural_network=USE_NEURAL_NETWORK
    )

    for name, fen in TEST_POSITIONS.items():
        test_position(engine, name, fen)

    print("\n" + "=" * 55)
    print("ALL TESTS COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()
