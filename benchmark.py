
import chess
import csv
import json
import time

from pathlib import Path
from engine import ChessEngine


# ============================================================
# CONFIGURATION
# ============================================================

SEARCH_DEPTH = 4
TIME_LIMIT = 10

TEST_NEURAL_NETWORK = True
TEST_CLASSICAL_ENGINE = True

OUTPUT_DIR = Path("results_benchmark")

CSV_FILE = OUTPUT_DIR / "benchmark_results.csv"
JSON_FILE = OUTPUT_DIR / "benchmark_summary.json"


# ============================================================
# TEST POSITIONS
# ============================================================

TEST_POSITIONS = [
    {
        "name": "Starting position",
        "fen": chess.STARTING_FEN,
        "test": "General search",
    },
    {
        "name": "Mate in 1",
        "fen": "7k/6pp/5KQ1/8/8/8/8/8 w - - 0 1",
        "test": "Checkmate detection",
    },
    {
        "name": "Capture queen",
        "fen": "4k3/8/8/8/8/4q3/4R3/4K3 w - - 0 1",
        "test": "High-value capture",
    },
    {
        "name": "Pawn promotion",
        "fen": "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "test": "Pawn promotion",
    },
]


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

def prepare_output_directory():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# POSITION VALIDATION
# ============================================================

def validate_result(board, move, test_name):

    if move is None or move not in board.legal_moves:
        return False, "No legal move returned"

    if test_name == "Checkmate detection":
        board.push(move)

        if board.is_checkmate():
            return True, "Checkmate found"

        return False, "Checkmate not found"

    if test_name == "High-value capture":
        captured_piece = board.piece_at(move.to_square)

        if captured_piece and captured_piece.piece_type == chess.QUEEN:
            return True, "Queen captured"

        return False, "Queen not captured"

    if test_name == "Pawn promotion":
        if move.promotion == chess.QUEEN:
            return True, "Promoted to queen"

        return False, "Queen promotion not found"

    return True, "Legal move returned"


# ============================================================
# RUN ONE EXPERIMENT
# ============================================================

def run_experiment(engine, position, engine_type):

    board = chess.Board(position["fen"])

    print("\n" + "=" * 65)
    print("ENGINE:", engine_type)
    print("POSITION:", position["name"])
    print("TEST:", position["test"])
    print("=" * 65)

    start_time = time.perf_counter()

    move = engine.get_best_move(
        board,
        depth=SEARCH_DEPTH,
        time_limit=TIME_LIMIT
    )

    elapsed = time.perf_counter() - start_time

    stats = engine.last_search_stats.copy()

    if move is not None and move in board.legal_moves:
        uci_move = move.uci()
        san_move = board.san(move)
    else:
        uci_move = None
        san_move = None

    passed, validation_message = validate_result(
        board,
        move,
        position["test"]
    )

    result = {
        "engine": engine_type,
        "position": position["name"],
        "test": position["test"],
        "best_move_uci": uci_move,
        "best_move_san": san_move,
        "passed": passed,
        "validation": validation_message,
        "completed_depth": stats.get("completed_depth", 0),
        "score_cp": stats.get("score_cp", 0),
        "nodes": stats.get("nodes", 0),
        "qnodes": stats.get("qnodes", 0),
        "total_nodes": stats.get("total_nodes", 0),
        "elapsed_seconds": round(elapsed, 4),
        "nodes_per_second": stats.get("nodes_per_second", 0),
        "tt_hits": stats.get("tt_hits", 0),
        "tt_cutoffs": stats.get("tt_cutoffs", 0),
        "beta_cutoffs": stats.get("beta_cutoffs", 0),
    }

    print("\nRESULT")
    print("Move:", san_move)
    print("Validation:", validation_message)
    print("Passed:", passed)
    print("Depth:", result["completed_depth"])
    print("Total nodes:", result["total_nodes"])
    print("Time:", result["elapsed_seconds"], "seconds")
    print("Nodes/sec:", result["nodes_per_second"])

    return result


# ============================================================
# SAVE CSV RESULTS
# ============================================================

def save_csv(results):

    if not results:
        return

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nCSV saved to:", CSV_FILE)


# ============================================================
# GENERATE SUMMARY
# ============================================================

def generate_summary(results):

    summary = {}

    for engine_type in ["Classical", "Neural Network"]:

        engine_results = [
            result for result in results
            if result["engine"] == engine_type
        ]

        if not engine_results:
            continue

        count = len(engine_results)

        summary[engine_type] = {
            "positions_tested": count,
            "passed_tests": sum(
                result["passed"] for result in engine_results
            ),
            "average_nodes": round(
                sum(r["total_nodes"] for r in engine_results) / count
            ),
            "average_time_seconds": round(
                sum(r["elapsed_seconds"] for r in engine_results) / count,
                4
            ),
            "average_nodes_per_second": round(
                sum(r["nodes_per_second"] for r in engine_results) / count
            ),
            "average_completed_depth": round(
                sum(r["completed_depth"] for r in engine_results) / count,
                2
            ),
        }

    report = {
        "benchmark_configuration": {
            "search_depth": SEARCH_DEPTH,
            "time_limit_seconds": TIME_LIMIT,
            "positions_tested": len(TEST_POSITIONS),
        },
        "results": summary,
        "individual_experiments": results,
    }

    with open(JSON_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("JSON report saved to:", JSON_FILE)

    return summary


# ============================================================
# DISPLAY COMPARISON
# ============================================================

def display_comparison(summary):

    print("\n")
    print("=" * 75)
    print("FINAL CHESS ENGINE BENCHMARK")
    print("=" * 75)

    headers = [
        "Engine",
        "Passed",
        "Avg Nodes",
        "Avg Time",
        "Avg NPS",
        "Avg Depth",
    ]

    print(
        f"{headers[0]:<18}"
        f"{headers[1]:<10}"
        f"{headers[2]:<13}"
        f"{headers[3]:<13}"
        f"{headers[4]:<13}"
        f"{headers[5]:<10}"
    )

    print("-" * 75)

    for engine_type, data in summary.items():

        print(
            f"{engine_type:<18}"
            f"{data['passed_tests']}/{data['positions_tested']:<9}"
            f"{data['average_nodes']:<13}"
            f"{data['average_time_seconds']:<13}"
            f"{data['average_nodes_per_second']:<13}"
            f"{data['average_completed_depth']:<10}"
        )

    print("=" * 75)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    prepare_output_directory()

    print("=" * 65)
    print("CHESS ENGINE BENCHMARK")
    print("=" * 65)

    print("Search depth:", SEARCH_DEPTH)
    print("Time limit:", TIME_LIMIT, "seconds")
    print("Positions:", len(TEST_POSITIONS))

    results = []

    configurations = []

    if TEST_CLASSICAL_ENGINE:
        configurations.append(("Classical", False))

    if TEST_NEURAL_NETWORK:
        configurations.append(("Neural Network", True))

    for engine_name, use_nn in configurations:

        print("\n\nInitializing:", engine_name)

        engine = ChessEngine(
            depth=SEARCH_DEPTH,
            use_neural_network=use_nn
        )

        for position in TEST_POSITIONS:

            result = run_experiment(
                engine,
                position,
                engine_name
            )

            results.append(result)

    save_csv(results)

    summary = generate_summary(results)

    display_comparison(summary)

    print("\nBENCHMARK COMPLETED")
    print("Results directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
