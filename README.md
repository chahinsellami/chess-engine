# Chess Engine

A chess engine that combines a neural network (trained on 1 million Lichess positions) with alpha-beta search. It has a local web GUI where you can play against it, against Stockfish, or watch engines play each other.

## Quick Start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python gui.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). Requires Python 3.11+.

## Features

- Play as human vs Neural Bot, human vs Stockfish, or engine vs engine
- Legal-move highlights, move history, undo, promotion choices
- Live position evaluation
- **Validation** tab checking PyTorch, checkpoint, dataset, and training metrics

Stockfish is optional: place the executable at `engines/stockfish/stockfish19.exe`. Levels 1–8 adjust its skill and search time.

## Model

A residual CNN (4 blocks, 128 channels, 43M parameters) that predicts the next move from a board position.

| Metric | Result |
|---|---|
| Training / validation / test positions | 801k / 99k / 100k |
| Test top-1 accuracy | 33.0% |
| Test top-5 accuracy | 69.2% |
| Training time | ~46 min (RTX 5070 Ti) |

Full metrics: [results_1m/final_metrics.json](results_1m/final_metrics.json).

## Retrain

```bash
python scripts/build_dataset_1m.py   # download games from Lichess
python prepare_data.py               # build tensors
python train_1m.py                   # train the model
```

## Project Layout

| Path | Purpose |
|---|---|
| `gui.py`, `templates/`, `static/` | Flask web app |
| `engine.py` | Search and evaluation |
| `play.py` | Neural-network inference and CLI play |
| `stockfish_opponent.py` | Stockfish wrapper |
| `prepare_data.py`, `train_1m.py` | Data prep and training |
| `scripts/` | Dataset builders and utilities |
| `checkpoints_1m/`, `results_1m/` | Trained model and metrics |
| `test_tactical_positions.py`, `test_stockfish.py` | Smoke tests |

## Credits

Piece artwork: [Cburnett set](https://github.com/lichess-org/lila/tree/master/public/piece/cburnett) from Lichess. Games: [Lichess](https://lichess.org) open database.

## License

MIT
