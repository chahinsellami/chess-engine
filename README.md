# Chess Engine - Machine Learning Project

A chess engine built for machine learning research. This repository contains the initial dataset inspection and processing pipeline.

## Quick Start

```bash
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python gui.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Overview

This project explores using machine learning to improve chess engine decision-making. The approach involves:

1. **Dataset Collection**: Gathering large volumes of chess games from Lichess API
2. **Data Processing**: Cleaning and structuring game data for ML training
3. **Model Training**: Developing neural networks to evaluate positions and suggest moves
4. **Engine Integration**: Combining ML models with traditional chess evaluation

## Setup

### Prerequisites

- Python 3.11+
- Git

### Installation

```bash
# Create virtual environment
py -3.11 -m venv .venv

# Activate
# Windows: .\.venv\Scripts\Activate.ps1
# Linux/Mac: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Key packages:
- `datasets` - Loading and processing large chess datasets
- `python-chess` - Chess board representation and move parsing
- `pandas` - Data manipulation and analysis
- `pyarrow` - Efficient data serialization
- `tqdm` - Progress bars for long operations
- `torch` - Building and training machine learning models

## Dataset Inspection

The project includes tools for exploring chess datasets:

### `scripts/inspect_dataset.py`

Explores chess game datasets to understand:
- Game length distributions
- Opening patterns
- Player ratings
- Move statistics

### `test_tactical_positions.py`

Validates move parsing functionality:
- SAN (Standard Algebraic Notation) parsing
- Move legality verification
- Board state reconstruction

### `scripts/build_dataset.py`

Processes raw chess data into ML-ready format:
- Fetches games from Lichess API
- Validates API connectivity
- Structures data for training pipelines

## Web GUI

The project includes a local web interface for playing against the engine and
checking the machine-learning setup.

```bash
python gui.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in a browser. The interface
includes:

- Responsive chess board with legal-move highlights
- Automatic engine replies using depth-3 search
- Move history, undo, new game, and pawn promotion choices
- Live position evaluation and engine status
- **Validation** tab for PyTorch, checkpoint, dataset, board-rule, and training
  metric checks

### Opponent Modes

The game panel supports:

- **Human vs Neural Bot**
- **Human vs Stockfish**
- **Engine vs Engine**, with Neural Bot or Stockfish on either side

Stockfish is loaded from `engines/stockfish/stockfish19.exe`. Levels 1–8 adjust
Stockfish skill and search time. The executable is not downloaded or modified
by the application.

## Screenshots

The following screenshots document the dataset inspection and preparation process:

![Dataset Build Pipeline](screenshots/dataset-build.png)
*Dataset structure and build pipeline output*

![API Connection Test](screenshots/dataset-conection-test.png)
*API connectivity validation results*

![Parser Test Cases](screenshots/parsetest.png)
*Move parser test cases and validation*

![Data Preparation Pipeline](screenshots/data-prepare.png)
*Data preprocessing pipeline for ML training*

## Project Structure

```
chess-engine/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .gitignore             # Local and generated-file ignore rules
├── gui.py                 # Local Flask web entry point
├── engine.py              # Chess search and evaluation
├── play.py                # Neural-network inference and CLI play
├── prepare_data.py        # 1-million-position tensor preparation
├── train_1m.py            # 1-million-position model training
├── test_tactical_positions.py # Chess and engine smoke tests
├── data/
│   ├── positions_1m.parquet
│   └── chess_training_data_1m.npz
├── checkpoints_1m/
├── results_1m/            # 1-million-position training metrics
├── scripts/               # Dataset builders and legacy utilities
├── templates/             # Flask page markup
├── static/                # Web UI styles, scripts, and SVG pieces
├── screenshots/           # Pipeline and app screenshots
├── docs/                  # Design and project documentation
└── archive/               # Preserved standard-size experiments
```

The board uses the open-source Cburnett chess piece artwork from
[Lichess](https://github.com/lichess-org/lila/tree/master/public/piece/cburnett).

## Next Steps

- Implement chess engine core logic (board representation, move generation)
- Develop position evaluation functions
- Integrate ML model training pipeline
- Add unit tests for engine components

## License

MIT License - feel free to use and modify for your own machine learning experiments.
