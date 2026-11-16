# Chess Engine - Machine Learning Project

A chess engine built for machine learning research. This repository contains the initial dataset inspection and processing pipeline.

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

### `inspect_dataset.py`

Explores chess game datasets to understand:
- Game length distributions
- Opening patterns
- Player ratings
- Move statistics

### `test_parser.py`

Validates move parsing functionality:
- SAN (Standard Algebraic Notation) parsing
- Move legality verification
- Board state reconstruction

### `build_dataset.py`

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
├── .gitignore            # Git ignore rules
├── inspect_dataset.py    # Dataset exploration script
├── test_parser.py        # Parser validation tests
├── build_dataset.py      # Dataset processing pipeline
├── gui.py                 # Local Flask web interface
├── templates/             # Web page markup
├── static/                # Web UI styles and behavior
└── screenshots/          # Documentation images
    ├── dataset-build.png
    ├── dataset-conection-test.png
    └── parsetest.png
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
