
import os
import csv
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/chess_training_data_1m.npz"

CHECKPOINT_DIR = "checkpoints_1m"
RESULTS_DIR = "results_1m"

BATCH_SIZE = 64
EPOCHS = 20

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

INPUT_CHANNELS = 18
NUM_CLASSES = 20480

CHANNELS = 128
NUM_RESIDUAL_BLOCKS = 4

SEED = 42

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("CHESS NEURAL NETWORK - 1 MILLION POSITIONS")
print("=" * 65)

print("Device:", device)

if device.type == "cuda":

    print("GPU:", torch.cuda.get_device_name(0))

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )

    torch.backends.cudnn.benchmark = True


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading 1M dataset...")

data = np.load(DATA_PATH)

X_train = torch.from_numpy(data["X_train"].copy())
y_train = torch.from_numpy(data["y_train"].copy())

X_val = torch.from_numpy(data["X_val"].copy())
y_val = torch.from_numpy(data["y_val"].copy())

X_test = torch.from_numpy(data["X_test"].copy())
y_test = torch.from_numpy(data["y_test"].copy())

data.close()

print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

print("Input shape:", tuple(X_train.shape))
print("Label shape:", tuple(y_train.shape))

if tuple(X_train.shape[1:]) != (18, 8, 8):
    raise ValueError("Incorrect input shape.")

if y_train.min() < 0 or y_train.max() >= NUM_CLASSES:
    raise ValueError("Invalid training labels.")

if y_val.min() < 0 or y_val.max() >= NUM_CLASSES:
    raise ValueError("Invalid validation labels.")

if y_test.min() < 0 or y_test.max() >= NUM_CLASSES:
    raise ValueError("Invalid test labels.")


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    TensorDataset(X_val, y_val),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(self, channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
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
# METRICS
# ============================================================

def calculate_accuracy(logits, targets):

    predictions = logits.topk(5, dim=1).indices

    top1 = (
        predictions[:, 0] == targets
    ).sum().item()

    top5 = (
        predictions == targets.unsqueeze(1)
    ).any(dim=1).sum().item()

    return top1, top5


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, loss_function):

    model.eval()

    total_loss = 0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    with torch.no_grad():

        for inputs, targets in loader:

            inputs = inputs.to(
                device,
                dtype=torch.float32,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):

                logits = model(inputs)
                loss = loss_function(logits, targets)

            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size

            top1, top5 = calculate_accuracy(
                logits,
                targets
            )

            total_top1 += top1
            total_top5 += top5
            total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "top1": 100 * total_top1 / total_samples,
        "top5": 100 * total_top5 / total_samples
    }


# ============================================================
# INITIALIZE
# ============================================================

print("\nInitializing model...")

model = ChessNet().to(device)

parameter_count = sum(
    p.numel() for p in model.parameters()
)

print("Parameters:", f"{parameter_count:,}")
print("Channels:", CHANNELS)
print("Residual blocks:", NUM_RESIDUAL_BLOCKS)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device.type == "cuda")
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0
best_epoch = 0

best_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "best_chess_model_1m.pt"
)

history_file = os.path.join(
    RESULTS_DIR,
    "training_history.csv"
)

history = []

start_time = time.time()

print("\nStarting training...")
print("=" * 65)

for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    model.train()

    running_loss = 0
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    for batch_index, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.to(
            device,
            dtype=torch.float32,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda")
        ):

            logits = model(inputs)
            loss = loss_function(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)

        running_loss += loss.item() * batch_size

        top1, top5 = calculate_accuracy(
            logits.detach(),
            targets
        )

        total_top1 += top1
        total_top5 += top5
        total_samples += batch_size

        if (batch_index + 1) % 200 == 0:

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {batch_index + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = running_loss / total_samples

    train_top1 = (
        100 * total_top1 / total_samples
    )

    train_top5 = (
        100 * total_top5 / total_samples
    )

    # Validation
    val = evaluate(
        model,
        val_loader,
        loss_function
    )

    scheduler.step(val["loss"])

    epoch_time = time.time() - epoch_start

    current_lr = optimizer.param_groups[0]["lr"]

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_top1": train_top1,
        "train_top5": train_top5,
        "val_loss": val["loss"],
        "val_top1": val["top1"],
        "val_top5": val["top5"],
        "learning_rate": current_lr,
        "epoch_seconds": epoch_time
    }

    history.append(record)

    print("\n" + "=" * 65)
    print(f"EPOCH {epoch}/{EPOCHS}")
    print("=" * 65)

    print(f"Training loss:    {train_loss:.4f}")
    print(f"Training top-1:   {train_top1:.2f}%")
    print(f"Training top-5:   {train_top5:.2f}%")

    print(f"Validation loss:  {val['loss']:.4f}")
    print(f"Validation top-1: {val['top1']:.2f}%")
    print(f"Validation top-5: {val['top5']:.2f}%")

    print(f"Learning rate:    {current_lr:.7f}")
    print(f"Epoch time:       {epoch_time:.1f}s")

    if device.type == "cuda":

        print(
            "GPU memory:",
            round(
                torch.cuda.memory_allocated() / 1024**3,
                2
            ),
            "GB"
        )

    # Save best checkpoint
    if val["top1"] > best_val_accuracy:

        best_val_accuracy = val["top1"]
        best_epoch = epoch

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val["top1"],
                "val_loss": val["loss"],
                "channels": CHANNELS,
                "residual_blocks": NUM_RESIDUAL_BLOCKS
            },
            best_checkpoint
        )

        print("New best checkpoint saved!")

    # Save history after every epoch
    with open(history_file, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys()
        )

        writer.writeheader()
        writer.writerows(history)

    print("=" * 65)


# ============================================================
# FINAL TEST
# ============================================================

print("\nTraining finished.")

print("Loading best checkpoint...")

checkpoint = torch.load(
    best_checkpoint,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

test = evaluate(
    model,
    test_loader,
    loss_function
)

total_time = time.time() - start_time

final_results = {
    "dataset": "Lichess standard chess games",
    "training_positions": len(X_train),
    "validation_positions": len(X_val),
    "testing_positions": len(X_test),
    "model_parameters": parameter_count,
    "channels": CHANNELS,
    "residual_blocks": NUM_RESIDUAL_BLOCKS,
    "batch_size": BATCH_SIZE,
    "epochs_completed": len(history),
    "best_epoch": best_epoch,
    "best_validation_top1": best_val_accuracy,
    "test_loss": test["loss"],
    "test_top1": test["top1"],
    "test_top5": test["top5"],
    "total_training_seconds": total_time,
    "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
}

metrics_file = os.path.join(
    RESULTS_DIR,
    "final_metrics.json"
)

with open(metrics_file, "w") as file:
    json.dump(final_results, file, indent=4)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 65)
print("FINAL TEST RESULTS - 1 MILLION POSITIONS")
print("=" * 65)

print("Best epoch:", best_epoch)

print(f"Validation top-1: {best_val_accuracy:.2f}%")
print(f"Test loss:        {test['loss']:.4f}")
print(f"Test top-1:       {test['top1']:.2f}%")
print(f"Test top-5:       {test['top5']:.2f}%")

print(f"Training time:    {total_time / 60:.2f} minutes")

print("\nCheckpoint:", best_checkpoint)
print("History:", history_file)
print("Metrics:", metrics_file)

print("\nTraining completed successfully!")
