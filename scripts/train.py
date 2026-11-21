
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/chess_training_data.npz"
CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

INPUT_CHANNELS = 18
NUM_CLASSES = 20480
CHANNELS = 128
NUM_RESIDUAL_BLOCKS = 4

SEED = 42


# ============================================================
# REPRODUCIBILITY AND DEVICE
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("CHESS NEURAL NETWORK TRAINING")
print("=" * 60)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )
    torch.backends.cudnn.benchmark = True

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

data = np.load(DATA_PATH)

X_train = torch.from_numpy(data["X_train"].copy())
y_train = torch.from_numpy(data["y_train"].copy())

X_val = torch.from_numpy(data["X_val"].copy())
y_val = torch.from_numpy(data["y_val"].copy())

X_test = torch.from_numpy(data["X_test"].copy())
y_test = torch.from_numpy(data["y_test"].copy())

print("Training positions:", len(X_train))
print("Validation positions:", len(X_val))
print("Testing positions:", len(X_test))

print("Input shape:", tuple(X_train.shape))
print("Label shape:", tuple(y_train.shape))

print("Input value range:",
      int(X_train.min()), "to", int(X_train.max()))

print("Label range:",
      int(y_train.min()), "to", int(y_train.max()))

if X_train.ndim != 4 or X_train.shape[1:] != (18, 8, 8):
    raise ValueError("Unexpected input shape. Expected (N, 18, 8, 8).")

if y_train.min() < 0 or y_train.max() >= NUM_CLASSES:
    raise ValueError("Training labels are outside the 20480 output classes.")

if y_val.min() < 0 or y_val.max() >= NUM_CLASSES:
    raise ValueError("Validation labels are outside the 20480 output classes.")

if y_test.min() < 0 or y_test.max() >= NUM_CLASSES:
    raise ValueError("Test labels are outside the 20480 output classes.")


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(device.type == "cuda")
)

test_loader = DataLoader(
    test_dataset,
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
# CHESS POLICY NETWORK
# ============================================================

class ChessNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Initial feature extraction
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

        # Residual tower
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(CHANNELS)
                for _ in range(NUM_RESIDUAL_BLOCKS)
            ]
        )

        # Policy head
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
# ACCURACY METRICS
# ============================================================

def calculate_accuracy(logits, targets):

    top5 = logits.topk(5, dim=1).indices

    top1_correct = (top5[:, 0] == targets).sum().item()

    top5_correct = (
        top5 == targets.unsqueeze(1)
    ).any(dim=1).sum().item()

    return top1_correct, top5_correct


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, loss_function):

    model.eval()

    total_loss = 0.0
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

            top1, top5 = calculate_accuracy(logits, targets)

            total_top1 += top1
            total_top5 += top5
            total_samples += batch_size

    return (
        total_loss / total_samples,
        100.0 * total_top1 / total_samples,
        100.0 * total_top5 / total_samples
    )


# ============================================================
# INITIALIZE MODEL
# ============================================================

print("\nCreating neural network...")

model = ChessNet().to(device)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Model parameters:", f"{parameter_count:,}")
print("Residual blocks:", NUM_RESIDUAL_BLOCKS)
print("Channels:", CHANNELS)
print("Output classes:", NUM_CLASSES)

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
# TRAINING LOOP
# ============================================================

best_val_accuracy = 0.0
best_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "best_chess_model.pt"
)

print("\nStarting training...")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
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

        top1, top5 = calculate_accuracy(logits.detach(), targets)

        total_top1 += top1
        total_top5 += top5
        total_samples += batch_size

        if (batch_index + 1) % 100 == 0:

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {batch_index + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = running_loss / total_samples
    train_top1 = 100.0 * total_top1 / total_samples
    train_top5 = 100.0 * total_top5 / total_samples

    # Validation
    val_loss, val_top1, val_top5 = evaluate(
        model,
        val_loader,
        loss_function
    )

    scheduler.step(val_loss)

    print("\n" + "-" * 60)
    print(f"EPOCH {epoch}/{EPOCHS}")
    print("-" * 60)

    print(f"Training loss:       {train_loss:.4f}")
    print(f"Training top-1:      {train_top1:.2f}%")
    print(f"Training top-5:      {train_top5:.2f}%")

    print(f"Validation loss:     {val_loss:.4f}")
    print(f"Validation top-1:    {val_top1:.2f}%")
    print(f"Validation top-5:    {val_top5:.2f}%")

    print(f"Learning rate:       {optimizer.param_groups[0]['lr']:.6f}")

    if device.type == "cuda":
        print(
            "GPU memory allocated:",
            round(torch.cuda.memory_allocated() / 1024**3, 2),
            "GB"
        )

    # Save best model based on validation top-1
    if val_top1 > best_val_accuracy:

        best_val_accuracy = val_top1

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_top1,
                "val_loss": val_loss
            },
            best_checkpoint
        )

        print("New best model saved!")

    print("-" * 60)


# ============================================================
# FINAL TEST
# ============================================================

print("\nTraining finished.")

print("Loading best model...")

checkpoint = torch.load(
    best_checkpoint,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

test_loss, test_top1, test_top5 = evaluate(
    model,
    test_loader,
    loss_function
)

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print("Best epoch:", checkpoint["epoch"])
print(f"Best validation top-1: {checkpoint['val_accuracy']:.2f}%")
print(f"Test loss:             {test_loss:.4f}")
print(f"Test top-1 accuracy:   {test_top1:.2f}%")
print(f"Test top-5 accuracy:   {test_top5:.2f}%")

print("\nModel saved at:", best_checkpoint)
print("Training complete!")
