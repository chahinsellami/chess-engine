
from datasets import load_dataset

print("Connecting to Hugging Face...")

dataset = load_dataset(
    "Lichess/standard-chess-games",
    split="train",
    streaming=True
)

print("\nConnected successfully!")
print("Reading the first 3 games...\n")

for i, game in enumerate(dataset):

    print(f"\n========== GAME {i + 1} ==========")

    for key, value in game.items():

        if key == "movetext":
            print(f"\n{key}:")
            print(str(value)[:1500])
        else:
            print(f"{key}: {value}")

    if i == 2:
        break

print("\nInspection completed!")
