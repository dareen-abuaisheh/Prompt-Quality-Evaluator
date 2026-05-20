"""
split_dataset.py
----------------

Simple utility to split the cleaned Prompt Quality Evaluator dataset into
train, validation, and test files.

Input:
    data/prompt_quality_dataset_cleaned.json

Output:
    data/splits/train.json
    data/splits/validation.json
    data/splits/test.json
"""

import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split


# =========================================================
# Configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "prompt_quality_dataset_cleaned.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"

TRAIN_PATH = OUTPUT_DIR / "train.json"
VALIDATION_PATH = OUTPUT_DIR / "validation.json"
TEST_PATH = OUTPUT_DIR / "test.json"

RANDOM_STATE = 42


# =========================================================
# Helper Functions
# =========================================================

def load_dataset(path):
    """Load the cleaned JSON dataset."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_dataset(rows, path):
    """Save a split as a formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)
        file.write("\n")


def get_scores(rows):
    """Return the score values for stratified splitting."""
    return [row["score"] for row in rows]


def score_distribution(rows):
    """Return the score counts in sorted order."""
    counts = Counter(get_scores(rows))
    return dict(sorted(counts.items()))


def print_split_summary(name, rows):
    """Print the size and score distribution for one split."""
    print(f"{name} size: {len(rows)}")
    print(f"{name} score distribution: {score_distribution(rows)}")


# =========================================================
# Main Script
# =========================================================

def main():
    # Load the cleaned dataset.
    data = load_dataset(INPUT_PATH)

    # First split: 80% train and 20% temporary data.
    train_rows, temp_rows = train_test_split(
        data,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=get_scores(data),
    )

    # Second split: divide the temporary data equally into validation and test.
    validation_rows, test_rows = train_test_split(
        temp_rows,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=get_scores(temp_rows),
    )

    # Save each split to disk.
    save_dataset(train_rows, TRAIN_PATH)
    save_dataset(validation_rows, VALIDATION_PATH)
    save_dataset(test_rows, TEST_PATH)

    # Print a short summary for checking the split.
    print(f"Total rows: {len(data)}")
    print_split_summary("Train", train_rows)
    print_split_summary("Validation", validation_rows)
    print_split_summary("Test", test_rows)


if __name__ == "__main__":
    main()
