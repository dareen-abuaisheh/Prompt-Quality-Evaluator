"""
preprocess_dataset.py
---------------------

Simple preprocessing utility for the Prompt Quality Evaluator dataset.

This script prepares the already-cleaned dataset for later splitting and
fine-tuning. It keeps the original JSON structure and uses only Python's
standard library.

Input:
    data/prompt_quality_dataset_cleaned.json

Output:
    data/prompt_quality_dataset_preprocessed.json
"""

import json
import re
from collections import Counter
from pathlib import Path


# =========================================================
# Configuration
# =========================================================

INPUT_PATH = Path("../data/prompt_quality_dataset_cleaned.json")
OUTPUT_PATH = Path("../data/prompt_quality_dataset_preprocessed.json")

REQUIRED_FIELDS = ["task", "reference", "submission", "rubric", "score", "rationale"]
TEXT_FIELDS = ["task", "reference", "submission", "rationale"]


# =========================================================
# Text Helpers
# =========================================================

def normalize_text(value):
    """
    Normalize whitespace without changing meaning.

    - Converts None to an empty string.
    - Strips leading/trailing spaces.
    - Collapses repeated spaces/tabs.
    - Collapses repeated newlines.
    """
    if value is None:
        return ""

    text = str(value).strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def normalize_for_duplicate_check(text):
    """Normalize a submission before exact duplicate comparison."""
    return normalize_text(text).casefold()


# =========================================================
# Loading and Saving
# =========================================================

def load_dataset(path):
    """Load a JSON dataset from disk."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list of row objects.")

    return data


def save_dataset(rows, path):
    """Save rows as a formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)
        file.write("\n")


# =========================================================
# Validation and Normalization
# =========================================================

def is_valid_row(row):
    """
    Check whether a row has the required fields and valid values.

    Invalid rows are removed because they are unsafe for fine-tuning.
    """
    if not isinstance(row, dict):
        return False

    for field in REQUIRED_FIELDS:
        if field not in row:
            return False

    if not isinstance(row["score"], int) or row["score"] < 0 or row["score"] > 4:
        return False

    for field in TEXT_FIELDS:
        if normalize_text(row[field]) == "":
            return False

    return True


def normalize_row(row):
    """
    Normalize text fields and keep the original schema.

    The field order is kept exactly as:
    task, reference, submission, rubric, score, rationale
    """
    return {
        "task": normalize_text(row["task"]),
        "reference": normalize_text(row["reference"]),
        "submission": normalize_text(row["submission"]),
        "rubric": row["rubric"],
        "score": row["score"],
        "rationale": normalize_text(row["rationale"]),
    }


def validate_and_normalize_rows(rows):
    """
    Remove invalid rows and normalize the valid ones.

    Returns:
        normalized_rows, invalid_rows_removed
    """
    normalized_rows = []
    invalid_rows_removed = 0

    for row in rows:
        if not is_valid_row(row):
            invalid_rows_removed += 1
            continue

        normalized_rows.append(normalize_row(row))

    return normalized_rows, invalid_rows_removed


def remove_duplicate_submissions(rows):
    """
    Remove exact duplicate submissions.

    The first occurrence is kept. Later rows with the same normalized
    submission text are removed.
    """
    seen_submissions = set()
    unique_rows = []
    removed_duplicates = 0

    for row in rows:
        key = normalize_for_duplicate_check(row["submission"])

        if key in seen_submissions:
            removed_duplicates += 1
            continue

        seen_submissions.add(key)
        unique_rows.append(row)

    return unique_rows, removed_duplicates


# =========================================================
# Reporting Helpers
# =========================================================

def score_distribution(rows):
    """Return a sorted score distribution for scores 0 through 4."""
    counts = Counter(row["score"] for row in rows)
    return {score: counts.get(score, 0) for score in range(5)}


def count_score_4_leakage(rows):
    """Count score-4 rows where submission is still identical to reference."""
    return sum(
        1
        for row in rows
        if row["score"] == 4
        and normalize_for_duplicate_check(row["submission"])
        == normalize_for_duplicate_check(row["reference"])
    )


def count_improved_references(rows):
    """
    Count references that look like improved gold-standard prompts.

    The cleaned dataset marks improved references with stronger prompt
    structure such as a leading "Task:" label and explicit output guidance.
    """
    improved = 0

    for row in rows:
        reference = normalize_text(row["reference"])
        lower_reference = reference.lower()

        has_task_label = reference.startswith("Task:")
        has_output_guidance = any(
            phrase in lower_reference
            for phrase in ["return ", "json", "bullet", "labeled fields"]
        )

        if has_task_label and has_output_guidance:
            improved += 1

    return improved


# =========================================================
# Main Pipeline
# =========================================================

def main():
    """Run preprocessing and print a concise summary."""
    original_rows = load_dataset(INPUT_PATH)
    original_row_count = len(original_rows)
    original_score_distribution = score_distribution(
        [row for row in original_rows if isinstance(row, dict) and "score" in row]
    )

    normalized_rows, invalid_rows_removed = validate_and_normalize_rows(original_rows)
    score_4_leakage_before = count_score_4_leakage(normalized_rows)
    deduplicated_rows, removed_duplicates = remove_duplicate_submissions(normalized_rows)

    score_4_leakage_after = count_score_4_leakage(deduplicated_rows)
    score_4_leakage_fixed = score_4_leakage_before - score_4_leakage_after
    improved_references = count_improved_references(deduplicated_rows)

    save_dataset(deduplicated_rows, OUTPUT_PATH)

    print("=" * 72)
    print("Prompt Quality Dataset Preprocessing Summary")
    print("=" * 72)
    print(f"Input file:  {INPUT_PATH}")
    print(f"Output file: {OUTPUT_PATH}")
    print()
    print(f"Original row count:     {original_row_count}")
    print(f"Final row count:        {len(deduplicated_rows)}")
    print(f"Removed duplicates:     {removed_duplicates}")
    print(f"Invalid rows removed:   {invalid_rows_removed}")
    print()
    print(f"Score distribution before: {original_score_distribution}")
    print(f"Score distribution after:  {score_distribution(deduplicated_rows)}")
    print()
    print(f"Score-4 leakage rows fixed:     {score_4_leakage_fixed}")
    print(f"Score-4 leakage rows remaining: {score_4_leakage_after}")
    print(f"References improved:            {improved_references}")
    print("=" * 72)


if __name__ == "__main__":
    main()
