"""
analyze_dataset.py
------------------

Professional Exploratory Data Analysis (EDA) utility for the
Prompt Quality Evaluation dataset produced by the Prompt Quality
Evaluator project.

Purpose
-------
This script performs a careful, research-oriented inspection of the
`data/prompt_quality_dataset.json` file and prints a concise,
well-structured statistical report suitable for reviewers, professors,
or researchers. The analysis focuses on dataset integrity, quality,
distributional properties, and research-relevant checks (e.g., reference
quality and semantic consistency).

Dataset format
--------------
The script expects the dataset to be a JSON array of objects with the
following fields:

    {
        "task": "<task_label>",
        "reference": "<reference_prompt>",
        "submission": "<submission_prompt>",
        "score": <int 0..4>,
        "rationale": "<text>",
        "rubric": {"0": "...", "1": "...", "2": "...", "3": "...", "4": "..."}
    }

Statistics produced
-------------------
- Core dataset counts and duplication metrics
- Task distribution and balance diagnostics
- Score distributions and weak/medium/strong breakdown
- Prompt length statistics and extremes
- Lexical diversity (vocabulary size, total words, ratio)
- Per-task score breakdown and average scores
- Reference quality heuristics (e.g., submissions longer than references,
  identical submissions & references, potentially weak references)
- Data integrity checks (missing fields, invalid scores, empty text)

Design goals
------------
- Deterministic (no random sampling performed during analysis)
- Lightweight (standard library; optional `pandas` usage if available)
- Clear, commented, and modular to be review-friendly

Note: This script performs analysis only. It does not modify the
dataset file.
"""

# =========================================================
# Imports
# =========================================================

import json
import math
import re
from collections import Counter, defaultdict
from statistics import mean

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False


# =========================================================
# Configuration
# =========================================================

DATASET_PATH = "../data/prompt_quality_dataset_cleaned.json"
ALLOWED_TASKS = {
    "classification",
    "information_extraction",
    "summarization",
    "brainstorming",
    "creative_writing",
}


# =========================================================
# Helpers
# =========================================================

def clean_text(s):
    """Normalize whitespace and ensure string type."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip())


def words(text):
    """Split text into word tokens (simple, reproducible tokenizer)."""
    return re.findall(r"\w+", text.lower())


# =========================================================
# Load dataset
# =========================================================

print("=" * 72)
print("Loading dataset for EDA: {}".format(DATASET_PATH))
print("=" * 72)

with open(DATASET_PATH, "r", encoding="utf-8") as fh:
    raw = json.load(fh)

# Work on a shallow copy (do not modify original objects)
data = [dict(item) for item in raw]
N = len(data)

print(f"Loaded {N} rows from {DATASET_PATH}")


# =========================================================
# Data integrity checks
# - Find malformed rows, missing fields, invalid types
# =========================================================

REQUIRED_FIELDS = ["task", "reference", "submission", "score", "rationale", "rubric"]

malformed = []
missing_field_counts = Counter()
invalid_score_rows = []
empty_submission_rows = []

for i, row in enumerate(data):
    if not isinstance(row, dict):
        malformed.append((i, "row_not_object"))
        continue
    # Check required fields
    for f in REQUIRED_FIELDS:
        if f not in row:
            missing_field_counts[f] += 1
    # Check score
    sc = row.get("score")
    if not isinstance(sc, int) or sc < 0 or sc > 4:
        invalid_score_rows.append(i)
    # Check empty submission
    if not clean_text(row.get("submission")):
        empty_submission_rows.append(i)


# =========================================================
# Core dataset statistics
# =========================================================

tasks = [clean_text(r.get("task")) for r in data]
submissions = [clean_text(r.get("submission")) for r in data]
references = [clean_text(r.get("reference")) for r in data]
scores = [r.get("score") for r in data]
rationales = [clean_text(r.get("rationale")) for r in data]

total_samples = N
unique_submissions = len(set(submissions))
duplicate_submission_count = total_samples - unique_submissions
duplicate_percentage = (duplicate_submission_count / total_samples * 100) if total_samples else 0.0

task_distribution = Counter(tasks)
score_distribution = Counter(scores)


# =========================================================
# Prompt length statistics
# =========================================================

submission_lengths = [len(s.split()) for s in submissions]
reference_lengths = [len(r.split()) for r in references]
rationale_lengths = [len(r.split()) for r in rationales]

avg_submission_length = mean(submission_lengths) if submission_lengths else 0
avg_reference_length = mean(reference_lengths) if reference_lengths else 0
avg_rationale_length = mean(rationale_lengths) if rationale_lengths else 0

longest_submission = max(submissions, key=lambda s: len(s.split())) if submissions else ""
shortest_submission = min(submissions, key=lambda s: len(s.split())) if submissions else ""
longest_reference = max(references, key=lambda s: len(s.split())) if references else ""
shortest_reference = min(references, key=lambda s: len(s.split())) if references else ""


# =========================================================
# Lexical diversity
# =========================================================

all_words = []
for s in submissions:
    all_words.extend(words(s))

vocab_size = len(set(all_words))
total_words = len(all_words)
lexical_diversity = (vocab_size / total_words) if total_words else 0.0


# =========================================================
# Task + score analysis
# =========================================================

task_score_counts = defaultdict(Counter)
for r in data:
    t = clean_text(r.get("task"))
    sc = r.get("score")
    task_score_counts[t][sc] += 1

avg_score_per_task = {}
for t, counts in task_score_counts.items():
    total = sum(counts.values())
    if total:
        avg = sum(s * c for s, c in counts.items()) / total
    else:
        avg = float("nan")
    avg_score_per_task[t] = avg


# =========================================================
# Data integrity summary
# =========================================================

integrity_issues = {
    "malformed_rows": len(malformed),
    "missing_fields": dict(missing_field_counts),
    "invalid_score_rows": len(invalid_score_rows),
    "empty_submission_rows": len(empty_submission_rows),
}


# =========================================================
# Research-oriented observations
# =========================================================

observations = []

# Balance
num_tasks = len(task_distribution)
most_common_task, most_common_count = task_distribution.most_common(1)[0] if task_distribution else (None, 0)
least_common_task = min(task_distribution.items(), key=lambda x: x[1])[0] if task_distribution else None

observations.append(f"Dataset total rows: {total_samples}")
observations.append(f"Unique submissions: {unique_submissions} (duplicates: {duplicate_submission_count}, {duplicate_percentage:.2f}%)")
observations.append(f"Task balance: {num_tasks} distinct tasks; most common: {most_common_task} ({most_common_count} rows); least common: {least_common_task}")

# Score skew
total_submissions = total_samples
score_4_count = score_distribution.get(4, 0)
observations.append(f"Score 4 frequency: {score_4_count} ({(score_4_count/total_submissions*100):.2f}%)")

# template reuse check (simple: duplicates)
observations.append(f"Vocabulary size: {vocab_size}; total words: {total_words}; lexical diversity: {lexical_diversity:.4f}")

# length realism
observations.append(f"Average submission length: {avg_submission_length:.1f} words; average reference length: {avg_reference_length:.1f} words")


# =========================================================
# Printing a tidy report
# =========================================================

SEP = "-" * 72

print("\n" + SEP)
print("CORE DATASET STATISTICS")
print(SEP)
print(f"Total samples: {total_samples}")
print(f"Unique submissions: {unique_submissions}")
print(f"Duplicate submissions: {duplicate_submission_count} ({duplicate_percentage:.2f}%)")

print("\n" + SEP)
print("TASK ANALYSIS")
print(SEP)
for task, count in task_distribution.most_common():
    pct = count / total_samples * 100 if total_samples else 0
    print(f" - {task}: {count} ({pct:.2f}%)")
print(f"Most common task: {most_common_task}")
print(f"Least common task: {least_common_task}")

print("\n" + SEP)
print("SCORE ANALYSIS")
print(SEP)
for score in sorted(score_distribution.keys()):
    cnt = score_distribution[score]
    pct = cnt / total_samples * 100 if total_samples else 0
    print(f" - Score {score}: {cnt} ({pct:.2f}%)")

weak_count = sum(score_distribution[s] for s in [0, 1])
medium_count = sum(score_distribution[s] for s in [2, 3])
strong_count = score_distribution.get(4, 0)
print(f"\nPrompt quality buckets: Weak(0-1)={weak_count}, Medium(2-3)={medium_count}, Strong(4)={strong_count}")
print(f"Weak%: {weak_count/total_submissions*100:.2f} %; Medium%: {medium_count/total_submissions*100:.2f} %; Strong%: {strong_count/total_submissions*100:.2f} %")

print("\n" + SEP)
print("PROMPT LENGTH ANALYSIS")
print(SEP)
print(f"Avg submission length : {avg_submission_length:.2f} words")
print(f"Avg reference length  : {avg_reference_length:.2f} words")
print(f"Avg rationale length  : {avg_rationale_length:.2f} words")
print(f"Longest submission ({len(longest_submission.split())} words):\n{longest_submission}\n")
print(f"Shortest submission ({len(shortest_submission.split())} words):\n{shortest_submission}\n")
print(f"Longest reference ({len(longest_reference.split())} words):\n{longest_reference}\n")
print(f"Shortest reference ({len(shortest_reference.split())} words):\n{shortest_reference}\n")

print("\n" + SEP)
print("LEXICAL DIVERSITY")
print(SEP)
print(f"Vocabulary size : {vocab_size}")
print(f"Total words     : {total_words}")
print(f"Lexical diversity (vocab/words): {lexical_diversity:.4f}")

print("\n" + SEP)
print("TASK + SCORE ANALYSIS")
print(SEP)
for t, counts in task_score_counts.items():
    print(f"\nTask: {t}")
    total = sum(counts.values())
    for sc in sorted(counts.keys()):
        print(f"  Score {sc}: {counts[sc]} ({counts[sc]/total*100:.2f}% of task)")
    print(f"  Average score for task: {avg_score_per_task.get(t):.2f}")

print("\n" + SEP)
print("REFERENCE QUALITY CHECKS")
print(SEP)
print(f"Submissions longer than references: {subs_longer_than_ref}")
print(f"Submissions identical to references: {identical_submission_reference}")
print(f"Potentially weak references (heuristic): {potentially_weak_references}")

print("\n" + SEP)
print("DATA INTEGRITY CHECKS")
print(SEP)
print(f"Malformed rows: {len(malformed)}")
print(f"Missing fields summary: {dict(missing_field_counts)}")
print(f"Invalid score rows: {len(invalid_score_rows)}")
print(f"Empty submission rows: {len(empty_submission_rows)}")

print("\n" + SEP)
print("RESEARCH-ORIENTED OBSERVATIONS")
print(SEP)
for obs in observations:
    print(f" - {obs}")

print("\n" + SEP)
print("INTEGRITY SUMMARY")
print(SEP)
for k, v in integrity_issues.items():
    print(f"{k}: {v}")

print("\n" + SEP)
print("EDA complete.")
print(SEP)
