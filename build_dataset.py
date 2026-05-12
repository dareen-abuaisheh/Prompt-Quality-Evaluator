"""
Dolly 15k CSV Exporter
----------------------

This script downloads the Databricks Dolly 15k dataset
from Hugging Face and saves it locally as a CSV file.

Output:
    data/dolly_dataset.csv
"""

# =========================================================
# Imports
# =========================================================

from datasets import load_dataset
import pandas as pd


# =========================================================
# Load Dataset
# =========================================================

print("Loading Dolly 15k dataset...\n")

dataset = load_dataset("databricks/databricks-dolly-15k")

# Access training split
train_data = dataset["train"]


# =========================================================
# Convert Dataset to Pandas DataFrame
# =========================================================

print("Converting dataset to DataFrame...\n")

df = pd.DataFrame(train_data)


# =========================================================
# Save Dataset as CSV
# =========================================================

output_path = "data/dolly_dataset.csv"

print("Saving dataset as CSV...\n")

df.to_csv(output_path, index=False)


# =========================================================
# Final Output
# =========================================================

print("=" * 60)
print("Dataset exported successfully!")
print(f"Saved to: {output_path}")
print(f"Total samples: {len(df)}")
print("=" * 60)