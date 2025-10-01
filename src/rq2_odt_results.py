# %%
import glob
import json
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
from matplotlib.lines import Line2D

# %%
# Read all ocs_ files from results directory
results_dir = Path("../results")
all_files = results_dir.glob("odt_*.json")

results = []
for file in all_files:
    results.extend(json.load(file.open("r")))

# combined_df = pd.concat(dfs, ignore_index=True)
df = pd.DataFrame(results)
# combined_df = combined_df[combined_df["system"] != "sqlite"]
# combined_df = combined_df[combined_df["num_performances"] <= 4]  # only affects sqlite
# combined_df["num_performances"] = combined_df["num_performances"].astype(str)
print(f"Read data; shape: {df.shape}")

df["performances"] = df["performances"].apply(lambda s: "_".join(s))

full_data = df[df.split.isna()]
split_data = df[~df.split.isna()]

# %%

# Aggregate over all performance metrics
# Plot: per system, x: num_configs, y: max_depth/num rules
# Table: (system, num_configs, max_depth/num rules)

# Average max depth over all performances for full dataset
full_data[["system", "num_configs", "performances", "max_depth"]].groupby(
    ["system", "performances", "num_configs"]
).max().groupby(["system", "num_configs"]).mean()

# %%

split_data[["system", "num_configs", "performances", "max_depth"]].groupby(
    ["system", "performances", "num_configs"]
).max().groupby(["system", "num_configs"]).mean()


# %%

# How to quantify error in a single number?
# It's just the WCP for the predictions

# Plot 1: x: depth / y: wcp
# Plot 2: x: 
