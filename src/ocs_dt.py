# %%
import pandas as pd
from common import load_data
from rich.console import Console
from cpmpy_dt import CPMpyDecisionTree

# %%

system = "gcc"


# %%

(
    perf_matrix_initial,
    _,
    _,
    all_performances,
    _,
    _,
) = load_data(system=system, data_dir="./data", input_properties_type="tabular")

results = []
console = Console()
scaling_factor = 10_000

# for num_performances in range(1, len(all_performances) + 1):
num_performances = 1
performances = all_performances[:num_performances]

print(f"{system}: Using {len(performances)} performances: {performances}")

# Normalize performance metrics
nmdf = (
    perf_matrix_initial[["inputname"] + performances]
    .groupby("inputname", as_index=True)
    .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
)
nmdf["worst_case_performance"] = nmdf[performances].max(axis=1)

# Prepare performance matrix
perf_matrix = pd.merge(
    perf_matrix_initial,
    nmdf,
    suffixes=("_raw", None),
    left_index=True,
    right_index=True,
)

# Create configuration-input performance matrix
cip = perf_matrix[
    ["configurationID", "inputname", "worst_case_performance"]
].pivot(
    index="configurationID",
    columns="inputname",
    values="worst_case_performance",
)

# %%

