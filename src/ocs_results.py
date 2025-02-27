# %%
import pandas as pd
import plotnine as p9
from pathlib import Path
import glob
import matplotlib.pyplot as plt

# %%
# Read all ocs_ files from results directory
results_dir = Path("../results")
# all_files = glob.glob(str(results_dir / "ocs_*.csv"))
# All except _cv files
all_files = list(
    filter(lambda s: "_cv.csv" not in s, glob.glob(str(results_dir / "ocs_*.csv")))
)

dfs = []
for file in sorted(all_files):
    df = pd.read_csv(file)
    # Extract system and performance metrics from filename
    filename = Path(file).stem
    _, system, opt = filename.split("_", maxsplit=2)
    df["system"] = system
    df["optimization_target"] = opt
    df["wcp_mean_normalized"] = df["wcp_mean"] / df["wcp_mean"].max()
    df["wcp_max_normalized"] = df["wcp_max"] / df["wcp_max"].max()
    df["wcp_gap"] = df["wcp_max"] - df["wcp_mean"]
    df["wcp_gap_normalized"] = df["wcp_gap"] / df["wcp_gap"].max()
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
combined_df = combined_df[combined_df["system"] != "sqlite"]
combined_df["num_performances"] = combined_df["num_performances"].astype(str)
print(f"Read data; shape: {combined_df.shape}")

# %%

## Impute missing values
groups = combined_df[["system", "num_performances"]].drop_duplicates()

new_data = []

for idx, row in groups.iterrows():
    max_cfg_in_group = combined_df[
        (combined_df["system"] == row["system"])
        & (combined_df["num_performances"] == row["num_performances"])
    ]["num_configs"].max()

    max_cfg_for_system = combined_df[combined_df["system"] == row["system"]][
        "num_configs"
    ].max()

    print(row["system"], row["num_performances"], max_cfg_in_group, max_cfg_for_system)

    if max_cfg_in_group < max_cfg_for_system:
        last_valid_value = combined_df[
            (combined_df["system"] == row["system"])
            & (combined_df["num_performances"] == row["num_performances"])
            & (combined_df["num_configs"] == max_cfg_in_group)
        ]

        # Repeat last row for all new configurations
        for new_cfg in range(max_cfg_in_group + 1, max_cfg_for_system + 1):
            new_row = last_valid_value.copy()
            new_row["num_configs"] = new_cfg
            new_data.append(new_row)

# 3. Concatenate new data with original data
new_data_df = pd.concat(new_data)
combined_df = (
    pd.concat([combined_df, new_data_df])
    .sort_values(by=["system", "num_performances", "num_configs"])
    .reset_index(drop=True)
)
print(f"Imputed {len(new_data)} rows; new shape: {combined_df.shape}")

# %%
# Group by system and num_performances to get max configs
max_configs_by_group = combined_df[["system", "num_performances", "num_configs"]].groupby(["system", "num_performances"]).max()

# Group by just system to get overall max configs per system
max_configs_by_system = combined_df[["system", "num_configs"]].groupby("system").max()

# Assert that for each system, all num_performances groups have the same max num_configs
for system in max_configs_by_system.index:
    system_max = max_configs_by_system.loc[system, "num_configs"]
    system_group_maxes = max_configs_by_group.loc[system]["num_configs"]
    assert all(system_group_maxes == system_max), f"System {system} has inconsistent max configs across num_performances"

print("All systems have consistent max num_configs across num_performances groups")
print("\nMax configs by system and num_performances:")
print(max_configs_by_group)

# %%

p = (
    p9.ggplot(
        combined_df,
        p9.aes(
            x="num_configs",
            y="wcp_mean_normalized",
            color="num_performances",
            group="num_performances",
        ),
    )
    + p9.geom_line()
    + p9.theme_minimal()
    + p9.scale_y_continuous(limits=[0, None])
    + p9.scale_x_continuous(
        breaks=[1] + list(range(5, combined_df["num_configs"].max() + 1, 5))
    )
    + p9.facet_wrap("~system", scales="free_x", nrow=2)
    + p9.labs(
        title="WCP Mean vs Number of Configurations",
        x="Number of Configurations",
        y="Normalized WCP Mean",
        color="Number of\nPerformances",
    )
)
p


# %%
# Reshape data to long format for plotting both metrics
plot_df = pd.melt(
    combined_df,
    id_vars=["num_configs", "system", "optimization_target", "num_performances"],
    value_vars=["wcp_mean_normalized", "wcp_max_normalized"],
    var_name="metric",
    value_name="value",
)

p = (
    p9.ggplot(
        plot_df,
        p9.aes(
            x="num_configs",
            y="value",
            color="num_performances",
            linetype="metric",
            group=p9.aes("metric"),
        ),
    )
    + p9.geom_line()
    + p9.theme_minimal()
    + p9.scale_y_continuous(limits=[0, None])
    + p9.scale_x_continuous(
        breaks=[1] + list(range(5, combined_df["num_configs"].max() + 1, 5))
    )
    + p9.facet_wrap("~system", scales="free_x", nrow=2)
    + p9.labs(
        title="WCP Metrics vs Number of Configurations",
        x="Number of Configurations",
        y="Normalized Value",
        color="Metric",
        linetype="Optimization Target",
    )
)
p

# %%
p = (
    p9.ggplot(
        combined_df[combined_df["system"] != "sqlite"],
        p9.aes(
            x="num_configs",
            y="wcp_mean",
            color="num_performances",
            group="num_performances",
        ),
    )
    + p9.geom_line()
    + p9.theme_minimal()
    + p9.scale_y_continuous(limits=[0, None])
    # + p9.scale_x_continuous(
    #     breaks=[1] + list(range(5, combined_df["num_configs"].max() + 1, 5))
    # )
    # + p9.scale_color_cmap(cmap_name="tab20")
    + p9.facet_wrap("~system", scales="free", nrow=2)
    + p9.labs(
        title="WCP Mean vs Number of Configurations",
        x="Number of Configurations",
        y="Normalized WCP Mean",
        color="Number of\nPerformances",
    )
)
p
# p.save("wcp_mean_vs_configs.pdf", height=6, width=10)


# %%
combined_df.head()

# %%

gccdf = combined_df[combined_df.system == "gcc"].drop(
    columns=["performances", "selected_configs", "optimization_target"]
)
# %%
gccdf.groupby(["system", "num_configs"], as_index=False).mean().plot(
    x="num_configs", y="wcp_mean_normalized"
)
plt.show()

# %%

# Here we plot the gap in WCP_max and WCP_mean when optimizing for WCP_mean
ccdf = combined_df[
    (combined_df["system"] != "sqlite")
]  # & (combined_df["optimization_target"] == "mean")]
ccdf["num_performances"] = ccdf["num_performances"].astype(str)
p = (
    p9.ggplot(
        ccdf,
        p9.aes(
            x="num_configs",
            y="wcp_gap",
            color="num_performances",
            linetype="optimization_target",
            # color="optimization_target",
            # group="optimization_target",
        ),
    )
    + p9.geom_line()
    + p9.theme_minimal()
    # + p9.scale_y_discrete()
    + p9.facet_wrap("~system", scales="free", nrow=2)
    + p9.labs(
        title="WCP Gap vs Number of Configurations by Optimization Target",
        x="Number of Configurations",
        y="WCP Gap (WCP_max - WCP_mean)",
        color="Number of\nPerformances",
    )
)
p


# %%

# Plot WCP gap vs num configs for each system
systems = combined_df["system"].unique()
num_systems = len(systems)

# Calculate number of rows and columns needed
num_cols = 2
num_rows = (num_systems + num_cols - 1) // num_cols  # Ceiling division

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))
axes = axes.flatten()

for idx, system in enumerate(systems):
    system_df = combined_df[combined_df["system"] == system]

    for num_perf in sorted(system_df["num_performances"].unique()):
        perf_data = system_df[system_df["num_performances"] == num_perf]
        axes[idx].plot(
            perf_data["num_configs"],
            perf_data["wcp_gap"],
            label=str(num_perf),
            marker="o",
            markersize=3,
        )

    axes[idx].set_title(f"System: {system}")
    axes[idx].set_xlabel("Number of Configurations")
    axes[idx].set_ylabel("WCP Gap")
    axes[idx].grid(True, linestyle="--", alpha=0.7)
    axes[idx].legend(title="Number of\nPerformances")

# Remove empty subplots
if num_systems < len(axes):
    for idx in range(num_systems, len(axes)):
        fig.delaxes(axes[idx])

plt.tight_layout()
plt.show()

# %%
