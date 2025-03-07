# %%
import pandas as pd
import plotnine as p9
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import numpy as np

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
# combined_df = combined_df[combined_df["system"] != "sqlite"]
combined_df = combined_df[combined_df["num_performances"] <= 4]  # only affects sqlite
# combined_df["num_performances"] = combined_df["num_performances"].astype(str)
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
max_configs_by_group = (
    combined_df[["system", "num_performances", "num_configs"]]
    .groupby(["system", "num_performances"])
    .max()
)

# Group by just system to get overall max configs per system
max_configs_by_system = combined_df[["system", "num_configs"]].groupby("system").max()

# Assert that for each system, all num_performances groups have the same max num_configs
for system in max_configs_by_system.index:
    system_max = max_configs_by_system.loc[system, "num_configs"]
    system_group_maxes = max_configs_by_group.loc[system]["num_configs"]
    assert all(system_group_maxes == system_max), (
        f"System {system} has inconsistent max configs across num_performances"
    )

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
gccdf[["system", "num_configs", "wcp_mean_normalized"]].groupby(
    ["system", "num_configs"], as_index=False
).mean().plot(x="num_configs", y="wcp_mean_normalized")
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

import matplotlib.cm as cm
from matplotlib.lines import Line2D

# Create a plot with wcp_max and wcp_mean as separate markers for each system + configuration
# Filter for optimization_target="mean" only


def plot_wcp_mean_max_by_system(
    combined_df, optimization_target="mean", plot_title=False
):
    mean_opt_df = (
        combined_df[combined_df["optimization_target"] == optimization_target]
        .groupby(["system", "num_configs"], as_index=False)
        .min()
    )
    performance_numbers = list(map(str, sorted(mean_opt_df["num_configs"].unique())))
    print(performance_numbers)
    mean_opt_df["num_configs"] = mean_opt_df["num_configs"].astype(str)

    # Create long format data for plotting wcp_max and wcp_mean as separate points
    long_format_df = pd.melt(
        mean_opt_df,
        id_vars=["system", "num_configs"],
        value_vars=["wcp_mean", "wcp_max"],
        var_name="metric",
        value_name="value",
    )  # .sort_values(by=["system", "num_configs", "metric"])

    # Add a config_id column to use for dodging (unique identifier for each configuration)
    long_format_df["config_id"] = long_format_df["num_configs"]

    # Create a unique group identifier for each system + config_id combination
    long_format_df["pair_id"] = (
        long_format_df["system"] + "_" + long_format_df["config_id"]
    )

    # Calculate the gap between wcp_max and wcp_mean for each configuration
    gap_df = long_format_df.pivot_table(
        index=["system", "config_id", "pair_id"], columns="metric", values="value"
    ).reset_index()
    gap_df["gap"] = gap_df["wcp_max"] - gap_df["wcp_mean"]

    # Merge the gap information back to the original dataframe
    long_format_df = long_format_df.merge(
        gap_df[["pair_id", "gap"]], on="pair_id", how="left"
    )

    # Get unique systems and performance numbers for plotting
    systems = long_format_df["system"].unique()
    # performance_numbers = list(map(str, sorted(long_format_df["num_configs"].unique())))

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define markers for the metrics
    markers = {"wcp_mean": "x", "wcp_max": 7}

    # Define a color palette similar to the one used in plotnine
    colors = cm.Set2.colors[: len(performance_numbers)]
    color_map = dict(zip(performance_numbers, colors))
    print(color_map)

    # Calculate the width for dodging
    dodge_width = 0.7
    bar_width = dodge_width / len(performance_numbers)
    offsets = np.linspace(
        -dodge_width / 2 + bar_width / 2,
        dodge_width / 2 - bar_width / 2,
        len(performance_numbers),
    )
    offset_map = dict(zip(performance_numbers, offsets))
    print(offset_map)

    # Plot each system and performance combination
    for system_idx, system in enumerate(systems):
        system_data = long_format_df[long_format_df["system"] == system]

        for perf in performance_numbers:
            perf_data = system_data[system_data["num_configs"] == perf]

            if len(perf_data) == 0:
                continue

            # Get the x position with dodge offset
            x_pos = system_idx + offset_map[perf]

            # Extract wcp_mean and wcp_max for this system and performance
            mean_data = perf_data[perf_data["metric"] == "wcp_mean"]
            max_data = perf_data[perf_data["metric"] == "wcp_max"]

            if len(mean_data) > 0 and len(max_data) > 0:
                mean_value = mean_data["value"].values[0]
                max_value = max_data["value"].values[0]
                gap = max_value - mean_value

                # Plot the connecting line with alpha based on gap
                alpha = 0.3 + (0.6 * (gap / gap_df["gap"].max()))
                ax.plot(
                    [x_pos, x_pos],
                    [mean_value, max_value],
                    # color=color_map[perf],
                    alpha=alpha,
                    linewidth=0.8,
                )

                # Plot the points
                ax.scatter(
                    x_pos,
                    mean_value,
                    marker=markers["wcp_mean"],
                    color="black",
                    s=25,
                    linewidth=0.5,
                )
                ax.scatter(
                    x_pos,
                    max_value,
                    marker=markers["wcp_max"],
                    color="black",
                    s=25,
                    linewidth=0.5,
                )

    # Set the x-ticks to the system names
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels(systems)

    # Add grid lines
    ax.yaxis.grid(True, linestyle="-", alpha=0.5)
    ax.set_axisbelow(True)

    # Add labels and title
    ax.set_xlabel("System")
    ax.set_ylabel("WCP Value")
    if plot_title:
        ax.set_title(
            f"WCP Mean and Max by System\nOptimization Target: {optimization_target}",
            fontweight="bold",
        )

    # Create custom legend elements
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="v",
            color="black",
            label="Max WCP",
            markerfacecolor="black",
            markersize=8,
            linestyle="",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            label="Mean WCP",
            markerfacecolor="black",
            markersize=8,
            linestyle="",
        ),
    ]

    # Add performance colors to legend
    # for perf in performance_numbers:
    #     legend_elements.append(
    #         Line2D([0], [0], color=color_map[perf], label=f"|P|={perf}", linewidth=2)
    #     )

    # Add the legend
    ax.legend(handles=legend_elements, loc="best")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        f"../results/wcp_spread_target_{optimization_target}.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


# plot_wcp_mean_max_by_system(combined_df, optimization_target="mean", plot_title=True)


plot_wcp_mean_max_by_system(combined_df, optimization_target="max", plot_title=True)

# %%

# Plot WCP max and mean with different line styles and colors by number of performances
ccdf = combined_df[combined_df["optimization_target"] == "both"].copy()
ccdf["num_performances"] = ccdf["num_performances"].astype(str)

# Create separate plots for max and mean WCP
p = (
    p9.ggplot(ccdf)
    +
    # WCP max line (solid)
    p9.geom_line(
        p9.aes(
            x="num_configs",
            y="wcp_max",
            color="num_performances",
            group="num_performances",
            # linetype="WCP Type"
        ),
        linetype="solid",
    )
    +
    # WCP mean line (dashed)
    p9.geom_line(
        p9.aes(
            x="num_configs",
            y="wcp_mean",
            color="num_performances",
            group="num_performances",
            # linetype="WCP Type"
        ),
        linetype="dashed",
    )
    + p9.theme_minimal()
    + p9.facet_wrap("~system", scales="free", nrow=2)
    + p9.labs(
        title="WCP Max and Mean vs Number of Configurations",
        x="Number of Configurations",
        y="WCP Value",
        color="Number of\nPerformances",
    )
    + p9.scale_linetype_manual(
        name="WCP Type", values=["solid", "dashed"], labels=["WCP Max", "WCP Mean"]
    )
)
p
# %%
combined_df.num_performances.max()
# %%

# %%
# Create matplotlib version of the same plot
print("\nCreating matplotlib version of the plot...")

# Filter data for optimization_target="both"
ccdf = combined_df[combined_df["optimization_target"] == "both"].copy()
ccdf["num_performances"] = ccdf["num_performances"].astype(str)

# Get unique systems and performance numbers
systems = ccdf["system"].unique()
performance_numbers = sorted(ccdf["num_performances"].unique())

# Calculate number of rows and columns for subplots
num_cols = 4
num_rows = (len(systems) + num_cols - 1) // num_cols  # Ceiling division

# Create figure and subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 3 * num_rows))
axes = axes.flatten()

# Define colors for different performance numbers
colors = plt.cm.Set2(np.linspace(0, 1, len(performance_numbers)))
color_map = dict(zip(performance_numbers, colors))

# Create custom legend handles
performance_handles = []
for perf in performance_numbers:
    performance_handles.append(
        plt.Line2D([0], [0], color=color_map[perf], linestyle="-", label=f"|P|={perf}")
    )

linestyle_handles = [
    plt.Line2D([0], [0], color="gray", linestyle="-", label="WCP Max"),
    plt.Line2D([0], [0], color="gray", linestyle="--", label="WCP Mean"),
]

# Plot for each system
for idx, system in enumerate(systems):
    system_df = ccdf[ccdf["system"] == system]
    ax = axes[idx]
    
    for perf in performance_numbers:
        perf_data = system_df[system_df["num_performances"] == perf]
        
        # Plot WCP max (solid line)
        ax.plot(
            perf_data["num_configs"],
            perf_data["wcp_max"],
            color=color_map[perf],
            linestyle="solid",
        )
        
        # Plot WCP mean (dashed line)
        ax.plot(
            perf_data["num_configs"],
            perf_data["wcp_mean"],
            color=color_map[perf],
            linestyle="dashed",
        )
    
    ax.set_title(f"System: {system} (#P={system_df['num_performances'].max()})")
    
    if idx >= num_cols:
        ax.set_xlabel("Number of Configurations")
    else:
        ax.set_xlabel("")
    
    # Only show y-axis label for plots in the first column
    if idx % num_cols == 0:
        ax.set_ylabel("WCP Value")
    else:
        ax.set_ylabel("")
    
    # Set integer ticks on x-axis with appropriate spacing
    x_min, x_max = ax.get_xlim()
    x_min = max(1, int(x_min))
    x_max = int(x_max) + 1
    
    # Determine appropriate tick spacing based on range
    x_range = x_max - x_min
    if x_range <= 10:
        step = 1
    elif x_range <= 20:
        step = 2
    elif x_range <= 50:
        step = 5
    else:
        step = 10
        
    ax.set_xticks(range(x_min, x_max, step))
    ax.set_xticklabels([str(x) for x in range(x_min, x_max, step)])
    
    ax.grid(True, linestyle="--", alpha=0.7)

# Remove empty subplots if any
if len(systems) < len(axes):
    for idx in range(len(systems), len(axes)):
        fig.delaxes(axes[idx])

# Add two legends to the first subplot
# Place both legends inside the plot at the bottom in a single row
legend1 = axes[5].legend(
    handles=performance_handles,
    title="#Performances",
    loc="upper right",
    # bbox_to_anchor=(0.0, 0.0),
    # ncol=len(performance_numbers),
    frameon=True,
    framealpha=0.8,
)

axes[5].add_artist(legend1)

legend2 = axes[5].legend(
    handles=linestyle_handles,
    title="WCP Type",
    loc="upper center",
    bbox_to_anchor=(0.35, 1.0),
    # ncol=2,
    frameon=True,
    framealpha=0.8,
)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the plot
plt.savefig(
    "../results/rq1_wcp_max_mean.pdf", dpi=300, bbox_inches="tight"
)

# Display the plot
plt.show()

# %%
