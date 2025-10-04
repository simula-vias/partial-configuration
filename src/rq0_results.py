# %%
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from glob import glob

# %%

files = glob("./results/wcp_1cfg_*.json")

dfs = [pd.read_json(f) for f in files]
df = pd.concat(dfs)

print(df.columns)

df.sample(6)




# %%
sdc_max_better = (df["sdc_max"] < df["overall_max"]).mean()
sdc_max_better_by = (df["sdc_max"] - df["overall_max"]).mean()

print(f"SDC max better: {sdc_max_better:.2%} (by {sdc_max_better_by:.2})")

sdc_avg_better = (df["sdc_avg"] < df["overall_avg"]).mean()
sdc_avg_better_by = (df["sdc_avg"] - df["overall_avg"]).mean()

print(f"SDC avg better: {sdc_avg_better:.2%} (by {sdc_avg_better_by:.2})")

aggregate_over_all_systems = (
    df[
        [
            "sdc_avg",
            "sdc_max",
            "overall_avg",
            "overall_max",
            "average_avg",
            "average_max",
        ]
    ]
    .agg(["mean", "std"])
    .round(3)
)

print(aggregate_over_all_systems)

wilcoxon_avg = stats.wilcoxon(df["sdc_avg"], df["overall_avg"], alternative="less")
ttest_avg = stats.ttest_ind(df["sdc_avg"], df["overall_avg"])
wilcoxon_max = stats.wilcoxon(df["sdc_max"], df["overall_max"], alternative="less")
ttest_max = stats.ttest_ind(df["sdc_max"], df["overall_max"])

print(f"Wilcoxon test (Avg. WCP SDC/Overall): {wilcoxon_avg.pvalue:.1e}")
print(f"T-test (Avg. WCP SDC/Overall): {ttest_avg.pvalue:.1e}")

print(f"Wilcoxon test (Max. WCP SDC/Overall): {wilcoxon_max.pvalue:.1e}")
print(f"T-test (Max. WCP SDC/Overall): {ttest_max.pvalue:.1e}")

# %%
# RQ0: This is currently the correct plot.
# Create plots for each system with aggregated results by num_performances
max_num_performances = 4

def create_system_plots():
    # Get unique systems
    systems = df['system'].unique()
    
    # Fixed layout with two rows
    n_cols = len(systems) // 2 + len(systems) % 2  # Ceiling division
    n_rows = 2
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 8), sharex=True)
    axes = axes.flatten()
    
    # For each system, create a subplot
    for i, system in enumerate(systems):
        system_data = df[df['system'] == system]
        
        # Filter to only include num_performances <= max_num_performances
        system_data = system_data[system_data['num_performances'] <= max_num_performances]
        
        # Group by num_performances and calculate mean
        grouped = system_data.groupby('num_performances').agg({
            'sdc_max': 'mean',
            'sdc_avg': 'mean',
            'overall_max': 'mean',
            'overall_avg': 'mean',
            'num_predictable_configs': 'mean'
        }).reset_index()
        
        # Plot data
        ax = axes[i]
        
        # Create a secondary y-axis for num_predictable_configs
        ax2 = ax.twinx()
        
        # Offset for SDC and Overall points to prevent overlap
        offset = 0.1
        
        for j, num_perf in enumerate(grouped['num_performances']):
            # SDC points (slightly to the left)
            sdc_x = num_perf - offset
            ax.plot(sdc_x, grouped.loc[j, 'sdc_max'], 'bv', linestyle='', label='SDC Max' if j == 0 else "")
            ax.plot(sdc_x, grouped.loc[j, 'sdc_avg'], 'b_', linestyle='', label='SDC Avg' if j == 0 else "")
            
            # Connect SDC max and avg with vertical line
            ax.plot([sdc_x, sdc_x], 
                   [grouped.loc[j, 'sdc_avg'], grouped.loc[j, 'sdc_max']], 
                   'b-', alpha=0.5)
            
            # Overall points (slightly to the right)
            overall_x = num_perf + offset
            ax.plot(overall_x, grouped.loc[j, 'overall_max'], 'rv', linestyle='', label='Overall Max' if j == 0 else "")
            ax.plot(overall_x, grouped.loc[j, 'overall_avg'], 'r_', linestyle='', label='Overall Avg' if j == 0 else "")
            
            # Connect Overall max and avg with vertical line
            ax.plot([overall_x, overall_x], 
                   [grouped.loc[j, 'overall_avg'], grouped.loc[j, 'overall_max']], 
                   'r-', alpha=0.5)
            
            # Plot num_predictable_configs on secondary y-axis with stars
            ax2.plot(num_perf, grouped.loc[j, 'num_predictable_configs'], 'g*', markersize=10, 
                    label='Predictable Configs' if j == 0 else "")
        
        ax.set_title(f'System: {system}')
        ax.set_xlabel('Number of Performances')
        ax.set_ylabel('WCP')
        ax2.set_ylabel('Number of Predictable Configs', color='g')
        
        # Add grid back
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-ticks to only integers
        ax.set_xticks(list(range(1, max_num_performances+1)))
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0, top=1)
        ax2.set_ylim(bottom=0)
        
        # Set the tick parameters for the second y-axis to be green
        ax2.tick_params(axis='y', colors='green')
        
        # Add legend to the first subplot only
        if i == 0:
            # Combine legends from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('../results/rq0_wcp_by_system_performances.png', dpi=300, bbox_inches='tight')
    plt.show()

# Call the function to create the plots
create_system_plots()

# %%
# RQ0: This is an alternative representation, but to be meaningful
# we should collect the data in a better format.
# Create a visualization with merged boxplots
def create_merged_boxplot_system_plots():
    # Get unique systems
    systems = df['system'].unique()
    
    # Fixed layout with two rows
    n_cols = len(systems) // 2 + len(systems) % 2  # Ceiling division
    n_rows = 2
    
    # Create figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 8))
    axes = axes.flatten()
    
    # For each system, create a subplot
    for i, system in enumerate(systems):
        system_data = df[df['system'] == system]
        
        # Filter to only include num_performances <= max_num_performances
        system_data = system_data[system_data['num_performances'] <= max_num_performances]
        
        # Group by num_performances
        grouped = system_data.groupby('num_performances')
        
        # Plot data
        ax = axes[i]
        
        # Create a secondary y-axis for num_predictable_configs
        ax2 = ax.twinx()
        
        # Prepare data for boxplots
        positions = []
        boxplot_data = []
        labels = []
        colors = []
        
        # For each num_performances value
        for num_perf in sorted(system_data['num_performances'].unique()):
            perf_data = grouped.get_group(num_perf)
            
            # Add SDC data (combining max and avg)
            positions.append(num_perf - 0.15)  # Slightly to the left
            # Create a combined array of max and avg values
            sdc_combined = np.concatenate([perf_data['sdc_max'], perf_data['sdc_avg']])
            boxplot_data.append(sdc_combined)
            labels.append('SDC' if num_perf == 1 else '')
            colors.append('blue')
            
            # Add Overall data (combining max and avg)
            positions.append(num_perf + 0.15)  # Slightly to the right
            # Create a combined array of max and avg values
            overall_combined = np.concatenate([perf_data['overall_max'], perf_data['overall_avg']])
            boxplot_data.append(overall_combined)
            labels.append('Overall' if num_perf == 1 else '')
            colors.append('red')
            
            # Plot num_predictable_configs on secondary y-axis with stars
            mean_configs = perf_data['num_predictable_configs'].mean()
            std_configs = perf_data['num_predictable_configs'].std()
            ax2.errorbar(num_perf, mean_configs, yerr=std_configs, 
                        fmt='g*', markersize=10, capsize=5,
                        label='Predictable Configs' if num_perf == 1 else "")
        
        # Create boxplots
        bplot = ax.boxplot(boxplot_data, positions=positions, patch_artist=True, 
                          widths=0.25, showfliers=False)
        
        # Color the boxplots
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Set title and labels
        ax.set_title(f'System: {system}')
        ax.set_xlabel('Number of Performances')
        ax.set_ylabel('WCP')
        ax2.set_ylabel('Number of Predictable Configs', color='g')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set x-ticks to only integers
        ax.set_xticks(list(range(1, max_num_performances+1)))
        
        # Set y-axis limits
        ax.set_ylim(bottom=0, top=1)
        ax2.set_ylim(bottom=0)
        
        # Set the tick parameters for the second y-axis to be green
        ax2.tick_params(axis='y', colors='green')
        
        # Add legend to the first subplot only
        if i == 0:
            # Create custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='blue', alpha=0.7, label='SDC (Max & Avg)'),
                Patch(facecolor='red', alpha=0.7, label='Overall (Max & Avg)'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='g', 
                          markersize=10, label='Predictable Configs')
            ]
            ax.legend(handles=legend_elements, loc='best')
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('../results/rq0_wcp_by_system_merged_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()

# Call the function to create the merged boxplot visualization
create_merged_boxplot_system_plots()

# %%
# RQ0: Create publication-quality plots
def create_publication_plots():
    # Get unique systems
    systems = sorted(df['system'].unique())
    
    # Fixed layout with two rows
    n_cols = len(systems) // 2 + len(systems) % 2  # Ceiling division
    n_rows = 2
    
    # Set up figure with better aesthetics for publication
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300
    })
    
    # Create figure and subplots with better proportions
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 5), sharex=True)
    axes = axes.flatten()
    
    # For each system, create a subplot
    for i, system in enumerate(systems):
        system_data = df[df['system'] == system]
        
        # Filter to only include num_performances <= max_num_performances
        system_data = system_data[system_data['num_performances'] <= max_num_performances]
        
        # Group by num_performances and calculate mean
        grouped = system_data.groupby('num_performances').agg({
            'sdc_max': 'mean',
            'sdc_avg': 'mean',
            'overall_max': 'mean',
            'overall_avg': 'mean',
            'num_predictable_configs': 'mean'
        }).reset_index()
        
        # Plot data
        ax = axes[i]
        
        # Create a secondary y-axis for num_predictable_configs
        ax2 = ax.twinx()
        
        # Offset for SDC and Overall points to prevent overlap
        offset = 0.1
        
        # Use more professional colors
        sdc_color = '#1f77b4'  # Blue
        overall_color = '#d62728'  # Red
        config_color = '#2ca02c'  # Green
        
        for j, num_perf in enumerate(grouped['num_performances']):
            # SDC points (slightly to the left)
            sdc_x = num_perf - offset
            ax.plot(sdc_x, grouped.loc[j, 'sdc_max'], marker='v', color=sdc_color, 
                   linestyle='', markersize=6, label='SDC Max' if j == 0 else "")
            ax.plot(sdc_x, grouped.loc[j, 'sdc_avg'], marker='_', color=sdc_color, 
                   linestyle='', markersize=8, label='SDC Avg' if j == 0 else "")
            
            # Connect SDC max and avg with vertical line
            ax.plot([sdc_x, sdc_x], 
                   [grouped.loc[j, 'sdc_avg'], grouped.loc[j, 'sdc_max']], 
                   '-', color=sdc_color, alpha=0.6, linewidth=1.5)
            
            # Overall points (slightly to the right)
            overall_x = num_perf + offset
            ax.plot(overall_x, grouped.loc[j, 'overall_max'], marker='v', color=overall_color, 
                   linestyle='', markersize=6, label='Overall Max' if j == 0 else "")
            ax.plot(overall_x, grouped.loc[j, 'overall_avg'], marker='_', color=overall_color, 
                   linestyle='', markersize=8, label='Overall Avg' if j == 0 else "")
            
            # Connect Overall max and avg with vertical line
            ax.plot([overall_x, overall_x], 
                   [grouped.loc[j, 'overall_avg'], grouped.loc[j, 'overall_max']], 
                   '-', color=overall_color, alpha=0.6, linewidth=1.5)
            
            # Plot num_predictable_configs on secondary y-axis with stars
            ax2.plot(num_perf, grouped.loc[j, 'num_predictable_configs'], marker='*', 
                    color=sdc_color, markersize=8, linestyle='',
                    label='SDC #Configs' if j == 0 else "")
        
        # Improve title and labels
        ax.set_title(f'System: {system}', pad=8)
        if i >= n_cols * (n_rows - 1):  # Only bottom row gets x-label
            ax.set_xlabel('Number of Performances')
        if i % n_cols == 0:  # Only leftmost plots get y-label
            ax.set_ylabel('Worst-Case Performance')
        
        # Set secondary y-axis label only for rightmost plots
        if (i + 1) % n_cols == 0 or i == len(systems) - 1:
            ax2.set_ylabel('#Configs') #, color=sdc_color)
        
        # Add subtle grid for readability
        ax.grid(True, linestyle=':', alpha=0.4, color='gray')
        
        # Set x-ticks to only integers
        ax.set_xticks(list(range(1, max_num_performances+1)))
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0, top=1)
        ax2.set_ylim(bottom=0)
        
        # Set the tick parameters for the second y-axis to be green
        # ax2.tick_params(axis='y')
        
        # Add spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
        
        # Add legend to the first subplot only
        if i == 3:
            # Combine legends from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            legend = ax.legend(lines1 + lines2, labels1 + labels2, 
                              loc='upper right', framealpha=0.9, 
                              edgecolor='gray', fancybox=False)
            legend.get_frame().set_linewidth(0.8)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout(pad=1.2, h_pad=1.5, w_pad=1.5)
    plt.savefig('../results/rq0_publication_quality.pdf', bbox_inches='tight')
    plt.savefig('../results/rq0_publication_quality.png', dpi=300, bbox_inches='tight')
    plt.show()

# Call the function to create publication-quality plots
create_publication_plots()


# %%
df["num_performances"] = df.performances.str.split("-").apply(len)
# %%
df["performances"].unique().shape
# %%
df["num_performances"].unique().shape
# %%

# We want a table that shows standard SDC is not super good.
# However, what is SDC in this case?

df[["system", "num_performances", "num_leaf_nodes", "sdc_max", "sdc_avg", "overall_max", "overall_avg", "average_max", "average_avg", "common_max", "common_avg" ]].groupby(["system", "num_performances"]).mean()
# %%
