import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Get all plot files
plot_dir = Path("plots")
plot_files = sorted(glob.glob(str(plot_dir / "decision_tree_overfitting_*.png")))

# Calculate grid dimensions
n_plots = len(plot_files)
n_rows = 2
n_cols = (n_plots + n_rows - 1) // n_rows


# Create figure with subplots
fig = plt.figure(figsize=(15, 6))

# Add each plot as a subplot
for i, plot_file in enumerate(plot_files):
    img = plt.imread(plot_file)
    ax = fig.add_subplot(n_rows, n_cols, i+1)
    ax.imshow(img)
    ax.axis('off')
    
    # Extract system name from filename for title
    # system = Path(plot_file).stem.replace('decision_tree_overfitting_', '')
    # ax.set_title(system)

plt.tight_layout()  # Reduce padding between subplots
plt.savefig('decision_tree_overfitting.png', bbox_inches='tight', dpi=400)
plt.close()
