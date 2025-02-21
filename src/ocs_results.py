# %%
import pandas as pd
import plotnine as p9
from pathlib import Path
import glob

# %%
# Read all ocs_ files from results directory
results_dir = Path("../results")
all_files = glob.glob(str(results_dir / "ocs_*_mean_ortools.csv"))

dfs = []
for file in all_files:
    df = pd.read_csv(file)
    # Extract system and performance metrics from filename
    filename = Path(file).stem
    _, system, opt, perfs = filename.split("_", maxsplit=3)
    df["system"] = system
    df["metrics"] = perfs
    df["wcp_mean_normalized"] = df["wcp_mean"] / df["wcp_mean"].max()
    df["wcp_max_normalized"] = df["wcp_max"] / df["wcp_max"].max()
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# %%
p = (
    p9.ggplot(combined_df, p9.aes(x="num_configs", y="wcp_mean_normalized", 
                                 color="system", linetype="optimization_target"))
    + p9.geom_line()
    + p9.theme_minimal()
    + p9.scale_y_continuous(limits=[0, None])
    + p9.scale_x_continuous(breaks=[1] + list(range(5, combined_df["num_configs"].max() + 1, 5)))
    + p9.labs(title="WCP Mean vs Number of Configurations",
              x="Number of Configurations",
              y="WCP Mean",
              color="System",
              linetype="Optimization Target")
)
p
# %%
combined_df.head()