# %%
import pandas as pd
import numpy as np
import plotnine as p9
from scipy import stats
from common import (
    load_data,
    pareto_rank,
    DecisionTreeClassifierWithMultipleLabels,
    DecisionTreeClassifierWithMultipleLabelsPandas,
)
import json
from pathlib import Path

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

# %%
data_dir = Path("../data")
random_state = 1234
test_size = 0.2
pareto_cutoff = 0.2
rank_by_domination_count = False
# classifier = "customdt"  # "dt", "rf"

num_performances = -1  # -1: all performances (increasing)

systems = json.load(open(data_dir / "metadata.json")).keys()

front_ratio = []
baseline_results = []

for s in systems:
    (
        perf_matrix_initial,
        input_features,
        config_features,
        all_performances_initial,
        input_preprocessor,
        config_preprocessor,
    ) = load_data(system=s, data_dir="../data")

    for num_p in range(1, max(num_performances, len(all_performances_initial)) + 1):
        all_performances = all_performances_initial[:num_p]

        # Normalization is needed for the Pareto cutoff
        # We can normalize before splitting, because
        # we normalize per input and we also split per input.
        # There is no data leakage.
        normalized_metrics = (
            perf_matrix_initial[["inputname"] + all_performances]
            .groupby("inputname", as_index=False)
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        )
        cutoff_mask = (normalized_metrics <= pareto_cutoff).all(axis=1)

        nmdf = (
            perf_matrix_initial[["inputname"] + all_performances]
            .groupby("inputname", as_index=True)
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        )
        perf_matrix = pd.merge(
            perf_matrix_initial,
            nmdf,
            suffixes=("_raw", None),
            left_index=True,
            right_index=True,
        )
        perf_matrix["feasible"] = cutoff_mask

        all_perf_raw = [f"{p}_raw" for p in all_performances]
        all_perf_norm = [f"{p}" for p in all_performances]

        icm_all = (
            perf_matrix[["inputname", "configurationID"] + all_performances]
            .sort_values(["inputname", "configurationID"])
            .set_index(["inputname", "configurationID"])
        )
        icm_ranked_measures = icm_all.groupby(
            "inputname"
        ).transform(  # Go from measured values to ranks within each input group
            lambda x: stats.rankdata(x, method="min")
        )
        icm_all["ranks"] = icm_all.groupby("inputname", group_keys=False).apply(
            lambda x: pareto_rank(
                x,
                cutoff=pareto_cutoff,
                rank_by_domination_count=rank_by_domination_count,
            )
        )

        ## Rank/Ratio Line Graph
        num_cfgs = config_features.shape[0]

        for i in range(1, 11):
            num_cfgs_in_front = (
                icm_all[icm_all["ranks"] <= i].reset_index().configurationID.nunique()
            )
            front_ratio.append((s, i, num_cfgs_in_front / num_cfgs))
            print(s, i, num_cfgs_in_front / num_cfgs)

        ### From here we must split the data
        splits = 5
        kf_inp = KFold(n_splits=splits, random_state=random_state, shuffle=True)

        inputnames = perf_matrix["inputname"].unique()

        ## Baselines
        for split_idx, (train_inp_idx, test_inp_idx) in enumerate(
            kf_inp.split(inputnames)
        ):
            train_inp = inputnames[train_inp_idx]
            test_inp = inputnames[test_inp_idx]
            train_perf = perf_matrix[perf_matrix.inputname.isin(train_inp)]
            test_perf = perf_matrix[perf_matrix.inputname.isin(test_inp)]

            icm = (
                train_perf[["inputname", "configurationID"] + all_performances]
                .sort_values(["inputname", "configurationID"])
                .set_index(["inputname", "configurationID"])
            )
            icm_ranked_measures = icm.groupby(
                "inputname"
            ).transform(  # Go from measured values to ranks within each input group
                lambda x: stats.rankdata(x, method="min")
            )
            icm["ranks"] = icm.groupby("inputname", group_keys=False).apply(
                lambda x: pareto_rank(
                    x,
                    cutoff=pareto_cutoff,
                    rank_by_domination_count=rank_by_domination_count,
                )
            )

            dataset = (
                icm[icm.ranks <= 1]
                .join(config_features)
                .join(input_features)
                .reset_index()
            )

            # Calculate the Pareto ranks for the test data
            icm_test = (
                test_perf[["inputname", "configurationID"] + all_performances]
                .sort_values(["inputname", "configurationID"])
                .set_index(["inputname", "configurationID"])
            )
            icm_test["ranks"] = icm_test.groupby("inputname", group_keys=False).apply(
                lambda x: pareto_rank(
                    x,
                    cutoff=pareto_cutoff,
                    rank_by_domination_count=rank_by_domination_count,
                )
            )

            ## HERE GOES SDC
            cfg_columns = ["configurationID"] + list(config_features.columns)
            top_cfgs = (
                dataset[cfg_columns + ["inputname"]]
                .groupby(cfg_columns, dropna=False, as_index=False)
                .count()
                .sort_values("inputname", ascending=False)
                .configurationID.tolist()
            )

            enc = LabelEncoder()
            enc.fit(dataset["configurationID"].tolist())

            grouped_df = (
                dataset.groupby("inputname")["configurationID"]
                .apply(enc.transform)
                .reset_index()
            )
            mlb = MultiLabelBinarizer()
            # Fit and transform the 'Values' column
            binary_matrix = mlb.fit_transform(grouped_df["configurationID"])

            # Create a new DataFrame with the binary matrix
            binary_df = pd.DataFrame(
                binary_matrix, columns=mlb.classes_, index=grouped_df["inputname"]
            )

            # X = input_preprocessor.fit_transform(
            #     input_features[
            #         input_features.index.get_level_values("inputname").isin(train_inp)
            #     ].sort_index()
            # )
            X = input_features[
                    input_features.index.get_level_values("inputname").isin(train_inp)
                ].sort_index()
            y = binary_df.values

            # TODO Cross-validation

            train_idx, val_idx = train_test_split(
                np.arange(X.shape[0]), test_size=0.2, random_state=random_state
            )
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            inputnames_train = train_inp[train_idx]
            inputnames_val = train_inp[val_idx]

            best_val_rank = 100_000
            best_depth = 0

            for i in range(1, X.shape[1]):
                clf = DecisionTreeClassifierWithMultipleLabelsPandas(
                    max_depth=i #, random_state=random_state
                )
                clf.fit(X_train, y_train)
                train_score = clf.score(X_train, y_train)
                val_score = clf.score(X_val, y_val)

                # Validation test
                pred_cfg_lbl = clf.predict(X_val)
                pred_cfg = enc.inverse_transform(pred_cfg_lbl).astype(int)
                inp_pred_map = pd.DataFrame(
                    zip(inputnames_val, pred_cfg),
                    columns=["inputname", "configurationID"],
                )
                val_rank = icm.merge(inp_pred_map, on=["inputname", "configurationID"])[
                    "ranks"
                ].mean()

                print(
                    f"Depth={i}, Train score={train_score:.2f}, Val score={val_score:.2f}, Val rank={val_rank:.2f}"
                )

                if val_rank < best_val_rank:
                    best_val_rank = val_rank 
                    best_depth = i

                if train_score == 1.0:
                    break

            print(f"Best depth {best_depth} ({best_val_rank})")
            clf = DecisionTreeClassifierWithMultipleLabelsPandas(
                max_depth=best_depth #, random_state=random_state
            )
            clf.fit(X, y)

            # X_test = input_preprocessor.transform(
            #     input_features.query("inputname.isin(@test_inp)")
            # )
            X_test = input_features[input_features.index.get_level_values("inputname").isin(test_inp)].sort_index()
            pred_cfg = enc.inverse_transform(clf.predict(X_test)).astype(int)

            inp_pred_map = pd.DataFrame(
                zip(test_inp, pred_cfg), columns=["inputname", "configurationID"]
            )
            sdc_ranks = icm_test.merge(
                inp_pred_map, on=["inputname", "configurationID"]
            )["ranks"]

            print(
                f"Average rank of the SDC configuration: {sdc_ranks.mean():.2f}+-{sdc_ranks.std():.2f}"
            )

            ## These are our evaluation baselines
            # The best configuration by averaging the ranks over all inputs
            best_cfg_id_overall = (
                icm[["ranks"]].groupby("configurationID").mean().idxmin().item()
            )

            # The best configuration per performance metric
            best_cfg_id_per_metric = (
                icm_ranked_measures.groupby("configurationID").mean().idxmin()
            )

            # The most common configuration in the Pareto fronts
            most_common_cfg_id = (
                dataset[["configurationID"] + [config_features.columns[0]]]
                .groupby(["configurationID"], as_index=False)
                .count()
                .sort_values(by=config_features.columns[0], ascending=False)
                .iloc[0]
                .configurationID
            )

            overall_ranks = icm_test.query(
                "configurationID == @best_cfg_id_overall"
            ).ranks
            print(
                f"Average rank of the overall best configuration: {overall_ranks.mean():.2f}+-{overall_ranks.std():.2f}"
            )

            for p in all_performances:
                cfg_id = best_cfg_id_per_metric[p]
                metric_p = icm_test.query("configurationID == @cfg_id").ranks
                print(
                    f"Average rank of the best configuration for {p}: {metric_p.mean():.2f}+-{metric_p.std():.2f}"
                )

            metric_rank = icm_test.query(
                "configurationID.isin(@best_cfg_id_per_metric.values)"
            ).ranks
            print(
                f"Average rank of the best configuration for all metrics: {metric_rank.mean():.2f}+-{metric_rank.std():.2f}"
            )

            common_rank = icm_test.query("configurationID == @most_common_cfg_id").ranks
            print(
                f"Average rank of the most common configuration: {common_rank.mean():.2f}+-{common_rank.std():.2f}"
            )

            # TODO Not sure std. dev. is correct here. We sample all random configs at once.
            random_ranks = np.random.randint(0, test_perf.configurationID.max(), 10) + 1
            random_rank = icm_test.query("configurationID.isin(@random_ranks)").ranks
            print(
                f"Average rank of random configuration: {random_rank.mean():.2f}+-{random_rank.std():.2f}"
            )

            baseline_results.append(
                (
                    s,
                    split_idx,
                    clf.unique_leaf_values(),
                    num_p,
                    sdc_ranks.mean(),
                    sdc_ranks.std(),
                    overall_ranks.mean(),
                    overall_ranks.std(),
                    metric_rank.mean(),
                    metric_rank.std(),
                    common_rank.mean(),
                    common_rank.std(),
                    random_rank.mean(),
                    random_rank.std(),
                )
            )

        print("")

baseline_df = pd.DataFrame(
    baseline_results,
    columns=[
        "system",
        "split",
        "configs",
        "num_performances",
        "sdc_avg",
        "sdc_std",
        "overall_avg",
        "overall_std",
        "metric_avg",
        "metric_std",
        "common_avg",
        "common_std",
        "random_avg",
        "random_std",
    ],
)
baseline_df.to_csv("../results/baselines.csv", index=False)

# %%
## Print baseline table in latex
baselines = ["sdc", "overall", "metric", "common", "random"]
print("System & |P| &" + " & ".join(map(lambda s: s.capitalize(), baselines)) + "\\\\")
for (r, p), v in baseline_df.groupby(["system", "num_performances"]).mean().iterrows():
    res = " & ".join(
        [
            "${avg:.2f}\\pm{std:.2f}$".format(avg=v[f"{b}_avg"], std=v[f"{b}_std"])
            for b in baselines
        ]
    )
    print(f"{r} & {p} & {res} \\\\")


# %%

from plotnine import *
import pandas as pd

# Assuming your dataframe is named 'df'
# If not, replace 'df' with the actual name of your dataframe

df_aggregated = baseline_df.groupby(['system', 'num_performances']).agg({
    'sdc_avg': 'mean',
    'sdc_std': 'mean',  # Taking mean of standard deviations
    'overall_avg': 'mean',
    'overall_std': 'mean',
    'common_avg': 'mean',
    'common_std': 'mean'
}).reset_index()

# Melt the dataframe to create a long format suitable for plotting
df_melted = pd.melt(df_aggregated, 
                    id_vars=['system', 'num_performances'],
                    value_vars=['sdc_avg', 'overall_avg', 'common_avg'],
                    var_name='metric',
                    value_name='average')

# Create a corresponding dataframe for standard deviations
df_std = pd.melt(df_aggregated, 
                 id_vars=['system', 'num_performances'],
                 value_vars=['sdc_std', 'overall_std', 'common_std'],
                 var_name='metric',
                 value_name='std')

# Rename the metric values to match between average and std dataframes
df_melted['metric'] = df_melted['metric'].str.replace('_avg', '')
df_std['metric'] = df_std['metric'].str.replace('_std', '')

# Merge the average and std dataframes
df_plot = pd.merge(df_melted, df_std, on=['system', 'num_performances', 'metric'])

# Create the plot
plot = (ggplot(df_plot, aes(x='factor(num_performances)', y='average', fill='metric'))
        + geom_boxplot(aes(ymin='average-std', ymax='average+std', middle='average', lower='average-std', upper='average+std'),
                       stat='identity', position=position_dodge(width=0.9))
        + facet_wrap('~ system', scales='free')
        + labs(x='Number of Performances', y='Average', title='Boxplots by System and Number of Performances')
        + theme_minimal()
        + theme(axis_text_x=element_text(angle=45, hjust=1),figure_size=(16, 12))
)

# Display the plot
plot.show()

# Save the plot (optional)
# plot.save('boxplot.png', dpi=300)

# %%
## Plot Rank/Ratio line graph
df = pd.DataFrame(front_ratio, columns=["System", "Rank", "Configurations in Front"])

plot = (
    p9.ggplot(df, mapping=p9.aes(x="Rank", y="Configurations in Front", color="System"))
    + p9.geom_line()
    + p9.geom_point()
)
plot.save("rank_ratio.pdf", bbox_inches="tight")
