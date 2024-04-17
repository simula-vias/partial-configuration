# %%
import pandas as pd
import numpy as np
import plotnine as p9
from scipy import stats
from common import load_data, pareto_rank
import json
import os
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, train_test_split
from mlxtend.frequent_patterns import fpgrowth, fpmax
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# %%
data_dir = Path("../data")
random_state = 1234
test_size = 0.2

# TODO Pareto front cut-off parameter

systems = json.load(open(data_dir / "metadata.json")).keys()

front_ratio = []
baseline_results = []

for s in systems:
    (
        perf_matrix,
        input_features,
        config_features,
        all_performances,
        input_preprocessor,
        config_preprocessor,
    ) = load_data(system=s, data_dir="../data")

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
    icm_all["ranks"] = icm_all.groupby("inputname", group_keys=False).apply(pareto_rank)

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
    for split_idx, (train_inp_idx, test_inp_idx) in enumerate(kf_inp.split(
        inputnames
    )):
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
        icm["ranks"] = icm.groupby("inputname", group_keys=False).apply(pareto_rank)

        dataset = (
            icm[icm.ranks <= 1].join(config_features).join(input_features).reset_index()
        )

        # Calculate the Pareto ranks for the test data
        icm_test = (
            test_perf[["inputname", "configurationID"] + all_performances]
            .sort_values(["inputname", "configurationID"])
            .set_index(["inputname", "configurationID"])
        )
        icm_test["ranks"] = icm_test.groupby("inputname", group_keys=False).apply(
            pareto_rank
        )

        ## HERE GOES SDC
        cfg_columns = ["configurationID"] + list(config_features.columns)
        top_cfgs = dataset[cfg_columns + ["inputname"]].groupby(cfg_columns, dropna=False, as_index=False).count().sort_values(
            "inputname", ascending=False
        ).configurationID.tolist()

        ## Here we select the configurations by decreasing coverage
        # If a configuration adds new items, we add it.
        # We repeat until all inputs are covered.

        covered_inputs = set()
        num_inputs = dataset.inputname.nunique()
        input_labels = pd.Series(np.zeros(num_inputs), index=dataset.inputname.unique())
        selected_configs = []

        for cid in top_cfgs:
            inpnames = dataset.query("configurationID == @cid").inputname.unique()
            new_inputs = set(inpnames).difference(covered_inputs)
            input_labels[list(new_inputs)] = cid
            covered_inputs.update(new_inputs)
            selected_configs.append(cid)

            if len(covered_inputs) == num_inputs:
                print(f"Reached full coverage with {len(selected_configs)} configurations")
                break

        input_labels = input_labels.sort_index()
        enc = LabelEncoder()
        y = enc.fit_transform(input_labels)

        X = input_preprocessor.fit_transform(
            input_features.query("inputname.isin(@input_labels.index)")
        )

        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X, y)

        X_test = input_preprocessor.transform(input_features.query("inputname.isin(@test_inp)"))
        pred_cfg = enc.inverse_transform(clf.predict(X_test)).astype(int)

        inp_pred_map = pd.DataFrame(zip(test_inp, pred_cfg), columns=["inputname", "configurationID"])
        sdc_ranks = icm_test.merge(inp_pred_map, on=["inputname", "configurationID"])["ranks"]

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

        overall_ranks = icm_test.query("configurationID == @best_cfg_id_overall").ranks
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
                len(selected_configs),
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
## Print baseline table
baselines = ["sdc", "overall", "metric", "common", "random"]
print("System & " + " & ".join(map(lambda s: s.capitalize(), baselines)) + "\\\\\\")
for r, v in baseline_df.groupby("system").mean().iterrows():
    res = " & ".join(["${avg:.2f}\\pm{std:.2f}$".format(avg=v[f"{b}_avg"], std=v[f"{b}_std"]) for b in baselines])
    print(f"{r} & {res} \\\\")

# %%
## Plot Rank/Ratio line graph
df = pd.DataFrame(front_ratio, columns=["System", "Rank", "Configurations in Front"])

plot = (
    p9.ggplot(df, mapping=p9.aes(x="Rank", y="Configurations in Front", color="System"))
    + p9.geom_line()
    + p9.geom_point()
)
plot.save("rank_ratio.pdf", bbox_inches="tight")
