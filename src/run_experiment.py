# %%
import pandas as pd
import numpy as np
import plotnine as p9
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from common import baseline_results, load_data, pareto_rank, DecisionTreeClassifierWithMultipleLabels
import json
from pathlib import Path

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# %%
data_dir = Path("./data")
random_state = 1234
test_size = 0.2
pareto_cutoff = 0.6
rank_by_domination_count = False
# classifier = "customdt"  # "dt", "rf"

num_performances = -1  # -1: all performances / takes first n if > 0

systems = json.load(open(data_dir / "metadata.json")).keys()

front_ratio = []
result_list = []

for s in systems:
    (
        perf_matrix_initial,
        input_features,
        config_features,
        all_performances_initial,
        input_preprocessor,
        config_preprocessor,
    ) = load_data(system=s, data_dir=data_dir)

    for num_p in range(1, max(num_performances, len(all_performances_initial)) + 1):
        all_performances = all_performances_initial[:num_p]
        # all_performances = ["kbs"]

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
                x, cutoff=pareto_cutoff, rank_by_domination_count=rank_by_domination_count
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
                lambda x: pareto_rank(x, cutoff=pareto_cutoff, rank_by_domination_count=rank_by_domination_count)
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
                lambda x: pareto_rank(x, cutoff=pareto_cutoff, rank_by_domination_count=rank_by_domination_count)
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

            ## Here we select the configurations by decreasing coverage
            # If a configuration adds new items, we add it.
            # We repeat until all inputs are covered.

            # TODO We do not need this for the custom decision tree model
            # TODO We should also explore alternatives for the normal decision tree model

            covered_inputs = set()
            num_inputs = dataset.inputname.nunique()
            input_labels = pd.Series(
                np.zeros(num_inputs), index=dataset.inputname.unique()
            )
            selected_configs = []

            for cid in top_cfgs:
                inpnames = dataset.query("configurationID == @cid").inputname.unique()
                new_inputs = set(inpnames).difference(covered_inputs)
                input_labels[list(new_inputs)] = cid
                covered_inputs.update(new_inputs)
                selected_configs.append(cid)

                if len(covered_inputs) == num_inputs:
                    print(
                        f"Reached full coverage with {len(selected_configs)} configurations"
                    )
                    break

            input_labels = input_labels.sort_index()
            enc = LabelEncoder()
            y = enc.fit_transform(input_labels)

            X = input_preprocessor.fit_transform(
                input_features.query("inputname.isin(@input_labels.index)")
            )

            # TODO Cross-validation
            # TODO Is decision tree the best model for the final prediction?

            train_idx, val_idx = train_test_split(
                np.arange(X.shape[0]), test_size=0.2, random_state=random_state
            )
            X_train = X[train_idx]
            X_val = X[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            inputnames_val = input_labels.index[val_idx]

            # X_train = X
            # y_train = y
            # X_val = X
            # y_val = y
            # inputnames_val = input_labels.index

            best_val_rank = 100_000
            best_depth = 0

            for i in range(1, X.shape[1]):
                print(i)
                # clf = DecisionTreeClassifierWithMultipleLabels(max_depth=i, random_state=random_state)
                clf = RandomForestClassifier(n_estimators=10, max_depth=i)
                clf.fit(X_train, y_train)
                val_score = clf.score(X_val, y_val)
                print("Scores", clf.score(X_train, y_train), val_score)

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
                print("Val rank", val_rank)

                if val_rank < best_val_rank:
                    best_val_rank = val_rank
                    best_depth = i

            print(f"Best depth {best_depth} ({best_val_rank})")
            # clf = DecisionTreeClassifierWithMultipleLabels(
            #     max_depth=best_depth, random_state=random_state
            # )
            clf = RandomForestClassifier(n_estimators=10, max_depth=best_depth, random_state=random_state)
            clf.fit(X, y)

            X_test = input_preprocessor.transform(
                input_features.query("inputname.isin(@test_inp)")
            )
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
            # Baseline results
            baseline = baseline_results(icm, icm_ranked_measures, icm_test, dataset, config_features, verbose=True)
            
            result_list.append(
                (
                    s,
                    split_idx,
                    clf.unique_leaf_values(),
                    num_p,
                    sdc_ranks.mean(),
                    sdc_ranks.std(),
                    baseline["overall"][0],
                    baseline["overall"][1],
                    baseline["metric"][0],
                    baseline["metric"][1],
                    baseline["common"][0],
                    baseline["common"][1],
                    baseline["random"][0],
                    baseline["random"][1],
                )
            )

        print("")

baseline_df = pd.DataFrame(
    result_list,
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
## Plot Rank/Ratio line graph
df = pd.DataFrame(front_ratio, columns=["System", "Rank", "Configurations in Front"])

plot = (
    p9.ggplot(df, mapping=p9.aes(x="Rank", y="Configurations in Front", color="System"))
    + p9.geom_line()
    + p9.geom_point()
)
plot.save("rank_ratio.pdf", bbox_inches="tight")
