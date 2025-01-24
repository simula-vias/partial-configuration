# %%
import pandas as pd
import numpy as np
import plotnine as p9
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from common import baseline_results_wc, load_data, DecisionTreeClassifierWithMultipleLabels
import json
from pathlib import Path
import itertools

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# %%
data_dir = Path("../data")
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

    all_perf_list = [[ap] for ap in all_performances_initial]

    for num_p in range(1, len(all_performances_initial) + 1):
        # TODO Do we want to consider all subsets of the performances?
        all_perf_list.append(all_performances_initial[:num_p])
        # all_perf_list.extend(list(map(list, itertools.combinations(all_performances_initial, num_p))))

    for all_performances in all_perf_list:
        num_p = len(all_performances)

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
        nmdf["worst_case_performance"] = nmdf[all_performances].max(axis=1)
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
            perf_matrix[["inputname", "configurationID", "worst_case_performance"]]
            .sort_values(["inputname", "configurationID"])
            .set_index(["inputname", "configurationID"])
        )
        # icm_ranked_measures = icm_all.groupby(
        #     "inputname"
        # ).transform(  # Go from measured values to ranks within each input group
        #     lambda x: x.argsort()+1
        # )
        icm_all["ranks"] = icm_all.groupby("inputname", group_keys=False).apply(
            lambda x: x["worst_case_performance"].argsort()+1
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
            train_inp = sorted(inputnames[train_inp_idx])
            test_inp = sorted(inputnames[test_inp_idx])
            train_perf = perf_matrix[perf_matrix.inputname.isin(train_inp)]
            test_perf = perf_matrix[perf_matrix.inputname.isin(test_inp)]

            icm = (
                perf_matrix[perf_matrix.inputname.isin(train_inp)][["inputname", "configurationID", "worst_case_performance"]]
                .sort_values(["inputname", "configurationID"])
                .set_index(["inputname", "configurationID"])
            )
            icm_ranked_measures = icm.groupby(
                "inputname"
            ).transform(  # Go from measured values to ranks within each input group
                lambda x: x.argsort()+1
            )
            icm["ranks"] = icm.groupby("inputname", group_keys=False).apply(
                lambda x: x["worst_case_performance"].argsort()+1
            )

            dataset = icm.join(config_features).join(input_features).reset_index()

            icm_test = (
                perf_matrix[~perf_matrix.inputname.isin(train_inp)][["inputname", "configurationID", "worst_case_performance"]]
                .sort_values(["inputname", "configurationID"])
                .set_index(["inputname", "configurationID"])
            )
            icm_test["ranks"] = icm_test.groupby("inputname", group_keys=False).apply(
                lambda x: x["worst_case_performance"].argsort()+1
            )

            def eval_prediction(pred_cfg_test):
                inp_pred_map = pd.DataFrame(
                    zip(test_inp, pred_cfg_test), columns=["inputname", "configurationID"]
                )
                return icm_test.merge(inp_pred_map, on=["inputname", "configurationID"])["worst_case_performance"] #.mean()


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

            input_labels = (
                icm.reset_index().groupby('inputname')
                .apply(lambda x: x.loc[x['worst_case_performance'].idxmin()], include_groups=False)
                # .set_index("inputname")
            )["configurationID"].astype(int)

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

            # for i in range(1, X.shape[1]):
            #     print(i)
            #     # clf = DecisionTreeClassifierWithMultipleLabels(max_depth=i, random_state=random_state)
            #     clf = RandomForestClassifier(n_estimators=10, max_depth=i)
            #     clf.fit(X_train, y_train)
            #     val_score = clf.score(X_val, y_val)
            #     print("Scores", clf.score(X_train, y_train), val_score)

            #     # Validation test
            #     pred_cfg_lbl = clf.predict(X_val)
            #     pred_cfg = enc.inverse_transform(pred_cfg_lbl).astype(int)
            #     inp_pred_map = pd.DataFrame(
            #         zip(inputnames_val, pred_cfg),
            #         columns=["inputname", "configurationID"],
            #     )
            #     val_rank = icm.merge(inp_pred_map, on=["inputname", "configurationID"])[
            #         "ranks"
            #     ].mean()
            #     print("Val rank", val_rank)

            #     if val_rank < best_val_rank:
            #         best_val_rank = val_rank
            #         best_depth = i

            # print(f"Best depth {best_depth} ({best_val_rank})")
            # clf = DecisionTreeClassifierWithMultipleLabels(
            #     max_depth=best_depth, random_state=random_state
            # )
            clf = RandomForestClassifier(n_estimators=10, random_state=random_state)
            clf.fit(X, y)

            X_test = input_preprocessor.transform(
                input_features.query("inputname.isin(@test_inp)")
            )
            pred_cfg = enc.inverse_transform(clf.predict(X_test)).astype(int)


            sdc_wcp = eval_prediction(pred_cfg)
            print(
                f"SDC-WCP: {sdc_wcp.mean():.2f}+-{sdc_wcp.std():.2f}"
            )

            ## These are our evaluation baselines
            # Baseline results
            baseline = baseline_results_wc(icm, icm_ranked_measures, icm_test, dataset, config_features, verbose=True)
            
            result_list.append(
                (
                    s,
                    split_idx,
                    0, #clf.unique_leaf_values(),
                    num_p,
                    "-".join(all_performances),
                    sdc_wcp.mean(),
                    sdc_wcp.std(),
                    baseline["best"][0],
                    baseline["best"][1],
                    baseline["average"][0],
                    baseline["average"][1],
                    baseline["overall"][0],
                    baseline["overall"][1],
                    baseline["metric"][0],
                    baseline["metric"][1],
                    baseline["common"][0],
                    baseline["common"][1],
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
        "performances",
        "sdc_avg",
        "sdc_std",
        "best_avg",
        "best_std",
        "average_avg",
        "average_std",
        "overall_avg",
        "overall_std",
        "metric_avg",
        "metric_std",
        "common_avg",
        "common_std",
    ],
)
baseline_df.to_csv("../results/baselines.csv", index=False)

# %%
## Print baseline table in latex
baselines = ["sdc", "overall", "metric", "common", "average", "best"]
print("System & |P| &" + " & ".join(map(lambda s: s.capitalize(), baselines)) + "\\\\\\midrule")
for (r, p, pn), v in baseline_df.groupby(["system", "num_performances", "performances"]).mean().iterrows():
    res = " & ".join(
        [
            "${avg:.2f}\\pm{std:.2f}$".format(avg=v[f"{b}_avg"], std=v[f"{b}_std"])
            for b in baselines
        ]
    )
    if p == 1:
        r = f"{r}-{pn}"
    print(f"{r} & {p} & {res} \\\\")

# %%
