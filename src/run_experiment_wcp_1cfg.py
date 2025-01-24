# %%
import pandas as pd
import numpy as np
import plotnine as p9
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from common import (
    baseline_results_wc,
    load_data,
    DecisionTreeClassifierWithMultipleLabels,
)
import json
from pathlib import Path
import itertools

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# %%
data_dir = Path("../data")
random_state = 1234
classifier = "rf"  # "dt", "rf"

num_performances = -1  # -1: all performances / takes first n if > 0

systems = json.load(open(data_dir / "metadata.json")).keys()

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
        if True or s == "sqlite":
            # sqlite has too many performance measures
            all_perf_list.append(all_performances_initial[:num_p])
        else:
            all_perf_list.extend(
                list(map(list, itertools.combinations(all_performances_initial, num_p)))
            )

    for all_performances in all_perf_list:
        num_p = len(all_performances)

        # We can normalize before splitting, because
        # we normalize per input and we also split per input.
        # There is no data leakage.
        normalized_metrics = (
            perf_matrix_initial[["inputname"] + all_performances]
            .groupby("inputname", as_index=False)
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
        )

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

        # We adjust the WCP by expressing it as the difference from the best WCP, i.e. the best WCP is always 0
        perf_matrix["worst_case_performance"] = (  # worst_case_performance_adjusted
            perf_matrix[["inputname", "worst_case_performance"]]
            .groupby("inputname", as_index=True)
            .transform(lambda x: x - x.min())
        )

        all_perf_raw = [f"{p}_raw" for p in all_performances]
        all_perf_norm = [f"{p}" for p in all_performances]

        icm_all = (
            perf_matrix[["inputname", "configurationID", "worst_case_performance"]]
            .sort_values(["inputname", "configurationID"])
            .set_index(["inputname", "configurationID"])
        )

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
                perf_matrix[perf_matrix.inputname.isin(train_inp)][
                    ["inputname", "configurationID", "worst_case_performance"]
                ]
                .sort_values(["inputname", "configurationID"])
                .set_index(["inputname", "configurationID"])
            )
            icm_all_perf = (
                perf_matrix[["inputname", "configurationID"] + all_performances]
                .sort_values(["inputname", "configurationID"])
                .set_index(["inputname", "configurationID"])
            )

            dataset = icm.join(config_features).join(input_features).reset_index()

            icm_test = (
                perf_matrix[~perf_matrix.inputname.isin(train_inp)][
                    ["inputname", "configurationID", "worst_case_performance"]
                ]
                .sort_values(["inputname", "configurationID"])
                .set_index(["inputname", "configurationID"])
            )

            def eval_prediction(pred_cfg_test):
                inp_pred_map = pd.DataFrame(
                    zip(test_inp, pred_cfg_test),
                    columns=["inputname", "configurationID"],
                )
                return icm_test.merge(
                    inp_pred_map, on=["inputname", "configurationID"]
                )["worst_case_performance"]  # .mean()

            ## HERE GOES SDC
            cfg_columns = ["configurationID"] + list(config_features.columns)
            top_cfgs = (
                dataset[cfg_columns + ["inputname"]]
                .groupby(cfg_columns, dropna=False, as_index=False)
                .count()
                .sort_values("inputname", ascending=False)
                .configurationID.tolist()
            )

            input_labels = (
                icm.reset_index()
                .groupby("inputname")
                .apply(
                    lambda x: x.loc[x["worst_case_performance"].idxmin()],
                    include_groups=False,
                )
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
            #     # print(i)
            #     # clf = DecisionTreeClassifierWithMultipleLabels(max_depth=i, random_state=random_state)
            #     if classifier == "dt":
            #         clf = DecisionTreeClassifier(max_depth=i, random_state=random_state)
            #     else:
            #         clf = RandomForestClassifier(
            #             n_estimators=10, max_depth=i, random_state=random_state
            #         )
            #     # clf = DecisionTreeClassifier(max_depth=i, random_state=random_state)
            #     clf.fit(X_train, y_train)

            #     # Validation test
            #     pred_cfg_lbl = clf.predict(X_val)
            #     pred_cfg = enc.inverse_transform(pred_cfg_lbl).astype(int)
            #     inp_pred_map = pd.DataFrame(
            #         zip(inputnames_val, pred_cfg),
            #         columns=["inputname", "configurationID"],
            #     )
            #     val_rank = icm.merge(inp_pred_map, on=["inputname", "configurationID"])[
            #         "worst_case_performance"
            #     ].mean()

            #     if val_rank < best_val_rank:
            #         best_val_rank = val_rank
            #         best_depth = i

            # print(f"Best depth {best_depth} ({best_val_rank})")

            if classifier == "dt":
                clf = DecisionTreeClassifier(max_depth=None, random_state=random_state)
            else:
                clf = RandomForestClassifier(
                    n_estimators=10, max_depth=None, random_state=random_state
                )
            clf.fit(X, y)

            X_test = input_preprocessor.transform(
                input_features.query("inputname.isin(@test_inp)")
            )
            pred_cfg = enc.inverse_transform(clf.predict(X_test)).astype(int)

            sdc_wcp = eval_prediction(pred_cfg)
            
            if classifier == "dt":
                leaf_nodes = np.where(clf.tree_.children_left == clf.tree_.children_right)[
                    0
                ]
                num_leaf_nodes = len(leaf_nodes)
                num_predictable_configs = np.unique(
                    clf.tree_.value[leaf_nodes, 0, :].argmax(axis=-1)
                ).shape[0]
                max_depth = clf.get_depth()
            else:
                num_leaf_nodes = 0
                num_predictable_configs = []
                for est in clf.estimators_:
                    leaf_nodes = np.where(est.tree_.children_left == est.tree_.children_right)[
                        0
                    ]
                    num_leaf_nodes += len(leaf_nodes)
                    num_predictable_configs.append(
                        np.unique(
                            est.tree_.value[leaf_nodes, 0, :].argmax(axis=-1)
                        )
                    )
                num_predictable_configs = np.unique(np.concatenate(num_predictable_configs)).shape[0]
                max_depth = max([est.get_depth() for est in clf.estimators_])
            num_total_configs = perf_matrix.configurationID.nunique()

            print(f"SDC-WCP: {sdc_wcp.mean():.2f}+-{sdc_wcp.std():.2f} (with {num_predictable_configs} configs)")

            ## These are our evaluation baselines
            # Baseline results
            baseline = baseline_results_wc(
                icm,
                icm_all_perf,
                icm_test,
                dataset,
                config_features,
                verbose=True,
            )

            result_list.append(
                (
                    s,
                    split_idx,
                    0,  # clf.unique_leaf_values(),
                    num_p,
                    "-".join(all_performances),
                    max_depth / X.shape[1],
                    num_leaf_nodes,
                    num_predictable_configs / num_total_configs,
                    sdc_wcp.mean(),
                    sdc_wcp.std(),
                    baseline["best"][0],
                    baseline["best"][1],
                    baseline["best_num_configs"] / num_total_configs,
                    baseline["average"][0],
                    baseline["average"][1],
                    baseline["overall"][0],
                    baseline["overall"][1],
                    baseline["metric"][0],
                    baseline["metric"][1],
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
        "tree_depth",
        "tree_leaves",
        "sdc_num_configs",
        "sdc_avg",
        "sdc_std",
        "best_avg",
        "best_std",
        "best_num_configs",
        "average_avg",
        "average_std",
        "overall_avg",
        "overall_std",
        "metric_avg",
        "metric_std",
    ],
)
baseline_df.to_csv("../results/wcp_1cfg.csv", index=False)

# %%
## Print baseline table in latex
baselines = ["sdc", "overall", "metric", "average"]
print(
    "System & |P| &"
    + " & ".join(map(lambda s: s.capitalize(), baselines))
    + "\\\\\\midrule"
)
aggdf = (
    baseline_df[
        [
            "system",
            "num_performances",
            "tree_depth",
            "tree_leaves",
            "sdc_num_configs",
            "sdc_avg",
            "sdc_std",
            "best_num_configs",
            "average_avg",
            "average_std",
            "overall_avg",
            "overall_std",
            "metric_avg",
            "metric_std",
        ]
    ]
    .groupby(["system", "num_performances"])
    .mean()
)

for (r, pn), v in aggdf.iterrows():
    res = " & ".join(
        [
            "${avg:.2f}\\pm{std:.2f}$".format(avg=100*v[f"{b}_avg"], std=100*v[f"{b}_std"])
            for b in baselines
        ]
    )
    # if p == 1:
    #     r = f"{r}-{pn}"
    print(f"{r} & {pn} & {res} \\\\")

# %%
# Plot single-metric results
baselines = ["sdc", "overall", "metric", "average"]
print(
    "System & |P| &"
    + " & ".join(map(lambda s: s.capitalize(), baselines))
    + "\\\\\\midrule"
)
aggdf = (
    baseline_df[baseline_df.num_performances == 1][
        [
            "system",
            "performances",
            "tree_depth",
            "sdc_num_configs",
            "sdc_avg",
            "sdc_std",
            "best_num_configs",
            "average_avg",
            "average_std",
            "overall_avg",
            "overall_std",
            "metric_avg",
            "metric_std",
        ]
    ]
    .groupby(["system", "performances"])
    .mean()
)

for (r, pn), v in aggdf.iterrows():
    res = " & ".join(
        [
            "${avg:.1f}\\pm{std:.1f}$".format(
                avg=100 * v[f"{b}_avg"], std=100 * v[f"{b}_std"]
            )
            for b in baselines
        ]
    )
    # if p == 1:
    #     r = f"{r}-{pn}"
    print(f"{r} & {pn} & {res} \\\\")


# %%
