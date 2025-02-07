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

from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.tree import DecisionTreeClassifier

# %%
data_dir = Path("../data")
random_state = 1234
classifier = "rf"  # "dt", "rf"

systems = json.load(open(data_dir / "metadata.json")).keys()

result_list = []
result_list_dict = []
for s in systems:
    print(s)
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
        nmdf = (
            perf_matrix_initial[["inputname"] + all_performances]
            .groupby("inputname", as_index=True)
            .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
            # .transform(lambda x: scale(x))
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


            if classifier == "dt":
                parameter_grid = {
                    "max_depth": range(1, X.shape[1] + 1),
                    "criterion": ["entropy", "gini"],
                }
                clf = DecisionTreeClassifier()
            else:
                parameter_grid = {
                    "n_estimators": [4, 8, 12, 16], #range(1, 100, 10),
                    "max_depth": range(1, X.shape[1] + 1),
                    "criterion": ["entropy", "gini"],
                }
                clf = RandomForestClassifier()

            clf = GridSearchCV(
                clf,
                parameter_grid,
                cv=KFold(n_splits=4, random_state=random_state, shuffle=True),
                n_jobs=-1,
                refit=True,
            )
            clf.fit(X, y)

            split_result = {
                "classifier": classifier,
                **clf.best_params_,
            }

            X_test = input_preprocessor.transform(
                input_features.query("inputname.isin(@test_inp)")
            )
            pred_cfg = enc.inverse_transform(clf.predict(X_test)).astype(int)

            sdc_wcp = eval_prediction(pred_cfg)

            # if classifier == "dt":
            #     leaf_nodes = np.where(
            #         clf.tree_.children_left == clf.tree_.children_right
            #     )[0]
            #     num_leaf_nodes = len(leaf_nodes)
            #     num_predictable_configs = np.unique(
            #         clf.tree_.value[leaf_nodes, 0, :].argmax(axis=-1)
            #     ).shape[0]
            #     max_depth = clf.get_depth()
            # else:
            #     num_leaf_nodes = 0
            #     num_predictable_configs = []
            #     for est in clf.estimators_:
            #         leaf_nodes = np.where(
            #             est.tree_.children_left == est.tree_.children_right
            #         )[0]
            #         num_leaf_nodes += len(leaf_nodes)
            #         num_predictable_configs.append(
            #             np.unique(est.tree_.value[leaf_nodes, 0, :].argmax(axis=-1))
            #         )
            #     num_predictable_configs = np.unique(
            #         np.concatenate(num_predictable_configs)
            #     ).shape[0]
            #     max_depth = max([est.get_depth() for est in clf.estimators_])
            num_total_configs = perf_matrix.configurationID.nunique()
            num_predictable_configs = -1  # icm.configurationID.nunique()
            max_depth = -1  # clf.best_params_["max_depth"]
            num_leaf_nodes = -1  # clf.best_params_["max_depth"]

            ## These are our evaluation baselines
            # Baseline results
            baseline = baseline_results_wc(
                icm,
                icm_all_perf,
                icm_test,
                dataset,
                config_features,
                verbose=False,
            )
            baseline["sdc_avg"] = sdc_wcp.mean()
            baseline["sdc_std"] = sdc_wcp.std()
            baseline["sdc_max"] = sdc_wcp.max()
            split_result.update(baseline)

            print(json.dumps(split_result, indent=2, sort_keys=True))

            result_list_dict.append(split_result)
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
                    baseline["best_avg"],
                    baseline["best_std"],
                    baseline["best_num_configs"] / num_total_configs,
                    baseline["average_avg"],
                    baseline["average_std"],
                    baseline["overall_avg"],
                    baseline["overall_std"],
                    baseline["metric_avg"],
                    baseline["metric_std"],
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
            "${avg:.2f}\\pm{std:.2f}$".format(
                avg=100 * v[f"{b}_avg"], std=100 * v[f"{b}_std"]
            )
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
