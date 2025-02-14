# %%
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
)
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.tree import DecisionTreeClassifier

from common import (
    baseline_results_wc,
    get_leaf_values,
    load_data,
)

# %%
data_dir = Path("../data")
random_state = 1234
classifier = "rf"  # "dt", "rf"

systems = json.load(open(data_dir / "metadata.json")).keys()

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
                    "n_estimators": [2, 4, 8, 12, 16],  # range(1, 100, 10),
                    "max_depth": range(1, X.shape[1] + 1),
                    "criterion": ["entropy", "gini"],
                    # "bootstrap": [True, False],
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
                "system": s,
                "split": split_idx,
                "classifier": classifier,
                "performances": "-".join(all_performances),
                **clf.best_params_,
            }

            X_test = input_preprocessor.transform(
                input_features.query("inputname.isin(@test_inp)")
            )
            pred_cfg = enc.inverse_transform(clf.predict(X_test)).astype(int)

            sdc_wcp = eval_prediction(pred_cfg)

            best_clf = clf.best_estimator_

            if classifier == "dt":
                num_leaf_nodes = get_leaf_values(best_clf.tree_).shape[0]
                num_predictable_configs = best_clf.n_classes_
                max_depth = best_clf.get_depth()
            elif classifier == "rf":
                num_leaf_nodes = sum(
                    get_leaf_values(est.tree_).shape[0] for est in best_clf.estimators_
                )
                num_predictable_configs = best_clf.n_classes_
                max_depth = max(est.get_depth() for est in best_clf.estimators_)

            num_total_configs = perf_matrix.configurationID.nunique()

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

            split_result["max_depth"] = max_depth / X.shape[1]
            split_result["num_leaf_nodes"] = num_leaf_nodes
            split_result["num_predictable_configs"] = num_predictable_configs
            split_result["num_total_configs"] = num_total_configs
            split_result["num_performances"] = num_p

            print(json.dumps(split_result, indent=2, sort_keys=True))

            result_list_dict.append(split_result)

            if s == "gcc" and all_performances == ["exec"]:
                break
        if s == "gcc" and all_performances == ["exec"]:
            break
    if s == "gcc" and all_performances == ["exec"]:
        break

        print("")
# %%
baseline_df = pd.DataFrame(result_list_dict)
baseline_df.to_csv("../results/wcp_1cfg.csv", index=False)

# %%


def make_latex_table(query_columns, baselines, base_query=None, pvalue_threshold=0.05):
    # TODO Lookup table for nice column names

    col_desc = "l" * len(query_columns) + "r" * len(baselines)
    print("\\begin{tabular}{" + col_desc + "}")
    print("\\toprule")
    print(
        " & ".join(map(lambda s: s.capitalize().replace("_", " "), query_columns))
        + " & "
        + " & ".join(map(lambda s: s.capitalize().replace("_", " "), baselines))
        + "\\\\\\midrule"
    )
    if base_query is not None:
        aggdf = baseline_df.query(base_query)
    else:
        aggdf = baseline_df
    aggdf = aggdf.groupby(query_columns).mean(numeric_only=True)
    for group_values, v in aggdf.iterrows():
        # Get just this group's results from result_list_dict
        filter_dict = dict(
            zip(
                query_columns,
                [group_values] if isinstance(group_values, str) else group_values,
            )
        )
        system_results = pd.DataFrame(
            [
                d
                for d in result_list_dict
                if all(d[k] == v for k, v in filter_dict.items())
            ]
        )

        # Check if SDC is significantly better than overall
        diff_avg_sdc = system_results["sdc_avg"] - system_results["overall_avg"]
        diff_max_sdc = system_results["sdc_max"] - system_results["overall_max"]
        non_zero_diff_avg_sdc = diff_avg_sdc[diff_avg_sdc != 0]
        non_zero_diff_max_sdc = diff_max_sdc[diff_max_sdc != 0]

        # Check if overall is significantly better than SDC
        diff_avg_overall = system_results["overall_avg"] - system_results["sdc_avg"]
        diff_max_overall = system_results["overall_max"] - system_results["sdc_max"]
        non_zero_diff_avg_overall = diff_avg_overall[diff_avg_overall != 0]
        non_zero_diff_max_overall = diff_max_overall[diff_max_overall != 0]

        # Only perform test if we have enough non-zero differences
        if len(non_zero_diff_avg_sdc) >= 5:  # minimum recommended sample size
            wilcoxon_avg_sdc = stats.wilcoxon(non_zero_diff_avg_sdc, alternative="less")
            avg_marker_sdc = "^*" if wilcoxon_avg_sdc.pvalue < pvalue_threshold else ""
        else:
            avg_marker_sdc = "^-"

        if len(non_zero_diff_max_sdc) > 5:
            wilcoxon_max_sdc = stats.wilcoxon(non_zero_diff_max_sdc, alternative="less")
            max_marker_sdc = "^*" if wilcoxon_max_sdc.pvalue < pvalue_threshold else ""
        else:
            max_marker_sdc = "^-"

        if len(non_zero_diff_avg_overall) >= 5:
            wilcoxon_avg_overall = stats.wilcoxon(
                non_zero_diff_avg_overall, alternative="less"
            )
            avg_marker_overall = (
                "^*" if wilcoxon_avg_overall.pvalue < pvalue_threshold else ""
            )
        else:
            avg_marker_overall = "^-"

        if len(non_zero_diff_max_overall) > 5:
            wilcoxon_max_overall = stats.wilcoxon(
                non_zero_diff_max_overall, alternative="less"
            )
            max_marker_overall = (
                "^*" if wilcoxon_max_overall.pvalue < pvalue_threshold else ""
            )
        else:
            max_marker_overall = "^-"

        # Format each baseline's results

        def plot_column(col):
            if col == "sdc_avg":
                marker = avg_marker_sdc
            elif col == "overall_avg":
                marker = avg_marker_overall
            elif col == "sdc_max":
                marker = max_marker_sdc
            elif col == "overall_max":
                marker = max_marker_overall
            else:
                marker = ""

            result = "{avg:.1f}{marker}".format(
                avg=100 * v[col],
                # std=100 * v[f"{b}_std"],
                marker=marker,
            )  # \\pm{std:.1f}
            if marker == "^*":
                result = "\\boldsymbol{" + result + "}"
            return f"${result}$"

        formatted_results = [plot_column(c) for c in columns]

        res = " & ".join(formatted_results)
        group_identifier = " & ".join([str(filter_dict[c]) for c in query_columns])
        print(f"{group_identifier} & {res} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")


# Plot per-system results
columns = ["sdc_avg", "overall_avg", "metric_avg", "average_avg"]
make_latex_table(["system"], columns)

# %%
## Print baseline table in latex
make_latex_table(["system", "num_performances"], columns)

# %%
# Plot single-metric results
make_latex_table(
    ["system", "performances"], columns, base_query="num_performances == 1"
)

# %%

rld = pd.DataFrame(result_list_dict)

sdc_max_better = (rld["sdc_max"] < rld["overall_max"]).mean()
sdc_max_better_by = (rld["sdc_max"] - rld["overall_max"]).mean()

print(f"SDC max better: {sdc_max_better:.2%} (by {sdc_max_better_by:.2})")

sdc_avg_better = (rld["sdc_avg"] < rld["overall_avg"]).mean()
sdc_avg_better_by = (rld["sdc_avg"] - rld["overall_avg"]).mean()

print(f"SDC avg better: {sdc_avg_better:.2%} (by {sdc_avg_better_by:.2})")

aggregate_over_all_systems = (
    rld[
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

wilcoxon_avg = stats.wilcoxon(rld["sdc_avg"], rld["overall_avg"], alternative="less")
ttest_avg = stats.ttest_ind(rld["sdc_avg"], rld["overall_avg"])
wilcoxon_max = stats.wilcoxon(rld["sdc_max"], rld["overall_max"], alternative="less")
ttest_max = stats.ttest_ind(rld["sdc_max"], rld["overall_max"])

print(f"Wilcoxon test (Avg. WCP SDC/Overall): {wilcoxon_avg.pvalue:.1e}")
print(f"T-test (Avg. WCP SDC/Overall): {ttest_avg.pvalue:.1e}")

print(f"Wilcoxon test (Max. WCP SDC/Overall): {wilcoxon_max.pvalue:.1e}")
print(f"T-test (Max. WCP SDC/Overall): {ttest_max.pvalue:.1e}")


# %%
