import argparse
import json
from pathlib import Path

import gosdt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from pydl85 import DL85Classifier

from common import (
    baseline_results_wc,
    get_leaf_values,
    load_data,
)

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Create the parser
parser = argparse.ArgumentParser(
    description="Train decision tree for worst-case performance."
)

# TODO
# - Switch gosdt to pydl85, _if_ runnable with max depth
# - Assess max depth parameter from trained models for all classifiers


# Add the arguments
parser.add_argument("--system", type=str, required=True, help="The system name.")
parser.add_argument(
    "--performances",
    type=str,
    required=True,
    help="A comma-separated list of performance measures.",
)
parser.add_argument(
    "--classifier",
    type=str,
    choices=["dt", "rf", "gosdt", "dl85"],
    default="gosdt",
    help="Classifier to use: dt, rf, gosdt, dl85 (default: gosdt).",
)
parser.add_argument(
    "--seed", type=int, default=1234, help="Random seed (default: 1234)."
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="./data",
    help="Directory containing the data (default: ./data).",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="./results",
    help="Directory to store the results (default: ./results).",
)

# Parse the arguments
args = parser.parse_args()

s = args.system
all_performances = sorted(args.performances.split(","))
random_state = args.seed
results_dir = Path(args.results_dir)

data_dir = Path(args.data_dir)
classifier = args.classifier

result_list_dict = []

print(s)
(
    perf_matrix_initial,
    input_features,
    config_features,
    all_performances_initial,
    input_preprocessor,
    config_preprocessor,
) = load_data(system=s, data_dir=data_dir)


num_p = len(all_performances)
print(f"start {num_p} perf: {all_performances}")

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
splits = 4
kf_inp = KFold(n_splits=splits, random_state=random_state, shuffle=True)

inputnames = perf_matrix["inputname"].unique()

## Baselines
for split_idx, (train_inp_idx, test_inp_idx) in enumerate(kf_inp.split(inputnames)):
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
        return icm_test.merge(inp_pred_map, on=["inputname", "configurationID"])[
            "worst_case_performance"
        ]  # .mean()

    ## HERE GOES SDC
    cfg_columns = ["configurationID"] + list(config_features.columns)
    top_cfgs = (
        dataset[cfg_columns + ["inputname"]]
        .groupby(cfg_columns, dropna=False, as_index=False)
        .count()
        .sort_values("inputname", ascending=False)
        .configurationID.tolist()
    )

    # Label each input with the best configuration from the training set
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

    if classifier in ("dt", "rf"):
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
    elif classifier == "gosdt":
        # Doesn't work with GridSearchCV for me
        # I receive segmentation fault errors
        # parameter_grid = {
        #     "regularization": [0.1, 1, 10],
        #     "depth_budget": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        # }
        clf = gosdt.GOSDTClassifier(
            regularization=max(1 / X.shape[0], 0.05), verbose=False
        )
    elif classifier == "dl85":
        # Does not support sklearn GridSearchCV
        # parameter_grid = {
        #     "max_depth": range(1, X.shape[1] + 1),
        #     "max_leaves": range(2, 100),
        #     "time_limit": [300],
        # }
        clf = DL85Classifier(max_depth=X.shape[1], maxcachesize=200_000_000)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")

    clf.fit(X, y)

    split_result = {
        "system": s,
        "split": split_idx,
        "classifier": classifier,
        "performances": "-".join(all_performances),
        # **clf.best_params_,
    }

    X_test = input_preprocessor.transform(
        input_features.query("inputname.isin(@test_inp)")
    )
    pred_cfg = enc.inverse_transform(clf.predict(X_test)).astype(int)

    sdc_wcp = eval_prediction(pred_cfg)

    if classifier == "dt":
        num_predictable_configs = clf.best_estimator_.n_classes_
        max_depth = clf.best_estimator_.get_depth()
    elif classifier == "rf":
        num_predictable_configs = clf.best_estimator_.n_classes_
        max_depth = max(est.get_depth() for est in clf.best_estimator_.estimators_)
    elif classifier == "gosdt":
        num_predictable_configs = clf.n_classes_
        max_depth = 0
    elif classifier == "dl85":
        num_predictable_configs = clf.classes_.shape[0]
        max_depth = clf.depth_

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

    split_result["max_depth"] = float(max_depth / X.shape[1])
    split_result["num_predictable_configs"] = int(num_predictable_configs)
    split_result["num_total_configs"] = int(num_total_configs)
    split_result["num_performances"] = int(num_p)

    # print(json.dumps(split_result, indent=2, sort_keys=True))

    result_list_dict.append(split_result)

json.dump(
    result_list_dict,
    open(
        results_dir / f"wcp_1cfg_{classifier}_{s}_{'-'.join(all_performances)}.json",
        "w",
    ),
)

print("Done.")
