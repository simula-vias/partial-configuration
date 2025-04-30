# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from odtlearn.flow_oct import FlowOCT
from odtlearn.utils.binarize import binarize

from common import load_data
import json
from pathlib import Path


def prepare_perf_matrx(perf_matrix_initial, performances):
    nmdf = (
        perf_matrix_initial[["inputname"] + performances]
        .groupby("inputname", as_index=True)
        # .transform(lambda x: scale(x))
        .transform(lambda x: (x - x.min()) / (x.max() - x.min()))
    )
    nmdf["worst_case_performance"] = nmdf[performances].max(axis=1)
    perf_matrix = pd.merge(
        perf_matrix_initial,
        nmdf,
        suffixes=("_raw", None),
        left_index=True,
        right_index=True,
    )
    # We adjust the WCP by expressing it as the difference from the best WCP, i.e. the best WCP is always 0
    perf_matrix["worst_case_performance"] = (
        perf_matrix[["inputname", "worst_case_performance"]]
        .groupby("inputname", as_index=True)
        .transform(lambda x: x - x.min())
    )
    return perf_matrix


def evaluate_wcp_mean(y_pred, inputs, C):
    total_cost = 0
    for inp, pred in zip(inputs, y_pred):
        total_cost += C.loc[inp][pred]
    return total_cost / len(y_pred)


def evaluate_wcp_max(y_pred, inputs, C):
    return max(C.loc[inp][pred] for inp, pred in zip(inputs, y_pred))


def train_flow_oct(
    X_train,
    y_train,
    max_depth=2,
    solver="cbc",
    time_limit=100,
):
    stcl = FlowOCT(depth=max_depth, solver=solver, time_limit=time_limit, verbose=False)

    stcl.fit(X_train, y_train)
    return stcl


def train_sklearn_dt(
    X_train,
    y_train,
    max_depth=2,
):
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


def evaluate_result(res, train_inp, test_inp, input_feature_columns, max_depth=None):
    num_configs = res["num_configs"]
    system = res["system"]
    performances = res["performances"]
    min_wcp_mean = res["wcp_mean"]
    min_wcp_max = res["wcp_max"]

    print("Loaded results:")
    print(f"System: {system}")
    print(f"Num configs: {num_configs}")
    print(f"Performances: {performances}")
    print(f"wcp_mean: {res['wcp_mean']}")
    print(f"wcp_max: {res['wcp_max']}")

    perf_matrix = prepare_perf_matrx(perf_matrix_initial, performances)

    features = perf_matrix[input_feature_columns].drop_duplicates()
    inputnames = perf_matrix["inputname"].drop_duplicates()

    # Transform features
    # X = input_preprocessor.fit_transform(features)
    # feature_names = input_preprocessor.get_feature_names_out()

    # TODO Validate for DT
    X_all = binarize(
        features,
        categorical_cols=[],
        integer_cols=features.columns,
        real_cols=[],
        n_bins=5,
    )

    if max_depth is None:
        max_depth = X_all.shape[1]

    # Create cost matrix for evaluation
    C = (
        perf_matrix[["inputname", "configurationID", "worst_case_performance"]]
        .reset_index()
        .pivot(
            index="inputname",
            columns="configurationID",
            values="worst_case_performance",
        )
        .sort_values("inputname")
    )
    # Label each input with the configuration with the lowest WCP for reference
    # y_argmin = np.argmin(C, axis=1)

    inp_cfg_map = res["input_to_config_map"]

    X_train = X_all.loc[inputnames.isin(train_inp).values]
    X_test = X_all.loc[inputnames.isin(test_inp).values]

    y_train = np.array([inp_cfg_map[inp] for inp in train_inp])
    y_test = np.array([inp_cfg_map[inp] for inp in test_inp])
    y_all = np.array([inp_cfg_map[inp] for inp in inputnames])

    # Print class distribution
    print("Class distribution:")
    unique_classes, class_counts = np.unique(y_all, return_counts=True)
    for c, count in zip(unique_classes, class_counts):
        print(f"Class {c}: {count} ({100 * count / len(y_all):.2f}%)")

    def evaluate(
        clf, train_inp, X_train, y_train, test_inp=None, X_test=None, y_test=None
    ):
        y_pred = clf.predict(X_train)
        train_acc = np.sum(y_pred == y_train) / y_train.shape[0]
        train_wcp_mean = evaluate_wcp_mean(y_pred, train_inp, C)
        train_wcp_max = evaluate_wcp_mean(y_pred, train_inp, C)
        num_classes_used = len(np.unique(y_pred))

        if test_inp is not None and X_test is not None and y_test is not None:
            test_pred = clf.predict(X_test).tolist()
            test_acc = np.sum(test_pred == y_test) / y_test.shape[0]
            test_wcp_mean = evaluate_wcp_mean(test_pred, test_inp, C)
            test_wcp_max = evaluate_wcp_max(test_pred, test_inp, C)
            test_wcp_mean_gap = test_wcp_mean / min_wcp_mean
            test_wcp_max_gap = test_wcp_max / min_wcp_max
        else:
            test_pred = None
            test_acc = None
            test_wcp_mean = None
            test_wcp_max = None
            test_wcp_mean_gap = None
            test_wcp_max_gap = None

        return {
            "model": "flow_oct",
            "num_classes_used": num_classes_used,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_wcp_mean": train_wcp_mean,
            "train_wcp_max": train_wcp_max,
            "test_wcp_mean": test_wcp_mean,
            "test_wcp_max": test_wcp_max,
            "train_pred": y_pred.tolist(),
            "test_pred": test_pred,
            "train_wcp_mean_gap": train_wcp_mean / min_wcp_mean,
            "train_wcp_max_gap": train_wcp_max / min_wcp_max,
            "test_wcp_mean_gap": test_wcp_mean_gap,
            "test_wcp_max_gap": test_wcp_max_gap,
        }

    results = []

    print("Training FlowOCT on full dataset")
    for d in range(1, max_depth + 1):
        clf = train_flow_oct(X_train=X_all, y_train=y_all, max_depth=d)
        eval_result = evaluate(clf, train_inp=inputnames, X_train=X_all, y_train=y_all)
        results.append(
            {
                "system": system,
                "num_configs": num_configs,
                "performances": performances,
                "max_depth": d,
                "split": False,
                **eval_result,
            }
        )

        if eval_result["train_acc"] == 1.0:
            break

    print("Training FlowOCT on train/test split")
    for d in range(1, max_depth + 1):
        #
        clf = train_flow_oct(X_train=X_train, y_train=y_train, max_depth=d)
        eval_result = evaluate(
            clf,
            train_inp=train_inp,
            X_train=X_train,
            y_train=y_train,
            test_inp=test_inp,
            X_test=X_test,
            y_test=y_test,
        )
        results.append(
            {
                "system": system,
                "num_configs": num_configs,
                "performances": performances,
                "max_depth": d,
                "split": True,
                **eval_result,
            }
        )

        if eval_result["train_acc"] == 1.0:
            break

    print("Training sklearn Decision Tree on full dataset")
    for d in range(1, max_depth + 1):
        clf = train_sklearn_dt(X_train=X_all, y_train=y_all, max_depth=d)
        eval_result = evaluate(clf, train_inp=inputnames, X_train=X_all, y_train=y_all)
        results.append(
            {
                "system": system,
                "num_configs": num_configs,
                "performances": performances,
                "max_depth": d,
                "split": False,
                **eval_result,
            }
        )

        if eval_result["train_acc"] == 1.0:
            break

    print("Training sklearn Decision Tree on train/test split")
    for d in range(1, max_depth + 1):
        clf = train_sklearn_dt(X_train=X_train, y_train=y_train, max_depth=d)
        eval_result = evaluate(
            clf,
            X_train=X_train,
            train_inp=train_inp,
            y_train=y_train,
            test_inp=test_inp,
            X_test=X_test,
            y_test=y_test,
        )
        results.append(
            {
                "system": system,
                "num_configs": num_configs,
                "performances": performances,
                "max_depth": d,
                "split": True,
                **eval_result,
            }
        )

        if eval_result["train_acc"] == 1.0:
            break

    return results


# %%

random_state = 1234
test_size = 0.40

system = "gcc"
# performance_str = "['exec']"
# num_configs = 4
# performances = json.loads(performance_str.replace("'", '"'))  # WTF?


(
    perf_matrix_initial,
    input_features,
    config_features,
    all_performances,
    input_preprocessor,
    config_preprocessor,
) = load_data(system=system, data_dir="../data", input_properties_type="tabular")

result_path = Path("../results") / f"ocs_{system}_both.json"
target_path = Path("../results") / result_path.name.replace("ocs_", "odt_")
best_configs = json.load(open(result_path, "r"))

splits = json.load(open("../data/splits.json"))
splits = splits[system]

# For CV, make another loop around the splits
train_inp = splits[0]["train_inputs"]
test_inp = splits[0]["test_inputs"]

assert all(inp in perf_matrix_initial["inputname"].unique() for inp in train_inp), (
    "Train inputs not in perf matrix"
)
assert all(inp in perf_matrix_initial["inputname"].unique() for inp in test_inp), (
    "Test inputs not in perf matrix"
)

# This is only to make our evaluation simpler
train_inp = sorted(train_inp)
test_inp = sorted(test_inp)

all_results = []

for res in best_configs:
    all_results.extend(
        evaluate_result(
            res, train_inp, test_inp, input_feature_columns=input_features.columns
        )
    )
    json.dump(
        all_results,
        open(target_path, "w"),
        indent=4,
    )

# %%

# TODO List
# Measure GAP
# Check if we can run multiprocessing
# Save results

# %%
