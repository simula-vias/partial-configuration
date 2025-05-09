# %%
import time
import numpy as np
import pandas as pd
from odtlearn.utils.binarize import binarize
from pydl85 import DL85Predictor
from sklearn import tree
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from common import load_data
import json
from pathlib import Path
import graphviz


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def prepare_perf_matrx(perf_matrix_initial, performances, configurations=None):
    nmdf = (
        perf_matrix_initial[["inputname"] + performances]
        .groupby("inputname", as_index=True)
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

    if configurations is not None:
        perf_matrix = perf_matrix[perf_matrix["configurationID"].isin(configurations)]

    return perf_matrix


def evaluate_wcp_mean(y_pred, inputs, C):
    total_cost = 0
    for inp, pred in zip(inputs, y_pred):
        total_cost += C.loc[inp][pred]
    return total_cost / len(y_pred)


def evaluate_wcp_max(y_pred, inputs, C):
    return max(C.loc[inp][pred] for inp, pred in zip(inputs, y_pred))


def evaluate_wcp(y_pred, C):
    # print("y_pred", y_pred)
    cost = C[np.arange(C.shape[0]), y_pred]
    return cost.max(), cost.mean()


def get_dl85_functions(mode, C):
    def wcp_max_mean(C_sub):
        wcp_max_per_config = C_sub.max(axis=0)
        wcp_mean_per_config = C_sub.mean(axis=0)
        wcp_per_config = 1_000 * wcp_max_per_config + wcp_mean_per_config
        return wcp_per_config.astype(int)

    def wcp_mean(C_sub):
        return C_sub.mean(axis=0)

    def wcp_max(C_sub):
        return C_sub.max(axis=0)

    if mode == "max_mean" or mode == "both":
        wcp_fn = wcp_max_mean
    elif mode == "max":
        wcp_fn = wcp_max
    elif mode == "mean":
        wcp_fn = wcp_mean
    else:
        raise ValueError(
            "Invalid mode. Choose 'max_mean' (or 'both'), 'max', or 'mean'."
        )

    def leaf_value_fn(tids):
        C_sub = C[list(tids)]  # (inputs in node, configs)

        wcp_per_config = wcp_fn(C_sub)

        label = np.argmin(wcp_per_config)
        return label

    def error_wcp_fn(tids):
        C_sub = C[list(tids)]  # (inputs in node, configs)

        wcp_per_config = wcp_fn(C_sub)

        error = wcp_per_config.min()
        return error

    return leaf_value_fn, error_wcp_fn


def train_dl85(X_train, C, max_depth=2, time_limit=600, max_error=0):
    leaf_value_fn, error_fn = get_dl85_functions(
        "max_mean",
        C,
    )

    print(
        f"Training DL8.5 with max_depth={max_depth}, time_limit={time_limit}, max_error={max_error}"
    )
    clf = DL85Predictor(
        max_depth=max_depth,
        error_function=error_fn,
        leaf_value_function=leaf_value_fn,
        time_limit=time_limit,
        # max_error=max_error,
        # stop_after_better=True,
        # verbose=True,
        # print_output=True,
    )
    clf.fit(X_train)
    return clf


def train_sklearn_dt(
    X_train,
    y_train,
    max_depth=None,
):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


# From odtlearn, installatin via pip failed somehow
def binarize(
    df, categorical_cols, integer_cols, real_cols, n_bins=4, bin_strategy="uniform"
):
    """
    Parameters
    ----------
    df: pandas dataframe
        A dataframe with only categorical/integer columns. There should not be any NA values.
    categorical_cols: list
        A list consisting of the names of categorical columns of df
    integer_cols: list
        A list consisting of the names of integer columns of df
    real_cols: list
        A list consisting of the names of real columns of df
    n_bins: int
        The number of bins to be used for encoding real columns
    bin_strategy: str
        The strategy to be used for discretizing real columns. It can be one of the following:
        'uniform': All bins in each feature have identical widths.
        'quantile': All bins in each feature have the same number of points.

    Return
    ----------
    the binarized version of the input dataframe.

    This function encodes each categorical column as a one-hot vector, i.e.,
    for each level of the feature, it creates a new binary column with a value
    of one if and only if the original column has the corresponding level.
    A similar approach for encoding integer features is used with a slight change.
    The new binary column should have a value of one if and only if the main column
    has the corresponding value or any value smaller than it.
    We first discretize the real columns according to the number of bins and the discretization strategy
    and then treat them as integer columns.
    """

    assert len(categorical_cols) > 0 or len(integer_cols) > 0 or len(real_cols) > 0, (
        "Must provide at least one of the three options of a list of categorical columns "
        "or integer columns or real valued columns to binarize."
    )

    if len(real_cols) > 0 and n_bins is None:
        raise ValueError(
            "The number of bins must be provided for encoding real columns."
        )
    if len(real_cols) > 0 and bin_strategy is None:
        raise ValueError("The bin strategy must be provided for encoding real columns.")
    if (
        len(real_cols) > 0
        and bin_strategy is None
        or bin_strategy not in ["uniform", "quantile"]
    ):
        raise ValueError(
            "The bin strategy must be one of the following: 'uniform' or 'quantile'."
        )

    if len(categorical_cols) > 0:
        X_cat = np.array(df[categorical_cols])
        enc = OneHotEncoder(handle_unknown="error", drop="if_binary")
        X_cat_enc = enc.fit_transform(X_cat).toarray()
        categorical_cols_enc = enc.get_feature_names_out(categorical_cols)
        X_cat_enc = X_cat_enc.astype(int)
    if len(real_cols) > 0:
        # We first bucketize the real columns according to number of bins and the
        # discretization strategy and then treat them as integer columns
        discretizer_unif = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy=bin_strategy
        )
        df[real_cols] = discretizer_unif.fit_transform(df[real_cols])
        integer_cols = integer_cols + real_cols
    if len(integer_cols) > 0:
        X_int = np.array(df[integer_cols])
        enc = OneHotEncoder(handle_unknown="error", drop="if_binary")
        X_int_enc = enc.fit_transform(X_int).toarray()
        integer_cols_enc = enc.get_feature_names_out(integer_cols)
        X_int_enc = X_int_enc.astype(int)

        for col in integer_cols:
            col_enc_set = []
            col_offset = None
            for i, col_enc in enumerate(integer_cols_enc):
                if col in col_enc:
                    col_enc_set.append(col_enc)
                    if col_offset is None:
                        col_offset = i
            if len(col_enc_set) < 3:
                continue
            for i, col_enc in enumerate(col_enc_set):
                if i == 0:
                    continue
                X_int_enc[:, (col_offset + i)] = (
                    X_int_enc[:, (col_offset + i)] | X_int_enc[:, (col_offset + i - 1)]
                )
    if len(categorical_cols) > 0 and len(integer_cols) > 0:
        df_enc = pd.DataFrame(
            np.c_[X_cat_enc, X_int_enc],
            columns=list(categorical_cols_enc) + list(integer_cols_enc),
        )
    elif len(categorical_cols) > 0 and len(integer_cols) == 0:
        df_enc = pd.DataFrame(
            X_cat_enc,
            columns=list(categorical_cols_enc),
        )
    elif len(categorical_cols) == 0 and len(integer_cols) > 0:
        df_enc = pd.DataFrame(
            X_int_enc,
            columns=list(integer_cols_enc),
        )
    return df_enc


def evaluate_result(res, train_inp, test_inp, input_feature_columns, max_depth=None):
    precision = 3

    num_configs = res["num_configs"]
    system = res["system"]
    performances = res["performances"]
    min_wcp_mean = int(
        res["wcp_mean"] * 1000 + 0.5
    )  # round(res["wcp_mean"], precision)
    min_wcp_max = int(res["wcp_max"] * 1000 + 0.5)  # round(res["wcp_max"], precision)
    configurations = res["selected_configs"]

    print("Loaded results:")
    print(f"System: {system}")
    print(f"Num configs: {num_configs}")
    print(f"Performances: {performances}")
    print(f"wcp_mean: {min_wcp_mean}")
    print(f"wcp_max: {min_wcp_max}")

    perf_matrix = prepare_perf_matrx(perf_matrix_initial, performances, configurations)

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
    # TODO Handle better
    # discretizer = KBinsDiscretizer(n_bins=5, encode="onehot")
    # X_all = discretizer.fit_transform(features).toarray().astype(bool)

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
        .round(3)
    )
    C = (C * 1000).astype(int)
    Cmat = C.reset_index().drop(columns=["inputname"]).values
    # Cmat = Cmat.astype(int)

    print(Cmat)

    assert Cmat.shape[1] == num_configs, Cmat.shape

    input_mask_all = inputnames.isin(inputnames).values
    input_mask_train = inputnames.isin(train_inp).values
    input_mask_test = inputnames.isin(test_inp).values

    X_train = X_all.loc[input_mask_train]
    X_test = X_all.loc[input_mask_test]

    inp_cfg_map = res["input_to_config_map"]
    y_train = np.array([inp_cfg_map[inp] for inp in train_inp])
    y_test = np.array([inp_cfg_map[inp] for inp in test_inp])
    y_all = np.array([inp_cfg_map[inp] for inp in inputnames])

    def evaluate(clf, input_mask_train, input_mask_test, X_train, C, X_test=None):
        y_pred = clf.predict(X_train)
        train_wcp_max, train_wcp_mean = evaluate_wcp(y_pred, C[input_mask_train].values)
        num_classes_used = len(np.unique(y_pred))

        if input_mask_test is not None and X_test is not None:
            test_pred = clf.predict(X_test)
            test_wcp_max, test_wcp_mean = evaluate_wcp(
                test_pred, C[input_mask_test].values
            )
            test_wcp_mean_gap = test_wcp_mean / min_wcp_mean
            test_wcp_max_gap = test_wcp_max / min_wcp_max
        else:
            test_pred = None
            test_wcp_mean = None
            test_wcp_max = None
            test_wcp_mean_gap = None
            test_wcp_max_gap = None

        return {
            "num_classes_used": num_classes_used,
            "train_wcp_mean": train_wcp_mean,
            "train_wcp_max": train_wcp_max,
            "test_wcp_mean": test_wcp_mean,
            "test_wcp_max": test_wcp_max,
            "train_pred": y_pred,
            "test_pred": test_pred,
            "train_wcp_mean_gap": train_wcp_mean / min_wcp_mean,
            "train_wcp_max_gap": train_wcp_max / min_wcp_max,
            "test_wcp_mean_gap": test_wcp_mean_gap,
            "test_wcp_max_gap": test_wcp_max_gap,
        }

    results = []

    print("Training DL8.5 on full dataset")

    y_uniq, y_inverse = np.unique(y_all, return_inverse=True)
    sklearn_clf = train_sklearn_dt(X_all, y_inverse)
    sklearn_pred = sklearn_clf.predict(X_all).tolist()
    sklearn_wcp_max, sklearn_wcp_mean = evaluate_wcp(sklearn_pred, Cmat)
    print(
        f"Sklearn {sklearn_clf.get_depth()} WCP max: {sklearn_wcp_max:.{precision}f}, mean: {sklearn_wcp_mean:.{precision}f}"
    )
    print(sklearn_pred)

    max_depth = sklearn_clf.get_depth()

    # print(json.dumps(sklearn_clf.tree_, indent=2, cls=NpEncoder))
    tree.plot_tree(sklearn_clf)
    plt.show()
    # last_error = 694.0118
    last_error = 0
    for d in [max_depth] + list(range(1, max_depth + 1)):
        # TODO We can set the max_error parameter for subsequent calls
        print(f"Training DL8.5 on full dataset with depth {d}")
        clf = train_dl85(
            X_train=X_all, C=Cmat, max_depth=d, max_error=last_error, time_limit=0
        )
        train_time = clf.runtime_
        print("Evaluating on full dataset")
        eval_result = evaluate(
            clf,
            input_mask_train=input_mask_all,
            input_mask_test=None,
            X_train=X_all,
            C=C,
        )
        results.append(
            {
                "model": "dl8.5",
                "system": system,
                "num_configs": num_configs,
                "performances": performances,
                "max_depth": d,
                "split": False,
                "train_time": train_time,
                "tree": clf.tree_,
                **eval_result,
            }
        )
        last_error = 0  # clf.error_

        print(
            f"{train_time=:.2f}s (wcp_max={eval_result['train_wcp_max']:.{precision}f}/{min_wcp_max:.{precision}f} / wcp_mean={eval_result['train_wcp_mean']:.{precision}f}/{min_wcp_mean:.{precision}f})"
        )
        print("")
        print(json.dumps(clf.tree_, indent=2, cls=NpEncoder))
        dot = clf.export_graphviz()
        graph = graphviz.Source(dot, format="png")
        graph.render(f"tree_{d}")

        break
        if np.isclose(
            eval_result["train_wcp_max"], res["wcp_max"], atol=1e-3
        ) and np.isclose(eval_result["train_wcp_mean"], res["wcp_mean"], atol=1e-3):
            print(
                f"Depth {d}: Reached optimal values (wcp_max={eval_result['train_wcp_max']:.{precision}f}/{min_wcp_max:.{precision}f} / wcp_mean={eval_result['train_wcp_mean']:.{precision}f}/{min_wcp_mean:.{precision}f})"
            )
            break
    else:
        print(
            f"Did not reach optimal values (wcp_max={eval_result['train_wcp_max']:.{precision}f}/{min_wcp_max:.{precision}f} / wcp_mean={eval_result['train_wcp_mean']:.{precision}f}/{min_wcp_mean:.{precision}f})"
        )

    # print("Training DL8.5 on train/test split")
    # for d in range(1, max_depth + 1):
    #     #
    #     clf = train_dl85(X_train=X_train, C=Cmat, max_depth=d)
    #     eval_result = evaluate(
    #         clf,
    #         input_mask_train=input_mask_train,
    #         input_mask_test=input_mask_test,
    #         X_train=X_train,
    #         C=C,
    #         X_test=X_test,
    #     )
    #     results.append(
    #         {
    #             "model": "dl8.5",
    #             "system": system,
    #             "num_configs": num_configs,
    #             "performances": performances,
    #             "max_depth": d,
    #             "split": True,
    #             **eval_result,
    #         }
    #     )

    #     # TODO Here we must use the CV results as reference
    #     if np.isclose(eval_result["train_wcp_max"], res["wcp_max"]) and np.isclose(
    #         eval_result["train_wcp_mean"], res["wcp_mean"]
    #     ):
    #         print(
    #             f"Depth {d}: Reached optimal values (wcp_max={eval_result['train_wcp_max']:.{precision}f}/{res['wcp_max']:.{precision}f} / wcp_mean={eval_result['train_wcp_mean']:.{precision}f}/{res['wcp_mean']:.{precision}f})"
    #         )
    #         break
    # else:
    #     print(
    #         f"Did not reach optimal values (wcp_max={eval_result['train_wcp_max']:.{precision}f}/{res['wcp_max']:.{precision}f} / wcp_mean={eval_result['train_wcp_mean']:.{precision}f}/{res['wcp_mean']:.{precision}f})"
    #     )

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
    if res["num_configs"] < 3:
        continue

    all_results.extend(
        evaluate_result(
            res, train_inp, test_inp, input_feature_columns=input_features.columns
        )
    )
    json.dump(
        all_results,
        open(target_path, "w"),
        indent=4,
        cls=NpEncoder,
    )
    break

# %%

# TODO List
# Measure GAP
# Check if we can run multiprocessing
# Save results

# %%
