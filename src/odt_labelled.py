# %%
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydl85 import DL85Predictor
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from common import load_data, NpEncoder


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


def replace_feature_and_class_names(node, feature_names):
    if "feat" in node:
        node["feat"] = feature_names[node["feat"]]

    if "left" in node:
        node["left"] = replace_feature_and_class_names(
            node["left"], feature_names=feature_names
        )

    if "right" in node:
        node["right"] = replace_feature_and_class_names(
            node["right"], feature_names=feature_names
        )

    return node


def get_tree_json(clf, feature_names):
    # copy.deepcopy is important, otherwise we change the clf object
    tr = copy.deepcopy(clf.get_tree_without_transactions_and_probas())
    return replace_feature_and_class_names(tr, feature_names=feature_names)


def get_dl85_functions(y_inverse):
    def leaf_value_fn(tids):
        classes, supports = np.unique(y_inverse.take(list(tids)), return_counts=True)
        maxindex = np.argmax(supports)
        return classes[maxindex]

    def error_fn(tids):
        _, supports = np.unique(y_inverse.take(list(tids)), return_counts=True)
        maxindex = np.argmax(supports)
        return sum(supports) - supports[maxindex]

    return leaf_value_fn, error_fn


def train_dl85(X_train, y_train, max_depth):
    leaf_value_fn, error_fn = get_dl85_functions(y_train)
    clf = DL85Predictor(
        max_depth=max_depth,
        error_function=error_fn,
        leaf_value_function=leaf_value_fn,
        # time_limit=600,
    )
    clf.fit(X_train)
    return clf


def evaluate_result(res, splits, input_feature_columns, max_depth=None):
    num_configs = res["num_configs"]
    system = res["system"]
    performances = res["performances"]
    min_wcp_mean = res["wcp_mean"]
    min_wcp_max = res["wcp_max"]
    inp_cfg_map = res["input_to_config_map"]

    print("Loaded results:")
    print(f"System: {system}")
    print(f"Num. configs: {num_configs}")
    print(f"Performances: {performances}")
    print(f"wcp_mean: {min_wcp_mean:.4f}")
    print(f"wcp_max: {min_wcp_max:.4f}")

    perf_matrix = prepare_perf_matrx(perf_matrix_initial, performances)

    features = perf_matrix[input_feature_columns].drop_duplicates()
    inputnames = perf_matrix["inputname"].drop_duplicates()

    # Transform features
    # X = input_preprocessor.fit_transform(features)
    # feature_names = input_preprocessor.get_feature_names_out()

    # TODO This needs to be validated, n_bins is only used for real_cols
    # Maybe the input_preprocessor above is the better choice
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
    y_argmin = np.argmin(C, axis=1)

    # This is the label assignment from the OCS results
    y_all = np.array([inp_cfg_map[inp] for inp in inputnames])

    # Check if they agree
    label_agreement = (y_argmin == y_all).mean()
    print(f"label agreement: {label_agreement:.2f}")

    def evaluate(
        clf, train_inp, X_train, y_train, test_inp=None, X_test=None, y_test=None
    ):
        y_pred = clf.predict(X_train)
        train_acc = np.sum(y_pred == y_train) / y_train.shape[0]
        train_wcp_mean = evaluate_wcp_mean(y_pred, train_inp, C)
        train_wcp_max = evaluate_wcp_mean(y_pred, train_inp, C)
        num_classes_used = len(np.unique(y_pred))

        if test_inp is not None and X_test is not None and y_test is not None:
            test_pred = clf.predict(X_test)
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
            "num_classes_used": num_classes_used,
            "train_acc": train_acc,
            "test_acc": test_acc,
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
    for d in range(1, max_depth + 1):
        clf = train_dl85(X_train=X_all, y_train=y_all, max_depth=d)
        eval_result = evaluate(clf, train_inp=inputnames, X_train=X_all, y_train=y_all)
        results.append(
            {
                "model": "dl85",
                "system": system,
                "num_configs": num_configs,
                "performances": performances,
                "max_depth": d,
                "split": None,
                "train_time": clf.runtime_,
                "tree": get_tree_json(
                    clf=clf,
                    feature_names=X_all.columns,
                ),
                **eval_result,
            }
        )

        if eval_result["train_acc"] == 1.0:
            print(f"Stop at depth {d}")
            break

    # For CV, make another loop around the splits
    print("Training DL8.5 on train/test split")
    for split in splits:
        fold = split["fold"]
        train_inp = sorted(split["train_inputs"])
        test_inp = sorted(split["test_inputs"])

        assert all(inp in inputnames.values for inp in train_inp), (
            f"Train inputs not in perf matrix ({fold=})"
        )
        assert all(inp in inputnames.values for inp in test_inp), (
            f"Test inputs not in perf matrix ({fold=})"
        )

        X_train = X_all.loc[inputnames.isin(train_inp).values]
        X_test = X_all.loc[inputnames.isin(test_inp).values]
        y_train = np.array([inp_cfg_map[inp] for inp in train_inp])
        y_test = np.array([inp_cfg_map[inp] for inp in test_inp])

        for d in range(1, max_depth + 1):
            clf = train_dl85(X_train=X_train, y_train=y_train, max_depth=d)
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
                    "model": "dl85",
                    "system": system,
                    "num_configs": num_configs,
                    "performances": performances,
                    "max_depth": d,
                    "split": fold,
                    "train_time": clf.runtime_,
                    "tree": get_tree_json(
                        clf=clf,
                        feature_names=X_all.columns,
                    ),
                    **eval_result,
                }
            )

            if eval_result["train_acc"] == 1.0:
                print(f"Stop at depth {d}")
                break

    print("-" * 8)

    return results


# %%

system = "gcc"

(
    perf_matrix_initial,
    input_features,
    config_features,
    all_performances,
    input_preprocessor,
    config_preprocessor,
) = load_data(system=system, data_dir="../data", input_properties_type="tabular")

result_base_dir = Path("../results")
result_path = result_base_dir / f"ocs_{system}_both.json"
target_path = result_base_dir / result_path.name.replace("ocs_", "odt_")
best_configs = json.load(result_path.open("r"))

splits = json.load(open("../data/splits.json"))
splits = splits[system]

all_results = []

for res in best_configs:
    all_results.extend(
        evaluate_result(
            res, input_feature_columns=input_features.columns, splits=splits
        )
    )
    json.dump(
        all_results,
        target_path.open("w"),
        indent=2,
        cls=NpEncoder,
    )

# %%
