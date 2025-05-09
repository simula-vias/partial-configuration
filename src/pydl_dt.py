# %%
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from common import load_data
import json
import pandas as pd
from pydl85 import DL85Predictor

# from odtlearn.utils.binarize import binarize
import time
import time
import numpy as np
import pandas as pd
from pydl85 import DL85Predictor
from sklearn import tree
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from common import load_data
import json
from pathlib import Path
import graphviz


def evaluate_wcp(y_pred, C):
    # print("y_pred", y_pred)
    cost = C[np.arange(C.shape[0]), y_pred]
    return cost.max(), cost.mean()


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
        enc = OneHotEncoder(
            handle_unknown="error", drop="if_binary", max_categories=n_bins
        )
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


def train_sklearn_dt(
    X_train,
    y_train,
    max_depth=None,
):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


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


# TODO Merge this into `wip.py`; replace odtlearn because it is not cost-sensitive
# Evaluate whether the results actually differ or if picking the argmin label would have been okay, too.

random_state = 1234
test_size = 0.40

system = "gcc"
performance_str = "['exec']"
# performances = json.loads(performance_str.replace("'", '"'))
precision = 3
best_configs = json.load(open("../results/ocs_gcc_both.json"))

res = next(
    bc for bc in best_configs if bc["num_configs"] == 3
)  # best_configs[system][performance_str][str(num_configs)]
selected_configs = res["selected_configs"]

num_configs = res["num_configs"]
system = res["system"]
performances = res["performances"]
min_wcp_mean = int(res["wcp_mean"] * 1000 + 0.5)  # round(res["wcp_mean"], precision)
min_wcp_max = int(res["wcp_max"] * 1000 + 0.5)  # round(res["wcp_max"], precision)
configurations = res["selected_configs"]

(
    perf_matrix_initial,
    input_features,
    config_features,
    all_performances,
    input_preprocessor,
    config_preprocessor,
) = load_data(system=system, data_dir="../data", input_properties_type="tabular")

perf_matrix = prepare_perf_matrx(perf_matrix_initial, performances, configurations)


# Split data
train_inp, test_inp = train_test_split(
    perf_matrix["inputname"].unique(),
    test_size=test_size,
    random_state=random_state,
)

# This is only to make our evaluation simpler
train_inp = sorted(train_inp)
test_inp = sorted(test_inp)

train_perf = perf_matrix[perf_matrix.inputname.isin(train_inp)].copy()
test_perf = perf_matrix[perf_matrix.inputname.isin(test_inp)]

# features = train_perf[input_features.columns].drop_duplicates()
features = perf_matrix[input_features.columns].drop_duplicates()
inputnames = perf_matrix["inputname"].drop_duplicates()

# # Transform features
# X = input_preprocessor.fit_transform(features)
# feature_names = input_preprocessor.get_feature_names_out()

# # Create cost matrix for evaluation
# C = (
#     # perf_matrix[perf_matrix.inputname.isin(train_inp)][
#     perf_matrix[
#         ["inputname", "configurationID", "worst_case_performance"]
#     ]
#     .reset_index()
#     .pivot(
#         index="inputname", columns="configurationID", values="worst_case_performance"
#     )
#     .sort_values("inputname")
#     .reset_index()
#     .drop(columns=["inputname"])
#     .values
# )

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

# if max_depth is None:
#     max_depth = X_all.shape[1]

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

# # TODO compare with the odtree binarizer function
# discretizer = KBinsDiscretizer(n_bins=5, encode="onehot")
# X_bin = discretizer.fit_transform(X).toarray().astype(bool)


# %%


def leaf_value_fn(tids):
    classes, supports = np.unique(y_inverse.take(list(tids)), return_counts=True)
    maxindex = np.argmax(supports)
    return classes[maxindex]


def error_wcp_fn(tids):
    _, supports = np.unique(y_inverse.take(list(tids)), return_counts=True)
    maxindex = np.argmax(supports)
    return sum(supports) - supports[maxindex]


for max_depth in range(1, sklearn_clf.get_depth()):
    clf = DL85Predictor(
        max_depth=max_depth,
        error_function=error_wcp_fn,
        leaf_value_function=leaf_value_fn,
        # time_limit=600,
    )
    start = time.perf_counter()
    clf.fit(X_all)
    duration = time.perf_counter() - start
    clf_pred = clf.predict(X_all)
    print(clf_pred)
    cost_max, cost_mean = evaluate_wcp(clf_pred, Cmat)
    print(
        f"DL8.5 {max_depth} WCP max: {cost_max:.{precision}f}, mean: {cost_mean:.{precision}f}"
    )
    dot = clf.export_graphviz()
    graph = graphviz.Source(dot, format="png")
    graph.render(f"tree_{max_depth}")
# graph.view()


# %%
def wcp_max_mean(C_sub):
    # wcp_max_per_config = C_sub.max(axis=0)
    # wcp_mean_per_config = C_sub.mean(axis=0)
    # wcp_per_config = 1_000 * wcp_max_per_config + wcp_mean_per_config
    # wcp_sum = C_sub.sum(axis=0)
    max_term_coeff = 1000
    power = 2
    mean_term_coeff = 1
    C_sub_float = C_sub.astype(float)
    max_proxy_term = max_term_coeff * np.power(C_sub_float, power).sum(axis=0)

    # This term approximates N * mean_value_contribution
    sum_term = mean_term_coeff * C_sub_float.sum(axis=0)

    combined_error = max_proxy_term + sum_term
    return combined_error.round().astype(int)
    # return wcp_sum.round().astype(int)


def leaf_value_fn(tids):
    C_sub = Cmat[list(tids)]  # (inputs in node, configs)

    wcp_per_config = wcp_max_mean(C_sub)

    label = np.argmin(wcp_per_config)
    return label


def error_wcp_fn(tids):
    C_sub = Cmat[list(tids)]  # (inputs in node, configs)

    wcp_per_config = wcp_max_mean(C_sub)

    error = wcp_per_config.min()
    # print(error)
    return error


clf = DL85Predictor(
    max_depth=max_depth,
    error_function=error_wcp_fn,
    leaf_value_function=leaf_value_fn,
    # max_error=1000 * min_wcp_max + min_wcp_mean + 1,
    # stop_after_better=True,
)
start = time.perf_counter()
clf.fit(X_all)
duration = time.perf_counter() - start

cost_max, cost_mean = evaluate_wcp(clf.predict(X_all), Cmat)
print(
    f"DL8.5 {max_depth} WCP max: {cost_max:.{precision}f}, mean: {cost_mean:.{precision}f}"
)
# %%


def evaluate_cost(y_pred, C):
    # print("y_pred", y_pred)
    cost = C[np.arange(C.shape[0]), y_pred]
    return cost.max(), cost.mean()


def wcp_max_mean(C_sub):
    wcp_max_per_config = C_sub.max(axis=0)
    wcp_mean_per_config = C_sub.mean(axis=0)
    wcp_per_config = 10_000 * wcp_max_per_config.round(4) + wcp_mean_per_config
    return wcp_per_config


def wcp_mean(C_sub):
    return C_sub.mean(axis=0)


def wcp_max(C_sub):
    return C_sub.max(axis=0)


def get_dt_functions(mode, C, allowed_configs):
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

    if allowed_configs is None:
        allowed_configs = np.arange(C.shape[1])

    C_allowed = C[:, allowed_configs]

    def leaf_value_fn(tids):
        C_sub = C_allowed[list(tids)]  # (inputs in node, configs)

        wcp_per_config = wcp_fn(C_sub)

        label_offset = np.argmin(wcp_per_config)
        label = allowed_configs[label_offset]
        return label

    def error_wcp_fn(tids):
        C_sub = C_allowed[list(tids)]  # (inputs in node, configs)

        wcp_per_config = wcp_fn(C_sub)

        error = wcp_per_config.min()
        return error

    return leaf_value_fn, error_wcp_fn


optimization_target = res["optimization_target"]
leaf_value_fn, error_fn = get_dt_functions(optimization_target, C, selected_configs)

for d in range(1, X_bin.shape[1]):
    clf = DL85Predictor(
        max_depth=d,
        error_function=error_fn,
        leaf_value_function=leaf_value_fn,
        time_limit=600,
    )
    start = time.perf_counter()
    clf.fit(X_bin)
    duration = time.perf_counter() - start

    cost_max, cost_mean = evaluate_cost(clf.predict(X_bin), C)
    print(
        f"Depth={d} wcp_max={cost_max:.5f} wcp_mean={cost_mean:.5f} Duration={duration:.4f}"
    )
    tree = clf.get_tree_without_transactions_and_probas()
    num_nodes = clf.get_nodes_count()

    # TODO Check tolerance values
    if np.isclose(cost_max, res["wcp_max"]) and np.isclose(cost_mean, res["wcp_mean"]):
        print(
            f"Depth {d}: Reached optimal values (wcp_max={cost_max:.4f}/{res['wcp_max']:.4f} / wcp_mean={cost_mean:.4f}/{res['wcp_mean']:.4f})"
        )
        break
else:
    print("Did not reach optimal values")


# %%
