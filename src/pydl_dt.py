# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from common import load_data
import json
import pandas as pd
from pydl85 import DL85Predictor
import time

# TODO Merge this into `wip.py`; replace odtlearn because it is not cost-sensitive
# Evaluate whether the results actually differ or if picking the argmin label would have been okay, too.

random_state = 1234
test_size = 0.40

system = "gcc"
performance_str = "['exec']"
num_configs = 2
performances = json.loads(performance_str.replace("'", '"'))

best_configs = json.load(open("../results/ocs_results.json"))

res = best_configs[system][performance_str][str(num_configs)]
selected_configs = res["selected_configs"]

(
    perf_matrix_initial,
    input_features,
    config_features,
    all_performances,
    input_preprocessor,
    config_preprocessor,
) = load_data(system=system, data_dir="../data", input_properties_type="tabular")

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

# Transform features
X = input_preprocessor.fit_transform(features)
feature_names = input_preprocessor.get_feature_names_out()

# Create cost matrix for evaluation
C = (
    # perf_matrix[perf_matrix.inputname.isin(train_inp)][
    perf_matrix[
        ["inputname", "configurationID", "worst_case_performance"]
    ]
    .reset_index()
    .pivot(
        index="inputname", columns="configurationID", values="worst_case_performance"
    )
    .sort_values("inputname")
    .reset_index()
    .drop(columns=["inputname"])
    .values
)

# %%

# TODO compare with the odtree binarizer function
discretizer = KBinsDiscretizer(n_bins=5, encode="onehot")
X_bin = discretizer.fit_transform(X).toarray().astype(bool)


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
    if mode == "max_mean" or mode =="both":
        wcp_fn = wcp_max_mean
    elif mode == "max":
        wcp_fn = wcp_max
    elif mode == "mean":
        wcp_fn = wcp_mean
    else:
        raise ValueError("Invalid mode. Choose 'max_mean' (or 'both'), 'max', or 'mean'.")

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
    print(f"Depth={d} wcp_max={cost_max:.5f} wcp_mean={cost_mean:.5f} Duration={duration:.4f}")
    tree = clf.get_tree_without_transactions_and_probas()
    num_nodes = clf.get_nodes_count()

    # TODO Check tolerance values
    if np.isclose(cost_max, res["wcp_max"]) and np.isclose(cost_mean, res["wcp_mean"]):
        print(f"Depth {d}: Reached optimal values (wcp_max={cost_max:.4f}/{res['wcp_max']:.4f} / wcp_mean={cost_mean:.4f}/{res['wcp_mean']:.4f})")
        break
else:
    print("Did not reach optimal values")


# %%
